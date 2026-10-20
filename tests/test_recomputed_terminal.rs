use std::sync::Arc;

use axum::{
    http::{Method, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use vllm_router_rs::{
    config::{KvConnector, PolicyConfig, RouterConfig, RoutingMode},
    routers::RouterFactory,
};

#[tokio::test]
async fn pd_recomputed_is_not_success_and_normal_stops_survive() {
    async fn prefill() -> Json<Value> {
        Json(json!({"choices":[], "kv_transfer_params":{
            "do_remote_prefill":true, "do_remote_decode":false,
            "remote_engine_id":"mock", "remote_block_ids":[1]
        }}))
    }
    async fn decode(Json(body): Json<Value>) -> axum::response::Response {
        let interrupted = body["model"] == "interrupted";
        let mut choice = json!({"index":0,"finish_reason":"stop","message":{"content":"A"}});
        if interrupted {
            choice["stop_reason"] = json!("recomputed");
        }
        let mut response = if body["stream"] == true {
            choice.as_object_mut().unwrap().remove("message");
            choice["delta"] = json!({"content":"A"});
            let event = format!(
                "data: {}\r\n\r\ndata: [DONE]\r\n\r\n",
                json!({"choices":[choice]})
            );
            ([("content-type", "text/event-stream")], event).into_response()
        } else {
            Json(json!({"choices":[choice]})).into_response()
        };
        response
            .headers_mut()
            .insert("connection", "keep-alive".parse().unwrap());
        response
    }
    let app = Router::new()
        .route("/p/health", get(|| async { "ok" }))
        .route("/d/health", get(|| async { "ok" }))
        .route("/p/v1/chat/completions", post(prefill))
        .route("/d/v1/chat/completions", post(decode));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let config = RouterConfig {
        mode: RoutingMode::VllmPrefillDecode {
            prefill_urls: vec![(format!("{base}/p"), None)],
            decode_urls: vec![format!("{base}/d")],
            prefill_policy: None,
            decode_policy: None,
            discovery_address: None,
        },
        policy: PolicyConfig::Random,
        kv_connector: KvConnector::Nixl,
        worker_startup_timeout_secs: 5,
        ..RouterConfig::default()
    };
    let context = Arc::new(
        vllm_router_rs::server::AppContext::new(config, reqwest::Client::new(), 64, None, vec![])
            .unwrap(),
    );
    let router = RouterFactory::create_router(&context).await.unwrap();
    for interrupted in [false, true] {
        for stream in [false, true] {
            // Include absent, false, and true: some paths key on field presence.
            for logprobs in [None, Some(false), Some(true)] {
                let mut request = json!({"model":if interrupted {"interrupted"} else {"normal"},
                    "messages":[{"role":"user","content":"test"}],"max_tokens":8,"stream":stream});
                if let Some(value) = logprobs {
                    request["logprobs"] = json!(value);
                }
                let response = router
                    .route_transparent(None, "/v1/chat/completions", &Method::POST, request)
                    .await;
                let hop_header_present = response.headers().contains_key("connection");
                let status = response.status();
                let body = String::from_utf8(
                    axum::body::to_bytes(response.into_body(), 4096)
                        .await
                        .unwrap()
                        .to_vec(),
                )
                .unwrap();
                if interrupted {
                    assert_eq!(
                        status,
                        if stream {
                            StatusCode::OK
                        } else {
                            StatusCode::BAD_GATEWAY
                        },
                        "{body}"
                    );
                    assert!(!hop_header_present);
                    assert!(body.contains("upstream_recomputed"), "{body}");
                    assert!(!body.contains("[DONE]"), "{body}");
                    assert!(!body.contains("finish_reason"), "{body}");
                } else {
                    assert_eq!(status, StatusCode::OK, "{body}");
                    assert!(body.contains("finish_reason"), "{body}");
                    assert_eq!(body.contains("[DONE]"), stream);
                }
            }
        }
    }
    for stream in [false, true] {
        let request = json!({"model":"interrupted", "stop":["recomputed"],
            "messages":[{"role":"user","content":"test"}], "stream":stream});
        let response = router
            .route_transparent(None, "/v1/chat/completions", &Method::POST, request)
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = String::from_utf8(
            axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(body.contains("stop_reason"));
        assert!(!body.contains("upstream_recomputed"));
    }
    server.abort();
}
