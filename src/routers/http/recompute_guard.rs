//! Prevent a backend recomputation interruption from becoming a successful response.
//! This is terminal validation, not an implementation of generation recovery.

use std::io;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use bytes::{Bytes, BytesMut};
use futures_util::{Stream, StreamExt};
use serde_json::{json, Value};

fn error_value(code: &str, message: &str) -> Value {
    json!({"error": {"message": message, "type": "upstream_error", "code": code}})
}

fn recomputed_error() -> Value {
    error_value(
        "upstream_recomputed",
        "Decode was interrupted for recomputation; the router cannot resume this response.",
    )
}

pub(super) fn is_recomputed(value: &Value) -> bool {
    value
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(|choices| {
            choices.iter().any(|choice| {
                choice.get("stop_reason").and_then(Value::as_str) == Some("recomputed")
            })
        })
}

pub(super) fn marker_is_user_stop(request: &Value) -> bool {
    let stop = &request["stop"];
    stop.as_str() == Some("recomputed")
        || stop
            .as_array()
            .is_some_and(|items| items.iter().any(|item| item.as_str() == Some("recomputed")))
}

pub(super) fn failure_response() -> Response {
    (StatusCode::BAD_GATEWAY, Json(recomputed_error())).into_response()
}

pub(super) fn encoding_error(response: &reqwest::Response) -> Option<Response> {
    response
        .headers()
        .get_all("content-encoding")
        .iter()
        .any(|value| {
            value.to_str().map_or(true, |value| {
                value
                    .split(',')
                    .map(str::trim)
                    .any(|coding| !coding.is_empty() && !coding.eq_ignore_ascii_case("identity"))
            })
        })
        .then(|| {
            (
                StatusCode::BAD_GATEWAY,
                Json(error_value(
                    "unsupported_upstream_encoding",
                    "Cannot validate an encoded decode response.",
                )),
            )
                .into_response()
        })
}

pub(super) fn enabled(path: &str, request: &Value) -> bool {
    matches!(path, "/v1/chat/completions" | "/v1/completions") && !marker_is_user_stop(request)
}

pub(super) fn interrupted(bytes: &[u8]) -> bool {
    serde_json::from_slice::<Value>(bytes).is_ok_and(|value| is_recomputed(&value))
}

// Native vLLM Chat/Completions emit single-line JSON or keep-alive comments,
// delimited by LF blank lines. Preserve complete frames without re-encoding them.
pub(super) fn guard_stream<S, E>(stream: S) -> impl Stream<Item = Result<Bytes, io::Error>> + Send
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::error::Error + Send + Sync + 'static,
{
    async_stream::try_stream! {
        let mut source = Box::pin(stream);
        let mut raw = BytesMut::new();
        let mut scan = 0;
        while let Some(chunk) = source.next().await {
            raw.extend_from_slice(&chunk.map_err(io::Error::other)?);
            while let Some(offset) = raw[scan..].windows(2).position(|pair| pair == b"\n\n") {
                let end = scan + offset + 2;
                let frame = raw.split_to(end).freeze();
                scan = 0;
                if frame[..frame.len() - 2].strip_prefix(b"data: ")
                    .is_some_and(interrupted) {
                    drop(source);
                    yield Bytes::from(format!("data: {}\n\n", recomputed_error()));
                    return;
                }
                yield frame;
            }
            // Retain the last byte so a delimiter can straddle chunks.
            scan = raw.len().saturating_sub(1);
        }
        if !raw.is_empty() { yield raw.freeze(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_encoding_checks_every_coding_case_insensitively() {
        for (values, rejected) in [
            (vec![], false),
            (vec!["identity"], false),
            (vec!["Identity"], false),
            (vec![" identity, IDENTITY "], false),
            (vec!["identity", "Identity"], false),
            (vec!["gzip"], true),
            (vec!["identity, gzip"], true),
            (vec!["identity", "gzip"], true),
            (vec!["gzip", "identity"], true),
        ] {
            let mut response = axum::http::Response::builder();
            for value in &values {
                response = response.header("content-encoding", *value);
            }
            let response = reqwest::Response::from(response.body("").unwrap());
            let error = encoding_error(&response);
            assert_eq!(error.is_some(), rejected, "{values:?}");
            if let Some(error) = error {
                assert_eq!(error.status(), StatusCode::BAD_GATEWAY);
            }
        }
    }

    fn feed(chunks: &[&[u8]]) -> Vec<u8> {
        let chunks: Vec<_> = chunks
            .iter()
            .map(|b| Ok::<_, io::Error>(Bytes::copy_from_slice(b)))
            .collect();
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            guard_stream(futures_util::stream::iter(chunks))
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .flat_map(Result::unwrap)
                .collect()
        })
    }

    #[test]
    fn all_chunk_boundaries_suppress_recompute_and_done() {
        let input = b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\",\"stop_reason\":\"recomputed\"}]}\n\ndata: [DONE]\n\n";
        for split in 0..=input.len() {
            let output = String::from_utf8(feed(&[&input[..split], &input[split..]])).unwrap();
            assert!(output.contains("upstream_recomputed"));
            assert!(!output.contains("[DONE]"));
            assert!(!output.contains("finish_reason"));
        }
    }

    #[test]
    fn content_before_interruption_is_preserved() {
        let first = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"思考\",\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{}\"}}]}}]}\n\n";
        let input = format!(
            "{first}data: {{\"choices\":[{{\"stop_reason\":\"recomputed\"}}]}}\n\ndata: [DONE]\n\n"
        );
        let chunks: Vec<_> = input.as_bytes().chunks(1).collect();
        let output = String::from_utf8(feed(&chunks)).unwrap();
        assert!(output.starts_with(first));
        assert_eq!(output.matches("upstream_recomputed").count(), 1);
        assert!(!output.contains("[DONE]"));
    }

    #[test]
    fn normal_stops_and_text_are_unchanged() {
        for reason in ["stop", "length", "tool_calls"] {
            let input = format!("data: {{\"choices\":[{{\"finish_reason\":\"{reason}\",\"delta\":{{\"content\":\"recomputed\"}}}}]}}\n\ndata: [DONE]\n\n");
            assert_eq!(feed(&[input.as_bytes()]), input.as_bytes());
        }
    }

    #[test]
    fn native_frames_preserve_bytes_across_all_chunk_boundaries() {
        let input = concat!(
            ": keep-alive\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"你好\\nworld\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n",
        )
        .as_bytes();
        for split in 0..=input.len() {
            assert_eq!(feed(&[&input[..split], &input[split..]]), input);
        }
        assert_eq!(feed(&input.chunks(1).collect::<Vec<_>>()), input);
    }

    #[test]
    fn large_normal_events_are_not_rejected() {
        let input = format!(
            "data: {}\n\n",
            json!({"choices":[{"delta":{"content":"x".repeat(1024 * 1024 + 1)}}]})
        );
        assert_eq!(feed(&[input.as_bytes()]), input.as_bytes());
    }

    #[test]
    fn any_choice_can_report_recompute() {
        assert!(is_recomputed(
            &json!({"choices":[{}, {"stop_reason":"recomputed"}]})
        ));
        assert!(!is_recomputed(&json!({"error":{"message":"recomputed"}})));
    }

    #[test]
    fn explicit_user_stop_is_not_an_unambiguous_recovery_marker() {
        assert!(marker_is_user_stop(&json!({"stop":"recomputed"})));
        assert!(marker_is_user_stop(&json!({"stop":["x","recomputed"]})));
        assert!(!marker_is_user_stop(&json!({"stop":["other"]})));
        assert!(!marker_is_user_stop(
            &json!({"messages":[{"content":"recomputed"}]})
        ));
    }

    #[tokio::test]
    async fn incomplete_event_retains_original_transport_behavior() {
        let input =
            futures_util::stream::iter(vec![Ok::<_, io::Error>(Bytes::from_static(b"data: {"))]);
        let output: Vec<_> = guard_stream(input).collect().await;
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].as_ref().unwrap().as_ref(), b"data: {");
    }

    #[tokio::test]
    async fn transport_error_remains_a_body_error() {
        let input = futures_util::stream::iter(vec![Err::<Bytes, _>(io::Error::other("injected"))]);
        let output: Vec<_> = guard_stream(input).collect().await;
        assert_eq!(output.len(), 1);
        assert!(output[0].is_err());
    }
    #[tokio::test]
    async fn upstream_is_dropped_before_error_is_delivered() {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };
        struct Source(Arc<AtomicBool>);
        impl Drop for Source {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        impl Stream for Source {
            type Item = Result<Bytes, io::Error>;
            fn poll_next(
                self: std::pin::Pin<&mut Self>,
                _: &mut std::task::Context<'_>,
            ) -> std::task::Poll<Option<Self::Item>> {
                std::task::Poll::Ready(Some(Ok(Bytes::from_static(
                    b"data: {\"choices\":[{\"stop_reason\":\"recomputed\"}]}\n\ndata: [DONE]\n\n",
                ))))
            }
        }
        let dropped = Arc::new(AtomicBool::new(false));
        let output = guard_stream(Source(dropped.clone()));
        futures_util::pin_mut!(output);
        assert!(output
            .next()
            .await
            .unwrap()
            .unwrap()
            .windows(19)
            .any(|s| s == b"upstream_recomputed"));
        assert!(dropped.load(Ordering::SeqCst));
        assert!(output.next().await.is_none());
    }
    #[tokio::test]
    async fn complete_event_is_delivered_without_waiting_for_more_bytes() {
        for data in [
            b": keep-alive\n\n".as_slice(),
            b"data: {\"choices\":[]}\n\n".as_slice(),
            b"data: {\"choices\":[{\"stop_reason\":\"recomputed\"}]}\n\n".as_slice(),
        ] {
            let interrupted = data.windows(10).any(|w| w == b"recomputed");
            let source =
                futures_util::stream::iter(vec![Ok::<_, io::Error>(Bytes::copy_from_slice(data))])
                    .chain(futures_util::stream::pending());
            let output = guard_stream(source);
            futures_util::pin_mut!(output);
            let event = tokio::time::timeout(std::time::Duration::from_millis(200), output.next())
                .await
                .expect("a complete LF event must not wait for another byte or EOF")
                .unwrap()
                .unwrap();
            if interrupted {
                assert!(String::from_utf8(event.to_vec())
                    .unwrap()
                    .contains("upstream_recomputed"));
            } else {
                assert_eq!(event.as_ref(), data);
            }
        }
    }
}
