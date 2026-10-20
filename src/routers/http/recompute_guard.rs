//! Prevent a backend recomputation interruption from becoming a successful response.
//! This is terminal validation, not an implementation of generation recovery.

use std::io;

use axum::{body::Body, http::StatusCode, response::Response};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde_json::{json, Value};

const MAX_EVENT_BYTES: usize = 1024 * 1024;

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
    Response::builder()
        .status(StatusCode::BAD_GATEWAY)
        .header("content-type", "application/json")
        .body(Body::from(recomputed_error().to_string()))
        .expect("constant response headers")
}

pub(super) async fn decode_response(
    response: reqwest::Response,
    prefill: Option<&Value>,
    streaming: bool,
    merge_logprobs: bool,
    user_stop: bool,
    path: &str,
) -> Result<Response, String> {
    let validate = matches!(path, "/v1/chat/completions" | "/v1/completions") && !user_stop;
    let status = response.status();
    let mut headers = response.headers().clone();
    let encoded = headers
        .get("content-encoding")
        .is_some_and(|value| value.as_bytes() != b"identity");
    if status.is_success() && encoded && validate {
        return Response::builder()
            .status(StatusCode::BAD_GATEWAY)
            .header("content-type", "application/json")
            .body(Body::from(
                error_value(
                    "unsupported_upstream_encoding",
                    "Cannot validate an encoded decode response.",
                )
                .to_string(),
            ))
            .map_err(|e| e.to_string());
    }
    for name in [
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "upgrade",
        "host",
    ] {
        headers.remove(name);
    }
    let body = if !status.is_success() || (streaming && !validate) {
        Body::from_stream(response.bytes_stream())
    } else if streaming {
        Body::from_stream(guard_stream(response.bytes_stream()))
    } else {
        let bytes = response.bytes().await.map_err(|e| e.to_string())?;
        match serde_json::from_slice::<Value>(&bytes) {
            Ok(mut value) => {
                if validate && is_recomputed(&value) {
                    return Ok(failure_response());
                }
                if merge_logprobs
                    && super::logprobs_merge::merge_logprobs_in_json(
                        prefill.unwrap_or(&Value::Null),
                        &mut value,
                    )
                {
                    Body::from(serde_json::to_vec(&value).map_err(|e| e.to_string())?)
                } else {
                    Body::from(bytes)
                }
            }
            Err(error) if merge_logprobs => return Err(error.to_string()),
            Err(_) => Body::from(bytes),
        }
    };
    let mut result = Response::builder()
        .status(status)
        .body(body)
        .map_err(|e| e.to_string())?;
    *result.headers_mut() = headers;
    Ok(result)
}

fn error_event(value: Value) -> Bytes {
    Bytes::from(format!("data: {value}\n\n"))
}

/// Lines are normalized to LF. CR is completed immediately, including when a
/// CRLF is split across network chunks; the following LF is then consumed.
#[derive(Default)]
struct Events {
    line: Vec<u8>,
    event: Vec<u8>,
    data: Vec<u8>,
    previous_cr: bool,
    first_line: bool,
    done: bool,
}

impl Events {
    fn new() -> Self {
        Self {
            first_line: true,
            ..Self::default()
        }
    }

    fn fail(&mut self, code: &str, message: &str) -> Bytes {
        self.done = true;
        self.line.clear();
        self.event.clear();
        self.data.clear();
        error_event(error_value(code, message))
    }

    fn push(&mut self, byte: u8) -> Option<Bytes> {
        if self.done {
            return None;
        }
        if self.previous_cr {
            self.previous_cr = false;
            if byte == b'\n' {
                return None;
            }
        }
        if self.event.len() + self.line.len() >= MAX_EVENT_BYTES {
            return Some(self.fail(
                "upstream_event_too_large",
                "Decode SSE event exceeds the router validation limit.",
            ));
        }
        if byte != b'\r' && byte != b'\n' {
            self.line.push(byte);
            return None;
        }
        self.previous_cr = byte == b'\r';
        let mut line = std::mem::take(&mut self.line);
        if self.first_line {
            self.first_line = false;
            if line.starts_with(&[0xef, 0xbb, 0xbf]) {
                line.drain(..3);
            }
        }
        self.event.extend_from_slice(&line);
        self.event.push(b'\n');
        if !line.is_empty() {
            if let Some(mut data) = line.strip_prefix(b"data:") {
                if data.starts_with(b" ") {
                    data = &data[1..];
                }
                self.data.extend_from_slice(data);
                self.data.push(b'\n');
            } else if line == b"data" {
                self.data.push(b'\n');
            }
            return None;
        }
        let data = std::mem::take(&mut self.data);
        if let Ok(value) = serde_json::from_slice::<Value>(&data) {
            if is_recomputed(&value) {
                self.done = true;
                self.event.clear();
                return Some(error_event(recomputed_error()));
            }
        }
        Some(Bytes::from(std::mem::take(&mut self.event)))
    }
}

pub(super) fn guard_stream<S, E>(stream: S) -> impl Stream<Item = Result<Bytes, io::Error>> + Send
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::error::Error + Send + Sync + 'static,
{
    futures_util::stream::unfold(
        (Box::pin(stream), Events::new(), Bytes::new(), 0usize),
        |(mut source, mut parser, mut chunk, mut offset)| async move {
            if parser.done {
                return None;
            }
            loop {
                while offset < chunk.len() {
                    let byte = chunk[offset];
                    offset += 1;
                    if let Some(event) = parser.push(byte) {
                        return Some((Ok(event), (source, parser, chunk, offset)));
                    }
                }
                match source.next().await {
                    Some(Ok(bytes)) => {
                        chunk = bytes;
                        offset = 0;
                    }
                    Some(Err(error)) => {
                        parser.done = true;
                        return Some((
                            Err(io::Error::other(error)),
                            (source, parser, chunk, offset),
                        ));
                    }
                    None => {
                        if !parser.line.is_empty() || !parser.event.is_empty() {
                            let error = parser.fail(
                                "upstream_incomplete_event",
                                "Decode stream ended inside an SSE event.",
                            );
                            return Some((Ok(error), (source, parser, chunk, offset)));
                        }
                        return None;
                    }
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(chunks: &[&[u8]]) -> Vec<u8> {
        let mut p = Events::new();
        chunks
            .iter()
            .flat_map(|chunk| chunk.iter().filter_map(|&b| p.push(b)).collect::<Vec<_>>())
            .flatten()
            .collect()
    }

    #[test]
    fn all_chunk_boundaries_suppress_recompute_and_done() {
        let input = b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\",\"stop_reason\":\"recomputed\"}]}\r\n\r\ndata: [DONE]\r\n\r\n";
        for split in 0..=input.len() {
            let output = String::from_utf8(feed(&[&input[..split], &input[split..]])).unwrap();
            assert!(output.contains("upstream_recomputed"));
            assert!(!output.contains("[DONE]"));
            assert!(!output.contains("finish_reason"));
        }
    }

    #[test]
    fn normal_stops_and_text_are_unchanged() {
        for reason in ["stop", "length", "tool_calls"] {
            let input = format!("data: {{\"choices\":[{{\"finish_reason\":\"{reason}\",\"delta\":{{\"content\":\"recomputed\"}}}}]}}\n\ndata: [DONE]\n\n");
            assert_eq!(feed(&[input.as_bytes()]), input.as_bytes());
        }
    }

    #[test]
    fn multiline_bom_and_bare_cr() {
        let input =
            b"\xef\xbb\xbfdata: {\"choices\":\rdata: [{\"stop_reason\":\"recomputed\"}]}\r\r";
        assert!(String::from_utf8(feed(&[input]))
            .unwrap()
            .contains("upstream_recomputed"));
    }

    #[test]
    fn oversize_events_fail_instead_of_bypassing_validation() {
        let input = vec![b'a'; MAX_EVENT_BYTES + 2];
        assert!(String::from_utf8(feed(&[&input]))
            .unwrap()
            .contains("upstream_event_too_large"));
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
    async fn incomplete_event_is_explicit_failure() {
        let input =
            futures_util::stream::iter(vec![Ok::<_, io::Error>(Bytes::from_static(b"data: {"))]);
        let output: Vec<_> = guard_stream(input).collect().await;
        assert_eq!(output.len(), 1);
        assert!(String::from_utf8(output[0].as_ref().unwrap().to_vec())
            .unwrap()
            .contains("upstream_incomplete_event"));
    }

    #[tokio::test]
    async fn transport_error_remains_a_body_error() {
        let input = futures_util::stream::iter(vec![Err::<Bytes, _>(io::Error::other("injected"))]);
        let output: Vec<_> = guard_stream(input).collect().await;
        assert_eq!(output.len(), 1);
        assert!(output[0].is_err());
    }
}
