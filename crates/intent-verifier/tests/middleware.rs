#![cfg(feature = "axum")]
mod common;
use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use axum::routing::post;
use common::*;
use http_body_util::BodyExt as _;
use intent_token::KeyPair;
use std::sync::Arc;

use intent_verifier::{
    InMemoryReplayStore, IntentLayer, VerifiedIntent, VerifiedToolCalls, Verifier,
};
use serde_json::{Value, json};
use tower::ServiceExt as _;

fn app(key: &KeyPair, layer: impl FnOnce(IntentLayer) -> IntentLayer) -> Router {
    let v = Verifier::from_jwks(&jwks(key), ISS, AUD).unwrap();
    // The handler echoes the subject and the body it received.
    Router::new().route(
        "/tools/run",
        post(
            |VerifiedIntent(c): VerifiedIntent, body: String| async move {
                axum::Json(json!({"sub": c.sub, "body": body}))
            },
        )
        .layer(layer(IntentLayer::new(v))),
    )
}

fn single() -> String {
    serde_json::to_string(&calls()[0]).unwrap()
}

async fn send(
    app: &Router,
    token: Option<&str>,
    body: String,
) -> (StatusCode, Value, axum::http::HeaderMap) {
    let mut b = Request::post("/tools/run").header(header::CONTENT_TYPE, "application/json");
    if let Some(t) = token {
        b = b.header(header::AUTHORIZATION, format!("Bearer {t}"));
    }
    let res = app
        .clone()
        .oneshot(b.body(Body::from(body)).unwrap())
        .await
        .unwrap();
    let (status, headers) = (res.status(), res.headers().clone());
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json, headers)
}

#[tokio::test]
async fn valid_request_passes_with_body_intact() {
    let key = KeyPair::generate().unwrap();
    let app = app(&key, |l| l.require_scope("data:read"));
    let token = mint(&key, &["data:read"]);
    let body = single();
    let (status, json, _) = send(&app, Some(&token), body.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["sub"], "user-42");
    assert_eq!(json["body"], body);
}

#[tokio::test]
async fn all_body_shapes_accepted() {
    let key = KeyPair::generate().unwrap();
    let app = app(&key, |l| l);
    let token = mint(&key, &["data:read"]);
    let call = serde_json::to_value(&calls()[0]).unwrap();
    for body in [
        call.to_string(),
        json!([call]).to_string(),
        json!({"tool_calls": [call]}).to_string(),
    ] {
        let (status, _, _) = send(&app, Some(&token), body.clone()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
}

#[tokio::test]
async fn x_intent_token_header_accepted() {
    let key = KeyPair::generate().unwrap();
    let app = app(&key, |l| l);
    let req = Request::post("/tools/run")
        .header("x-intent-token", mint(&key, &["data:read"]))
        .body(Body::from(single()))
        .unwrap();
    assert_eq!(app.oneshot(req).await.unwrap().status(), StatusCode::OK);
}

#[tokio::test]
async fn missing_token_is_401() {
    let key = KeyPair::generate().unwrap();
    let app = app(&key, |l| l);
    let (status, json, headers) = send(&app, None, single()).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(json["error"], "missing_token");
    assert_eq!(
        headers[header::WWW_AUTHENTICATE],
        r#"Bearer error="invalid_token""#
    );
}

#[tokio::test]
async fn tampered_call_is_401_request_mismatch() {
    let key = KeyPair::generate().unwrap();
    let app = app(&key, |l| l);
    let token = mint(&key, &["data:read"]);
    let body = serde_json::to_string(&tampered()[0]).unwrap();
    let (status, json, headers) = send(&app, Some(&token), body).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(json["error"], "request_mismatch");
    assert!(headers.contains_key(header::WWW_AUTHENTICATE));
    assert!(!json.to_string().contains(&token));
}

#[tokio::test]
async fn missing_scope_is_401() {
    let key = KeyPair::generate().unwrap();
    let app = app(&key, |l| l.require_scope("data:write"));
    let token = mint(&key, &["data:read"]);
    let (status, json, _) = send(&app, Some(&token), single()).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(json["error"], "missing_scope");
}

#[tokio::test]
async fn garbage_token_is_401() {
    let key = KeyPair::generate().unwrap();
    let app = app(&key, |l| l);
    let (status, json, _) = send(&app, Some("abc.def.ghi"), single()).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(json["error"], "malformed_token");
}

#[tokio::test]
async fn oversized_body_is_413() {
    let key = KeyPair::generate().unwrap();
    let app = app(&key, |l| l.max_body_bytes(32));
    let token = mint(&key, &["data:read"]);
    let (status, _, _) = send(&app, Some(&token), single()).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn bad_json_is_400() {
    let key = KeyPair::generate().unwrap();
    let app = app(&key, |l| l);
    let token = mint(&key, &["data:read"]);
    for body in [
        "{not json",
        "42",
        r#"{"tool_calls": 3}"#,
        "[]",
        r#"{"x":1}"#,
        r#"[{"name":"read_file","extra":1}]"#,
    ] {
        let (status, _, _) = send(&app, Some(&token), body.into()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }
}

#[tokio::test]
async fn extractor_without_layer_is_500() {
    let app = Router::new().route("/x", axum::routing::get(|_: VerifiedIntent| async { "ok" }));
    let res = app
        .oneshot(Request::get("/x").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[tokio::test]
async fn smuggled_call_beside_tool_calls_is_rejected() {
    // A handler parsing this as a single ToolCall would run delete_db while
    // the middleware verified only the read_file call in `tool_calls`.
    let key = KeyPair::generate().unwrap();
    let app = app(&key, |l| l);
    let token = mint(&key, &["data:read"]);
    let call = serde_json::to_value(&calls()[0]).unwrap();
    let body = json!({"name": "delete_db", "arguments": {}, "tool_calls": [call]});
    let (status, _, _) = send(&app, Some(&token), body.to_string()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn handler_sees_verified_tool_calls() {
    let key = KeyPair::generate().unwrap();
    let v = Verifier::from_jwks(&jwks(&key), ISS, AUD).unwrap();
    let app = Router::new().route(
        "/tools/run",
        post(|VerifiedToolCalls(c): VerifiedToolCalls| async move { axum::Json(c) })
            .layer(IntentLayer::new(v)),
    );
    let token = mint(&key, &["data:read"]);
    let (status, json, _) = send(&app, Some(&token), single()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json, serde_json::to_value(calls()).unwrap());
}

fn replay_app(key: &KeyPair, capacity: usize) -> Router {
    let v = Verifier::from_jwks(&jwks(key), ISS, AUD)
        .unwrap()
        .with_replay_store(Arc::new(InMemoryReplayStore::new(capacity)));
    Router::new().route(
        "/tools/run",
        post(|| async { "ok" }).layer(IntentLayer::new(v)),
    )
}

#[tokio::test]
async fn replayed_token_is_401() {
    let key = KeyPair::generate().unwrap();
    let app = replay_app(&key, 10);
    let token = mint(&key, &["data:read"]);
    let (status, _, _) = send(&app, Some(&token), single()).await;
    assert_eq!(status, StatusCode::OK);
    let (status, json, headers) = send(&app, Some(&token), single()).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(json["error"], "replayed");
    assert!(headers.contains_key(header::WWW_AUTHENTICATE));
}

#[tokio::test]
async fn full_replay_store_is_503() {
    let key = KeyPair::generate().unwrap();
    let app = replay_app(&key, 1);
    let (status, _, _) = send(&app, Some(&mint(&key, &["data:read"])), single()).await;
    assert_eq!(status, StatusCode::OK);
    let (status, json, headers) = send(&app, Some(&mint(&key, &["data:read"])), single()).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(json["error"], "replay_store_unavailable");
    assert_eq!(headers[header::RETRY_AFTER], "1");
}
