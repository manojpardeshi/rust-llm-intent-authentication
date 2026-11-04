use std::path::PathBuf;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt as _;
use intent_server::{ApiKeys, AppState, Metrics, ServerConfig, router};
use intent_token::KeyPair;
use serde_json::{Value, json};
use tower::ServiceExt as _;

const KEY: &str = "test-key-123";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(root().join(rel)).unwrap()
}

fn app(max_body: usize) -> Router {
    let dir = std::env::temp_dir().join(format!("intent-server-test-{}", uuid_like()));
    std::fs::create_dir_all(&dir).unwrap();
    let key_path = dir.join("signing.key.json");
    KeyPair::generate().unwrap().save(&key_path).unwrap();
    let config = ServerConfig::from_toml_str(&format!(
        "key_path = {:?}\npolicy_path = {:?}\nrules_path = {:?}\nmax_body_bytes = {max_body}",
        key_path,
        root().join("config/policy.example.toml"),
        root().join("config/rules.example.toml"),
    ))
    .unwrap();
    let state =
        AppState::from_config(&config, ApiKeys::from_csv(KEY), Some(Metrics::new())).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    router(state)
}

fn uuid_like() -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

async fn call(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let res = app.clone().oneshot(req).await.unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, body)
}

fn post(uri: &str, key: Option<&str>, body: String) -> Request<Body> {
    let mut b = Request::post(uri).header(header::CONTENT_TYPE, "application/json");
    if let Some(k) = key {
        b = b.header(header::AUTHORIZATION, format!("Bearer {k}"));
    }
    b.body(Body::from(body)).unwrap()
}

fn get(uri: &str) -> Request<Body> {
    Request::get(uri).body(Body::empty()).unwrap()
}

async fn evaluate(app: &Router, scenario: &str) -> (StatusCode, Value) {
    call(
        app,
        post(
            "/v1/evaluate",
            Some(KEY),
            read(&format!("examples/scenarios/{scenario}.json")),
        ),
    )
    .await
}

fn tool_calls_of(scenario: &str) -> Value {
    serde_json::from_str::<Value>(&read(&format!("examples/scenarios/{scenario}.json"))).unwrap()
        ["tool_calls"]
        .clone()
}

#[tokio::test]
async fn auth_required_but_public_endpoints_open() {
    let app = app(256 * 1024);
    let body = read("examples/scenarios/allow_read.json");
    for uri in ["/v1/evaluate", "/v1/verify"] {
        for key in [None, Some("wrong")] {
            let (status, _) = call(&app, post(uri, key, body.clone())).await;
            assert_eq!(status, StatusCode::UNAUTHORIZED, "{uri} {key:?}");
        }
    }
    // x-api-key header also works.
    let req = Request::post("/v1/evaluate")
        .header(header::CONTENT_TYPE, "application/json")
        .header("x-api-key", KEY)
        .body(Body::from(body))
        .unwrap();
    assert_eq!(call(&app, req).await.0, StatusCode::OK);

    assert_eq!(
        call(&app, get("/healthz")).await,
        (StatusCode::OK, json!({"status": "ok"}))
    );
    assert_eq!(
        call(&app, get("/.well-known/jwks.json")).await.0,
        StatusCode::OK
    );
}

#[tokio::test]
async fn scenario_decisions() {
    let app = app(256 * 1024);

    let (s, b) = evaluate(&app, "allow_read").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(b["decision"], "allow");
    assert!(b["token"]["token"].is_string());
    assert!(b["req_hash"].is_string());

    let (s, b) = evaluate(&app, "deny_delete").await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    assert_eq!(b["decision"], "deny");
    assert!(b.get("token").is_none());

    let (s, b) = evaluate(&app, "stepup_transfer").await;
    assert_eq!(s, StatusCode::ACCEPTED);
    assert_eq!(b["decision"], "step_up");
    assert!(b["reason"].is_string());

    let (s, b) = evaluate(&app, "injection").await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    assert_eq!(b["intent"]["category"], "Malicious");
}

#[tokio::test]
async fn verify_roundtrip_tamper_and_scope() {
    let app = app(256 * 1024);
    let (_, b) = evaluate(&app, "allow_read").await;
    let token = b["token"]["token"].as_str().unwrap().to_owned();
    let calls = tool_calls_of("allow_read");
    let verify = |calls: Value, scope: Option<&str>| {
        let mut body = json!({"token": token, "tool_calls": calls});
        if let Some(s) = scope {
            body["scope"] = json!(s);
        }
        post("/v1/verify", Some(KEY), body.to_string())
    };

    let (s, b) = call(&app, verify(calls.clone(), Some("data:read"))).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(b["valid"], true);
    assert_eq!(b["claims"]["sub"], "user-42");

    let tampered: Value = serde_json::from_str(&read("examples/tampered_calls.json")).unwrap();
    let (s, b) = call(&app, verify(tampered, None)).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    assert_eq!(
        (b["valid"].clone(), b["error"].clone()),
        (json!(false), json!("request_mismatch"))
    );

    let (s, b) = call(&app, verify(calls, Some("admin:all"))).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    assert_eq!(b["error"], "missing_scope");
    assert!(!b.to_string().contains(&token));
}

#[tokio::test]
async fn jwks_kid_matches_token() {
    let app = app(256 * 1024);
    let (_, b) = evaluate(&app, "allow_read").await;
    let token = b["token"]["token"].as_str().unwrap();
    let header_b64 = token.split('.').next().unwrap();
    use base64::Engine as _;
    let header: Value = serde_json::from_slice(
        &base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(header_b64)
            .unwrap(),
    )
    .unwrap();

    let (s, jwks) = call(&app, get("/.well-known/jwks.json")).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(jwks["keys"][0]["kid"], header["kid"]);
    assert!(jwks["keys"][0].get("d").is_none());
}

#[tokio::test]
async fn oversized_body_is_rejected() {
    let app = app(1024);
    let big = json!({"subject": "u", "messages": [{"role": "user", "content": "x".repeat(4096)}]});
    let (s, _) = call(&app, post("/v1/evaluate", Some(KEY), big.to_string())).await;
    assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn metrics_exposed_after_evaluate() {
    let app = app(256 * 1024);
    evaluate(&app, "allow_read").await;
    let (s, b) = call(&app, get("/metrics")).await;
    assert_eq!(s, StatusCode::OK);
    let text = b.as_str().unwrap();
    assert!(text.contains("intent_evaluations_total"), "{text}");
    assert!(text.contains("intent_evaluate_duration_seconds"));
}

#[tokio::test]
async fn non_json_is_rejected() {
    let app = app(256 * 1024);
    let req = Request::post("/v1/evaluate")
        .header(header::AUTHORIZATION, format!("Bearer {KEY}"))
        .header(header::CONTENT_TYPE, "text/plain")
        .body(Body::from("hi"))
        .unwrap();
    assert_eq!(call(&app, req).await.0, StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

#[test]
fn refuses_to_start_without_keys() {
    let config =
        ServerConfig::from_toml_str("key_path = 'k'\npolicy_path = 'p'\nrules_path = 'r'").unwrap();
    assert!(matches!(
        AppState::from_config(&config, ApiKeys::default(), None),
        Err(intent_server::Error::NoApiKeys)
    ));
}
