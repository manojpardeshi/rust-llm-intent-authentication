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
    app_with(&format!("max_body_bytes = {max_body}"), KEY)
}

fn app_with(extra: &str, keys: &str) -> Router {
    let dir = std::env::temp_dir().join(format!("intent-server-test-{}", uuid_like()));
    std::fs::create_dir_all(&dir).unwrap();
    let key_path = dir.join("signing.key.json");
    KeyPair::generate().unwrap().save(&key_path).unwrap();
    let config = ServerConfig::from_toml_str(&format!(
        "key_path = {:?}\npolicy_path = {:?}\nrules_path = {:?}\n{extra}",
        key_path,
        root().join("config/policy.example.toml"),
        root().join("config/rules.example.toml"),
    ))
    .unwrap();
    let state =
        AppState::from_config(&config, ApiKeys::from_csv(keys), Some(Metrics::new())).unwrap();
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

fn request_with(calls: Value) -> String {
    json!({"subject": "user-42", "messages": [], "tool_calls": calls}).to_string()
}

/// IMPROVEMENTS 1.1: batching a forbidden call with an allowed one must not
/// launder it. Each call is decided on its own and the strictest wins.
#[tokio::test]
async fn batching_cannot_bypass_per_call_policy() {
    let app = app(256 * 1024);
    let email = json!({"name": "send_email", "arguments": {"to": "a@b.c", "body": "hi"}});
    let delete = json!({"name": "delete_records", "arguments": {"table": "users"}});
    let drop = json!({"name": "run_sql", "arguments": {"query": "drop table users"}});
    let read = json!({"name": "read_file", "arguments": {"path": "/srv/a"}});

    let cases = [
        (json!([delete.clone()]), StatusCode::FORBIDDEN, "deny"),
        (
            json!([delete, email.clone()]),
            StatusCode::FORBIDDEN,
            "deny",
        ),
        (json!([drop.clone()]), StatusCode::ACCEPTED, "step_up"),
        (
            json!([drop, email.clone()]),
            StatusCode::ACCEPTED,
            "step_up",
        ),
    ];
    for (calls, status, decision) in cases {
        let (s, body) = call(
            &app,
            post("/v1/evaluate", Some(KEY), request_with(calls.clone())),
        )
        .await;
        assert_eq!(
            (s, body["decision"].as_str()),
            (status, Some(decision)),
            "{calls} -> {body}"
        );
        assert!(body.get("token").is_none(), "{calls}");
    }

    // Two allowed calls: union of scopes, shortest TTL.
    let (s, body) = call(
        &app,
        post(
            "/v1/evaluate",
            Some(KEY),
            request_with(json!([read, email])),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    assert_eq!(body["scopes"], json!(["data:read", "messages:send"]));
    assert_eq!(body["ttl"], json!(60));
    assert_eq!(body["intent"]["category"], json!("SendMessage"));
}

/// IMPROVEMENTS 1.3: integers beyond double precision cannot be bound to a
/// token without collisions, so the request is rejected up front.
#[tokio::test]
async fn unsafe_integer_arguments_are_rejected() {
    let app = app(256 * 1024);
    let body = r#"{"subject":"u","tool_calls":[{"name":"read_file","arguments":{"path":"/a","id":9007199254740993}}]}"#;
    let (s, b) = call(&app, post("/v1/evaluate", Some(KEY), body.into())).await;
    assert_eq!(
        (s, b["error"].as_str()),
        (StatusCode::BAD_REQUEST, Some("invalid_request")),
        "{b}"
    );
}

fn sha256_hex(s: &str) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(s.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// IMPROVEMENTS 2.1: a key bound to subject patterns cannot ask for others.
#[tokio::test]
async fn subject_bindings_limit_which_subjects_a_key_may_request() {
    let extra = format!("[subject_bindings]\n{:?} = [\"user-*\"]", sha256_hex(KEY));
    let app = app_with(&extra, &format!("{KEY},unbound-key"));
    let body = |subject: &str| {
        json!({"subject": subject, "tool_calls": [{"name": "read_file", "arguments": {"path": "/a"}}]})
            .to_string()
    };
    let (s, _) = call(&app, post("/v1/evaluate", Some(KEY), body("user-42"))).await;
    assert_eq!(s, StatusCode::OK);
    let (s, b) = call(&app, post("/v1/evaluate", Some(KEY), body("svc-admin"))).await;
    assert_eq!(
        (s, b["error"].as_str()),
        (StatusCode::FORBIDDEN, Some("subject_not_allowed"))
    );
    // Keys without a binding are unrestricted unless require_subject_binding.
    let (s, _) = call(
        &app,
        post("/v1/evaluate", Some("unbound-key"), body("svc-admin")),
    )
    .await;
    assert_eq!(s, StatusCode::OK);

    let strict = app_with(
        &format!("require_subject_binding = true\n{extra}"),
        &format!("{KEY},unbound-key"),
    );
    let (s, _) = call(
        &strict,
        post("/v1/evaluate", Some("unbound-key"), body("user-1")),
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN);
}

/// IMPROVEMENTS 2.3: /v1/verify can be made single-use.
#[tokio::test]
async fn verify_single_use_consumes_jti() {
    for (single_use, second) in [(false, StatusCode::OK), (true, StatusCode::UNAUTHORIZED)] {
        let app = app_with(&format!("verify_single_use = {single_use}"), KEY);
        let (_, ev) = evaluate(&app, "allow_read").await;
        let verify = json!({
            "token": ev["token"]["token"],
            "tool_calls": tool_calls_of("allow_read"),
        })
        .to_string();
        let (s1, _) = call(&app, post("/v1/verify", Some(KEY), verify.clone())).await;
        assert_eq!(s1, StatusCode::OK);
        let (s2, b2) = call(&app, post("/v1/verify", Some(KEY), verify)).await;
        assert_eq!(s2, second, "single_use={single_use} {b2}");
        if single_use {
            assert_eq!(b2["error"], "replayed");
        }
    }
}

#[tokio::test]
async fn metrics_can_require_auth() {
    let open = app(256 * 1024);
    assert_eq!(call(&open, get("/metrics")).await.0, StatusCode::OK);
    let locked = app_with("metrics_require_auth = true", KEY);
    assert_eq!(
        call(&locked, get("/metrics")).await.0,
        StatusCode::UNAUTHORIZED
    );
    let req = Request::get("/metrics")
        .header(header::AUTHORIZATION, format!("Bearer {KEY}"))
        .body(Body::empty())
        .unwrap();
    assert_eq!(call(&locked, req).await.0, StatusCode::OK);
}
