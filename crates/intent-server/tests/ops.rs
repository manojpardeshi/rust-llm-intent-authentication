//! Key rotation, hot reload and rate limiting, exercised in-process.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt as _;
use intent_server::{ApiKeys, AppState, Metrics, ServerConfig, router};
use intent_token::{Jwks, KeyPair};
use serde_json::{Value, json};
use tower::ServiceExt as _;

const KEY: &str = "test-key-123";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(root().join(rel)).unwrap()
}

/// Unique temp dir, removed on drop.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "intent-server-ops-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // Canonical so file-watch events match on macOS (/var -> /private/var).
        Self(dir.canonicalize().unwrap())
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

fn jwks_json(key: &KeyPair) -> String {
    serde_json::to_string(&Jwks {
        keys: vec![key.public_jwk()],
    })
    .unwrap()
}

/// Replaces the key file, as an operator would when rotating.
fn install_key(path: &Path, key: &KeyPair) {
    std::fs::remove_file(path).ok();
    key.save(path).unwrap();
}

struct Env {
    dir: TestDir,
    state: AppState,
    app: Router,
}

impl Env {
    fn new(extra_config: &str) -> (Self, KeyPair) {
        let dir = TestDir::new();
        let key = KeyPair::generate().unwrap();
        install_key(&dir.path("signing.key.json"), &key);
        std::fs::create_dir(dir.path("retired")).unwrap();
        std::fs::write(dir.path("policy.toml"), read("config/policy.example.toml")).unwrap();
        std::fs::write(dir.path("rules.toml"), read("config/rules.example.toml")).unwrap();
        let config = ServerConfig::from_toml_str(&format!(
            "key_path = {:?}\npolicy_path = {:?}\nrules_path = {:?}\n\
             retired_keys_paths = [{:?}]\n{extra_config}",
            dir.path("signing.key.json"),
            dir.path("policy.toml"),
            dir.path("rules.toml"),
            dir.path("retired"),
        ))
        .unwrap();
        let state =
            AppState::from_config(&config, ApiKeys::from_csv(KEY), Some(Metrics::new())).unwrap();
        let app = router(state.clone());
        (Self { dir, state, app }, key)
    }
}

async fn call(app: &Router, req: Request<Body>) -> (StatusCode, Value, axum::http::HeaderMap) {
    let res = app.clone().oneshot(req).await.unwrap();
    let status = res.status();
    let headers = res.headers().clone();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, body, headers)
}

fn post(uri: &str, key: Option<&str>, body: String) -> Request<Body> {
    let mut b = Request::post(uri).header(header::CONTENT_TYPE, "application/json");
    if let Some(k) = key {
        b = b.header(header::AUTHORIZATION, format!("Bearer {k}"));
    }
    b.body(Body::from(body)).unwrap()
}

async fn evaluate_as(
    app: &Router,
    key: Option<&str>,
) -> (StatusCode, Value, axum::http::HeaderMap) {
    call(
        app,
        post(
            "/v1/evaluate",
            key,
            read("examples/scenarios/allow_read.json"),
        ),
    )
    .await
}

async fn evaluate(app: &Router) -> (StatusCode, Value) {
    let (s, b, _) = evaluate_as(app, Some(KEY)).await;
    (s, b)
}

async fn verify(app: &Router, token: &str) -> (StatusCode, Value) {
    let calls = serde_json::from_str::<Value>(&read("examples/scenarios/allow_read.json")).unwrap()
        ["tool_calls"]
        .clone();
    let (s, b, _) = call(
        app,
        post(
            "/v1/verify",
            Some(KEY),
            json!({"token": token, "tool_calls": calls}).to_string(),
        ),
    )
    .await;
    (s, b)
}

async fn jwks_kids(app: &Router) -> Vec<String> {
    let (s, b, _) = call(
        app,
        Request::get("/.well-known/jwks.json")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    b["keys"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| k["kid"].as_str().unwrap().to_owned())
        .collect()
}

fn token_kid(token: &str) -> String {
    use base64::Engine as _;
    let header: Value = serde_json::from_slice(
        &base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(token.split('.').next().unwrap())
            .unwrap(),
    )
    .unwrap();
    header["kid"].as_str().unwrap().to_owned()
}

// --- rotation -----------------------------------------------------------

#[tokio::test]
async fn key_rotation_with_retired_keys() {
    let (env, key_a) = Env::new("");
    let (s, b) = evaluate(&env.app).await;
    assert_eq!(s, StatusCode::OK);
    let old_token = b["token"]["token"].as_str().unwrap().to_owned();
    assert_eq!(token_kid(&old_token), key_a.kid());

    // Rotate: B becomes the signing key, A's public JWKS goes to retired/.
    let key_b = KeyPair::generate().unwrap();
    install_key(&env.dir.path("signing.key.json"), &key_b);
    std::fs::write(env.dir.path("retired/a.json"), jwks_json(&key_a)).unwrap();
    env.state.reload().unwrap();

    assert_eq!(
        jwks_kids(&env.app).await,
        vec![key_b.kid().to_owned(), key_a.kid().to_owned()]
    );
    let (s, b) = verify(&env.app, &old_token).await;
    assert_eq!((s, &b["valid"]), (StatusCode::OK, &json!(true)));

    let (_, b) = evaluate(&env.app).await;
    let new_token = b["token"]["token"].as_str().unwrap().to_owned();
    assert_eq!(token_kid(&new_token), key_b.kid());
    assert_eq!(verify(&env.app, &new_token).await.0, StatusCode::OK);

    // Retire A for good: old tokens stop verifying.
    std::fs::remove_file(env.dir.path("retired/a.json")).unwrap();
    env.state.reload().unwrap();
    assert_eq!(jwks_kids(&env.app).await, vec![key_b.kid().to_owned()]);
    let (s, b) = verify(&env.app, &old_token).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    assert_eq!(b["error"], "unknown_key");
    assert_eq!(verify(&env.app, &new_token).await.0, StatusCode::OK);
}

#[tokio::test]
async fn bad_retired_keys_fail_reload_and_keep_serving() {
    let (env, key_a) = Env::new("");
    let (_, b) = evaluate(&env.app).await;
    let token = b["token"]["token"].as_str().unwrap().to_owned();

    // Retired key equal to the active key.
    std::fs::write(env.dir.path("retired/a.json"), jwks_json(&key_a)).unwrap();
    assert!(env.state.reload().is_err());
    // Private key material in a retired file.
    std::fs::write(
        env.dir.path("retired/a.json"),
        json!({"keys": [{"kty":"OKP","crv":"Ed25519","alg":"EdDSA","use":"sig",
            "kid":"x","x": key_a.public_x(), "d": "secret"}]})
        .to_string(),
    )
    .unwrap();
    let err = env.state.reload().unwrap_err().to_string();
    assert!(err.contains("private key material"), "{err}");
    // Garbage.
    std::fs::write(env.dir.path("retired/a.json"), "not json").unwrap();
    assert!(env.state.reload().is_err());

    assert_eq!(verify(&env.app, &token).await.0, StatusCode::OK);
    assert_eq!(jwks_kids(&env.app).await, vec![key_a.kid().to_owned()]);
}

// --- reload -------------------------------------------------------------

fn deny_reads_policy() -> String {
    format!(
        "[[rule]]\ncategory = \"ReadData\"\ndecision = \"deny\"\nreason = \"reads frozen\"\n\n{}",
        read("config/policy.example.toml")
    )
}

#[tokio::test]
async fn reload_applies_new_policy_and_survives_invalid_one() {
    let (env, _) = Env::new("");
    assert_eq!(evaluate(&env.app).await.0, StatusCode::OK);

    std::fs::write(env.dir.path("policy.toml"), deny_reads_policy()).unwrap();
    // Nothing changes until a reload is triggered.
    assert_eq!(evaluate(&env.app).await.0, StatusCode::OK);
    env.state.reload().unwrap();
    let (s, b) = evaluate(&env.app).await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    assert_eq!(b["reason"], "reads frozen");

    // An invalid policy is rejected; the previous (deny) policy still applies.
    std::fs::write(
        env.dir.path("policy.toml"),
        "[[rule]]\ncategory = \"Nope\"\n",
    )
    .unwrap();
    assert!(env.state.reload().is_err());
    assert_eq!(evaluate(&env.app).await.0, StatusCode::FORBIDDEN);

    // Same for invalid rules.
    std::fs::write(
        env.dir.path("policy.toml"),
        read("config/policy.example.toml"),
    )
    .unwrap();
    std::fs::write(env.dir.path("rules.toml"), "garbage = [").unwrap();
    assert!(env.state.reload().is_err());
    assert_eq!(evaluate(&env.app).await.0, StatusCode::FORBIDDEN);

    // Fixing the files recovers.
    std::fs::write(
        env.dir.path("rules.toml"),
        read("config/rules.example.toml"),
    )
    .unwrap();
    env.state.reload().unwrap();
    assert_eq!(evaluate(&env.app).await.0, StatusCode::OK);

    let (_, text, _) = call(
        &env.app,
        Request::get("/metrics").body(Body::empty()).unwrap(),
    )
    .await;
    let text = text.as_str().unwrap();
    assert!(
        text.contains("intent_reloads_total{result=\"ok\"} 2"),
        "{text}"
    );
    assert!(
        text.contains("intent_reloads_total{result=\"error\"} 2"),
        "{text}"
    );
}

#[tokio::test]
async fn new_state_has_nothing_to_reload() {
    use std::sync::Arc;
    let key = KeyPair::generate().unwrap();
    let issuer = intent_token::JwtIssuer::new(
        intent_token::IssuerConfig::new("intent-server", "tool-gateway"),
        &key,
    )
    .unwrap();
    let jwks = issuer.jwks();
    let classifier = intent_classify::HybridClassifier::new(
        intent_classify::RuleClassifier::from_path(&root().join("config/rules.example.toml"))
            .unwrap(),
        None,
    );
    let policy =
        intent_policy::Policy::from_path(&root().join("config/policy.example.toml")).unwrap();
    let state = AppState::new(
        intent_core::Evaluator::new(Arc::new(classifier), Arc::new(policy), Arc::new(issuer)),
        &jwks,
        "intent-server",
        "tool-gateway",
        ApiKeys::from_csv(KEY),
        None,
    )
    .unwrap();
    let err = state.reload().unwrap_err().to_string();
    assert!(err.contains("no config to reload from"), "{err}");
    // Still serves.
    assert_eq!(evaluate(&router(state)).await.0, StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn file_watch_triggers_reload() {
    let (env, _) = Env::new("watch = true");
    let _guard = intent_server::spawn_reload_triggers(&env.state).unwrap();
    assert_eq!(evaluate(&env.app).await.0, StatusCode::OK);

    // Give the watcher a moment to arm, then replace the file the way editors
    // do (write a temp file and rename over the original).
    tokio::time::sleep(Duration::from_millis(300)).await;
    let tmp = env.dir.path("policy.toml.tmp");
    std::fs::write(&tmp, deny_reads_policy()).unwrap();
    std::fs::rename(&tmp, env.dir.path("policy.toml")).unwrap();

    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if evaluate(&env.app).await.0 == StatusCode::FORBIDDEN {
            break;
        }
        assert!(Instant::now() < deadline, "watcher never reloaded");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

// --- rate limiting ------------------------------------------------------

const LIMIT: &str = "[rate_limit]\nper_second = 1\nburst = 2\n";

#[tokio::test]
async fn rate_limit_burst_then_429_with_retry_after() {
    let (env, _) = Env::new(LIMIT);
    for _ in 0..2 {
        assert_eq!(evaluate_as(&env.app, Some(KEY)).await.0, StatusCode::OK);
    }
    let (s, b, h) = evaluate_as(&env.app, Some(KEY)).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(b, json!({"error": "rate_limited"}));
    let retry: u64 = h[header::RETRY_AFTER].to_str().unwrap().parse().unwrap();
    assert!((1..=2).contains(&retry), "{retry}");

    // /v1/verify shares the key's bucket; public endpoints are exempt.
    assert_eq!(verify(&env.app, "x").await.0, StatusCode::TOO_MANY_REQUESTS);
    for uri in ["/healthz", "/.well-known/jwks.json", "/metrics"] {
        for _ in 0..5 {
            let (s, ..) = call(&env.app, Request::get(uri).body(Body::empty()).unwrap()).await;
            assert_eq!(s, StatusCode::OK, "{uri}");
        }
    }
    let (_, text, _) = call(
        &env.app,
        Request::get("/metrics").body(Body::empty()).unwrap(),
    )
    .await;
    assert!(
        text.as_str()
            .unwrap()
            .contains("intent_rate_limited_total 2")
    );
}

#[tokio::test]
async fn rate_limit_buckets_are_per_api_key_and_unauthorized_is_free() {
    let dir = TestDir::new();
    let key = KeyPair::generate().unwrap();
    install_key(&dir.path("signing.key.json"), &key);
    let config = ServerConfig::from_toml_str(&format!(
        "key_path = {:?}\npolicy_path = {:?}\nrules_path = {:?}\n{LIMIT}",
        dir.path("signing.key.json"),
        root().join("config/policy.example.toml"),
        root().join("config/rules.example.toml"),
    ))
    .unwrap();
    let app =
        router(AppState::from_config(&config, ApiKeys::from_csv("key-a,key-b"), None).unwrap());

    // Unauthenticated garbage gets 401 and never touches any bucket.
    for _ in 0..10 {
        assert_eq!(
            evaluate_as(&app, Some("wrong")).await.0,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(evaluate_as(&app, None).await.0, StatusCode::UNAUTHORIZED);
    }
    for _ in 0..2 {
        assert_eq!(evaluate_as(&app, Some("key-a")).await.0, StatusCode::OK);
    }
    assert_eq!(
        evaluate_as(&app, Some("key-a")).await.0,
        StatusCode::TOO_MANY_REQUESTS
    );
    // key-b is independent and still has its full burst.
    for _ in 0..2 {
        assert_eq!(evaluate_as(&app, Some("key-b")).await.0, StatusCode::OK);
    }
    assert_eq!(
        evaluate_as(&app, Some("key-b")).await.0,
        StatusCode::TOO_MANY_REQUESTS
    );
    // A bad key while key-a is exhausted is still 401, not 429.
    assert_eq!(
        evaluate_as(&app, Some("wrong")).await.0,
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn rate_limit_unauthenticated_mode_keys_by_client_ip() {
    let dir = TestDir::new();
    let key = KeyPair::generate().unwrap();
    install_key(&dir.path("signing.key.json"), &key);
    let config = ServerConfig::from_toml_str(&format!(
        "key_path = {:?}\npolicy_path = {:?}\nrules_path = {:?}\n\
         allow_unauthenticated = true\n{LIMIT}",
        dir.path("signing.key.json"),
        root().join("config/policy.example.toml"),
        root().join("config/rules.example.toml"),
    ))
    .unwrap();
    let app = router(AppState::from_config(&config, ApiKeys::default(), None).unwrap());

    let from = |ip: &str| {
        let mut req = post(
            "/v1/evaluate",
            None,
            read("examples/scenarios/allow_read.json"),
        );
        let addr: SocketAddr = format!("{ip}:4000").parse().unwrap();
        req.extensions_mut().insert(ConnectInfo(addr));
        req
    };
    for _ in 0..2 {
        assert_eq!(call(&app, from("10.0.0.1")).await.0, StatusCode::OK);
    }
    assert_eq!(
        call(&app, from("10.0.0.1")).await.0,
        StatusCode::TOO_MANY_REQUESTS
    );
    assert_eq!(call(&app, from("10.0.0.2")).await.0, StatusCode::OK);
    // Same IP, different source port: same bucket.
    let mut req = from("10.0.0.1");
    req.extensions_mut()
        .insert(ConnectInfo("10.0.0.1:5000".parse::<SocketAddr>().unwrap()));
    assert_eq!(call(&app, req).await.0, StatusCode::TOO_MANY_REQUESTS);
    // No ConnectInfo at all: shared bucket, no panic.
    for _ in 0..2 {
        let req = post(
            "/v1/evaluate",
            None,
            read("examples/scenarios/allow_read.json"),
        );
        assert_eq!(call(&app, req).await.0, StatusCode::OK);
    }
    let req = post(
        "/v1/evaluate",
        None,
        read("examples/scenarios/allow_read.json"),
    );
    assert_eq!(call(&app, req).await.0, StatusCode::TOO_MANY_REQUESTS);
}

// --- evaluate time budget -------------------------------------------------

struct SlowClassifier;

#[async_trait::async_trait]
impl intent_core::IntentClassifier for SlowClassifier {
    async fn classify(
        &self,
        _: &intent_core::IntentRequest,
    ) -> intent_core::Result<intent_core::Intent> {
        tokio::time::sleep(Duration::from_secs(10)).await;
        unreachable!("the budget should have expired first")
    }
}

/// IMPROVEMENTS 3 (timeout mismatch): a classifier slower than the budget
/// yields a fail-closed deny, not a 408 from the outer request timeout.
#[tokio::test]
async fn slow_classification_fails_closed_within_budget() {
    use std::sync::Arc;
    let key = KeyPair::generate().unwrap();
    let issuer = intent_token::JwtIssuer::new(
        intent_token::IssuerConfig::new("intent-server", "tool-gateway"),
        &key,
    )
    .unwrap();
    let jwks = issuer.jwks();
    let policy =
        intent_policy::Policy::from_path(&root().join("config/policy.example.toml")).unwrap();
    let state = AppState::new(
        intent_core::Evaluator::new(Arc::new(SlowClassifier), Arc::new(policy), Arc::new(issuer)),
        &jwks,
        "intent-server",
        "tool-gateway",
        ApiKeys::from_csv(KEY),
        None,
    )
    .unwrap()
    .with_evaluate_budget(Duration::from_millis(100));
    let started = Instant::now();
    let (s, b) = evaluate(&router(state)).await;
    assert_eq!(s, StatusCode::FORBIDDEN, "{b}");
    assert_eq!(b["decision"], "deny");
    assert_eq!(b["intent"]["action"], "classification_timed_out");
    assert!(b.get("token").is_none());
    assert!(started.elapsed() < Duration::from_secs(5));
}
