mod common;
use std::time::Duration;

use common::*;
use intent_token::KeyPair;
use intent_verifier::{RemoteJwksOptions, Verifier, VerifyError};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer};

async fn server(mode: Mode) -> (MockServer, Dynamic, String) {
    let server = MockServer::start().await;
    let dynamic = Dynamic::new(mode);
    Mock::given(method("GET"))
        .and(path("/jwks.json"))
        .respond_with(dynamic.clone())
        .mount(&server)
        .await;
    let url = format!("{}/jwks.json", server.uri());
    (server, dynamic, url)
}

async fn fetches(server: &MockServer) -> usize {
    server.received_requests().await.unwrap().len()
}

fn opts(refresh: Duration, min: Duration) -> RemoteJwksOptions {
    RemoteJwksOptions {
        refresh_interval: refresh,
        min_refresh_interval: min,
        ..Default::default()
    }
}

const LONG: Duration = Duration::from_secs(3600);

async fn verifier(url: &str, o: RemoteJwksOptions) -> Verifier {
    Verifier::from_jwks_url(url, ISS, AUD, o).await.unwrap()
}

async fn cat(v: &Verifier, token: &str) -> Result<(), &'static str> {
    v.verify(token, &calls(), None)
        .await
        .map(|_| ())
        .map_err(|e| e.category())
}

#[tokio::test]
async fn initial_fetch_then_verify() {
    let key = KeyPair::generate().unwrap();
    let (server, _d, url) = server(Mode::Keys(jwks(&key))).await;
    let v = verifier(&url, opts(LONG, Duration::ZERO)).await;
    assert_eq!(fetches(&server).await, 1);
    for _ in 0..3 {
        assert_eq!(cat(&v, &mint(&key, &["data:read"])).await, Ok(()));
    }
    assert_eq!(fetches(&server).await, 1, "no refetch while fresh");
}

#[tokio::test]
async fn initial_fetch_failure_is_an_error() {
    let (_server, _d, url) = server(Mode::Fail).await;
    let err = Verifier::from_jwks_url(&url, ISS, AUD, RemoteJwksOptions::default())
        .await
        .err()
        .unwrap();
    assert!(matches!(err, VerifyError::Fetch(_)), "{err}");
}

#[tokio::test]
async fn key_rotation_refreshes_once_and_replaces_keys() {
    let old = KeyPair::generate().unwrap();
    let new = KeyPair::generate().unwrap();
    let (server, d, url) = server(Mode::Keys(jwks(&old))).await;
    let v = verifier(&url, opts(LONG, Duration::ZERO)).await;
    let old_token = mint(&old, &["data:read"]);
    assert_eq!(cat(&v, &old_token).await, Ok(()));

    d.set(Mode::Keys(jwks(&new)));
    let new_token = mint(&new, &["data:read"]);
    assert_eq!(cat(&v, &new_token).await, Ok(()));
    assert_eq!(fetches(&server).await, 2);

    // Key set was replaced wholesale: the rotated-out key no longer works.
    assert_eq!(cat(&v, &old_token).await, Err("unknown_key"));
}

#[tokio::test]
async fn junk_kid_flood_causes_at_most_one_extra_fetch() {
    let key = KeyPair::generate().unwrap();
    let (server, _d, url) = server(Mode::Keys(jwks(&key))).await;
    let v = verifier(&url, opts(LONG, Duration::from_millis(300))).await;
    tokio::time::sleep(Duration::from_millis(350)).await;

    let mut tasks = Vec::new();
    for _ in 0..20 {
        let v = v.clone();
        // Each task signs with its own, never-published key.
        let token = mint(&KeyPair::generate().unwrap(), &["data:read"]);
        tasks.push(tokio::spawn(async move { cat(&v, &token).await }));
    }
    for t in tasks {
        assert_eq!(t.await.unwrap(), Err("unknown_key"));
    }
    let n = fetches(&server).await;
    assert!(n <= 2, "expected at most 1 extra fetch, saw {} total", n);

    // Within the throttle window further junk kids do not fetch at all.
    let before = fetches(&server).await;
    for _ in 0..5 {
        let token = mint(&KeyPair::generate().unwrap(), &["data:read"]);
        assert_eq!(cat(&v, &token).await, Err("unknown_key"));
    }
    assert_eq!(fetches(&server).await, before);
}

#[tokio::test]
async fn unknown_kid_right_after_fetch_is_throttled() {
    let key = KeyPair::generate().unwrap();
    let (server, _d, url) = server(Mode::Keys(jwks(&key))).await;
    let v = verifier(&url, opts(LONG, Duration::from_secs(60))).await;
    let token = mint(&KeyPair::generate().unwrap(), &["data:read"]);
    assert_eq!(cat(&v, &token).await, Err("unknown_key"));
    assert_eq!(fetches(&server).await, 1);
}

#[tokio::test]
async fn refresh_failure_keeps_old_keys() {
    let key = KeyPair::generate().unwrap();
    let (server, d, url) = server(Mode::Keys(jwks(&key))).await;
    let v = verifier(&url, opts(Duration::from_millis(10), Duration::ZERO)).await;
    d.set(Mode::Fail);
    tokio::time::sleep(Duration::from_millis(30)).await;

    assert_eq!(cat(&v, &mint(&key, &["data:read"])).await, Ok(()));
    assert!(fetches(&server).await >= 2, "a refresh was attempted");
    let other = mint(&KeyPair::generate().unwrap(), &["data:read"]);
    assert_eq!(cat(&v, &other).await, Err("unknown_key"));
    assert_eq!(cat(&v, &mint(&key, &["data:read"])).await, Ok(()));
}

#[tokio::test]
async fn bad_refresh_body_keeps_old_keys() {
    let key = KeyPair::generate().unwrap();
    let (_server, d, url) = server(Mode::Keys(jwks(&key))).await;
    let v = verifier(&url, opts(Duration::from_millis(10), Duration::ZERO)).await;
    d.set(Mode::Raw(r#"{"keys":[]}"#.into()));
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert_eq!(cat(&v, &mint(&key, &["data:read"])).await, Ok(()));
}

#[tokio::test]
async fn oversized_body_rejected() {
    let key = KeyPair::generate().unwrap();
    let (_server, d, url) = server(Mode::Keys(jwks(&key))).await;
    let o = RemoteJwksOptions {
        max_body_bytes: 64,
        ..Default::default()
    };
    let err = Verifier::from_jwks_url(&url, ISS, AUD, o)
        .await
        .err()
        .unwrap();
    assert!(
        matches!(&err, VerifyError::Fetch(m) if m.contains("exceeds")),
        "{err}"
    );

    d.set(Mode::Raw("x".repeat(100_000)));
    let err = Verifier::from_jwks_url(&url, ISS, AUD, RemoteJwksOptions::default())
        .await
        .err()
        .unwrap();
    assert!(matches!(err, VerifyError::Fetch(_)));
}

#[tokio::test]
async fn redirects_are_not_followed() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(
            wiremock::ResponseTemplate::new(302).insert_header("location", "http://127.0.0.1:1/x"),
        )
        .mount(&server)
        .await;
    let url = format!("{}/jwks.json", server.uri());
    let err = Verifier::from_jwks_url(&url, ISS, AUD, RemoteJwksOptions::default())
        .await
        .err()
        .unwrap();
    assert!(
        matches!(&err, VerifyError::Fetch(m) if m.contains("302")),
        "{err}"
    );
}

#[tokio::test]
async fn url_scheme_policy() {
    let o = RemoteJwksOptions::default();
    for url in [
        "http://example.com/jwks.json",
        "http://10.0.0.1/jwks.json",
        "ftp://localhost/jwks.json",
        "not a url",
    ] {
        let err = Verifier::from_jwks_url(url, ISS, AUD, o.clone())
            .await
            .err()
            .unwrap();
        assert!(matches!(err, VerifyError::Config(_)), "{url}: {err}");
    }
    // Loopback http passes validation (and then fails to connect: nothing listens).
    for url in [
        "http://localhost:1/j",
        "http://127.0.0.1:1/j",
        "http://[::1]:1/j",
    ] {
        let err = Verifier::from_jwks_url(url, ISS, AUD, o.clone())
            .await
            .err()
            .unwrap();
        assert!(matches!(err, VerifyError::Fetch(_)), "{url}: {err}");
    }
    // allow_insecure_http lifts the restriction for other hosts.
    let o = RemoteJwksOptions {
        allow_insecure_http: true,
        timeout: Duration::from_millis(200),
        ..Default::default()
    };
    let err = Verifier::from_jwks_url("http://127.0.0.2:1/j", ISS, AUD, o)
        .await
        .err()
        .unwrap();
    assert!(matches!(err, VerifyError::Fetch(_)));
}

#[tokio::test]
async fn refresh_interval_expiry_refetches() {
    let key = KeyPair::generate().unwrap();
    let (server, _d, url) = server(Mode::Keys(jwks(&key))).await;
    let v = verifier(&url, opts(Duration::from_millis(50), Duration::ZERO)).await;
    assert_eq!(cat(&v, &mint(&key, &["data:read"])).await, Ok(()));
    assert_eq!(fetches(&server).await, 1);
    tokio::time::sleep(Duration::from_millis(80)).await;
    assert_eq!(cat(&v, &mint(&key, &["data:read"])).await, Ok(()));
    assert_eq!(fetches(&server).await, 2);
}
