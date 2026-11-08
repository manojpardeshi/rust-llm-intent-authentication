mod common;
use common::*;
use intent_token::KeyPair;
use std::sync::Arc;

use intent_verifier::{InMemoryReplayStore, ReplayError, ReplayStore, Verifier, VerifyError};

fn category(r: Result<intent_token::Claims, VerifyError>) -> &'static str {
    r.unwrap_err().category()
}

#[tokio::test]
async fn valid_token_verifies() {
    let key = KeyPair::generate().unwrap();
    let v = Verifier::from_jwks(&jwks(&key), ISS, AUD).unwrap();
    let token = mint(&key, &["data:read"]);
    let claims = v.verify(&token, &calls(), Some("data:read")).await.unwrap();
    assert_eq!(claims.sub, "user-42");
}

#[tokio::test]
async fn tampered_calls_rejected() {
    let key = KeyPair::generate().unwrap();
    let v = Verifier::from_jwks(&jwks(&key), ISS, AUD).unwrap();
    let token = mint(&key, &["data:read"]);
    assert_eq!(
        category(v.verify(&token, &tampered(), None).await),
        "request_mismatch"
    );
}

#[tokio::test]
async fn wrong_scope_rejected() {
    let key = KeyPair::generate().unwrap();
    let v = Verifier::from_jwks(&jwks(&key), ISS, AUD).unwrap();
    let token = mint(&key, &["data:read"]);
    assert_eq!(
        category(v.verify(&token, &calls(), Some("data:write")).await),
        "missing_scope"
    );
}

#[tokio::test]
async fn wrong_audience_and_garbage() {
    let key = KeyPair::generate().unwrap();
    let v = Verifier::from_jwks(&jwks(&key), ISS, "other-gateway").unwrap();
    let token = mint(&key, &["data:read"]);
    assert_eq!(
        category(v.verify(&token, &calls(), None).await),
        "wrong_audience"
    );
    assert_eq!(
        category(v.verify("not-a-jwt", &calls(), None).await),
        "malformed_token"
    );
}

#[tokio::test]
async fn unknown_kid_rejected_without_fetch() {
    // A static verifier has no URL to fetch from; unknown keys just fail.
    let trusted = KeyPair::generate().unwrap();
    let other = KeyPair::generate().unwrap();
    let v = Verifier::from_jwks(&jwks(&trusted), ISS, AUD).unwrap();
    let token = mint(&other, &["data:read"]);
    assert_eq!(
        category(v.verify(&token, &calls(), None).await),
        "unknown_key"
    );
}

fn with_store(key: &KeyPair, capacity: usize) -> Verifier {
    Verifier::from_jwks(&jwks(key), ISS, AUD)
        .unwrap()
        .with_replay_store(Arc::new(InMemoryReplayStore::new(capacity)))
}

#[tokio::test]
async fn second_use_is_replayed() {
    let key = KeyPair::generate().unwrap();
    let v = with_store(&key, 10);
    let token = mint(&key, &["data:read"]);
    v.verify(&token, &calls(), None).await.unwrap();
    assert_eq!(category(v.verify(&token, &calls(), None).await), "replayed");
    // Clones share the store.
    assert_eq!(
        category(v.clone().verify(&token, &calls(), None).await),
        "replayed"
    );
}

#[tokio::test]
async fn failed_checks_do_not_consume_jti() {
    let key = KeyPair::generate().unwrap();
    let v = with_store(&key, 10);
    let token = mint(&key, &["data:read"]);
    assert_eq!(
        category(v.verify(&token, &tampered(), None).await),
        "request_mismatch"
    );
    assert_eq!(
        category(v.verify(&token, &calls(), Some("data:write")).await),
        "missing_scope"
    );
    v.verify(&token, &calls(), Some("data:read")).await.unwrap();
    assert_eq!(category(v.verify(&token, &calls(), None).await), "replayed");
}

#[tokio::test]
async fn without_store_reuse_is_allowed() {
    let key = KeyPair::generate().unwrap();
    let v = Verifier::from_jwks(&jwks(&key), ISS, AUD).unwrap();
    let token = mint(&key, &["data:read"]);
    v.verify(&token, &calls(), None).await.unwrap();
    v.verify(&token, &calls(), None).await.unwrap();
}

#[tokio::test]
async fn full_store_fails_closed() {
    let key = KeyPair::generate().unwrap();
    let v = with_store(&key, 1);
    v.verify(&mint(&key, &["data:read"]), &calls(), None)
        .await
        .unwrap();
    let err = v
        .verify(&mint(&key, &["data:read"]), &calls(), None)
        .await
        .unwrap_err();
    assert!(matches!(err, VerifyError::Replay(ReplayError::Full)));
    assert_eq!(err.category(), "replay_store_unavailable");
}

#[tokio::test]
async fn backend_error_fails_closed() {
    struct Broken;
    impl ReplayStore for Broken {
        fn check_and_insert(&self, _: &str, _: i64) -> Result<(), ReplayError> {
            Err(ReplayError::Backend("down".into()))
        }
    }
    let key = KeyPair::generate().unwrap();
    let v = Verifier::from_jwks(&jwks(&key), ISS, AUD)
        .unwrap()
        .with_replay_store(Arc::new(Broken));
    let token = mint(&key, &["data:read"]);
    assert_eq!(
        category(v.verify(&token, &calls(), None).await),
        "replay_store_unavailable"
    );
}
