mod common;
use common::*;
use intent_token::KeyPair;
use intent_verifier::{Verifier, VerifyError};

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
