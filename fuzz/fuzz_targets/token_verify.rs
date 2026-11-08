#![no_main]

use std::sync::OnceLock;

use intent_token::{IssuerConfig, JwtIssuer, KeyPair, TokenVerifier};
use libfuzzer_sys::fuzz_target;

const ISS: &str = "intent-server";
const AUD: &str = "tool-gateway";

fn verifier() -> &'static TokenVerifier {
    static VERIFIER: OnceLock<TokenVerifier> = OnceLock::new();
    VERIFIER.get_or_init(|| {
        let key = KeyPair::from_seed([7u8; 32]);
        let issuer = JwtIssuer::new(IssuerConfig::new(ISS, AUD), &key).expect("fixed key is valid");
        TokenVerifier::from_jwks(&issuer.jwks(), ISS, AUD).expect("jwks from fixed key")
    })
}

// Arbitrary text as a bearer token. Without the signing key, verification
// must never panic and must never succeed.
fuzz_target!(|data: &[u8]| {
    let Ok(token) = std::str::from_utf8(data) else {
        return;
    };
    let result = verifier().verify(token, &[], None);
    assert!(result.is_err(), "verify accepted an arbitrary token");
});
