//! Issue and verify intent-bound EdDSA JWTs.
//!
//! A token carries the classified intent, the scopes granted by policy and a
//! `req_hash` over the exact tool calls that were evaluated. Verifiers
//! recompute `req_hash` from the tool calls they receive, so a token cannot be
//! replayed for a different action.

mod keys;

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use intent_core::{
    Decision, Intent, IntentCategory, IntentRequest, IssuedToken, TokenIssuer, ToolCall,
    encode_digest, request_digest, tool_calls_digest,
};
use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};

pub use keys::{Jwk, Jwks, KeyPair};

/// Hard upper bound on token lifetime, regardless of policy.
pub const DEFAULT_MAX_TTL: Duration = Duration::from_secs(300);

#[derive(Debug, thiserror::Error)]
pub enum TokenError {
    #[error("key error: {0}")]
    Key(String),
    #[error("refusing to issue a token for a non-allow decision")]
    NotAllowed,
    #[error("request digest: {0}")]
    Digest(#[from] intent_core::Error),
    #[error("token is missing a kid header")]
    MissingKid,
    #[error("unknown signing key {0:?}")]
    UnknownKey(String),
    #[error("invalid token: {0}")]
    Jwt(#[from] jsonwebtoken::errors::Error),
    #[error("token does not grant scope {0:?}")]
    MissingScope(String),
    #[error("tool calls do not match the request the token was issued for")]
    RequestMismatch,
}

impl TokenError {
    /// Short, token-free category for a verification failure, suitable for
    /// API responses and logs.
    pub fn category(&self) -> &'static str {
        use jsonwebtoken::errors::ErrorKind;
        match self {
            Self::Jwt(j) => match j.kind() {
                ErrorKind::ExpiredSignature => "expired",
                ErrorKind::ImmatureSignature => "not_yet_valid",
                ErrorKind::InvalidSignature => "bad_signature",
                ErrorKind::InvalidIssuer => "wrong_issuer",
                ErrorKind::InvalidAudience => "wrong_audience",
                _ => "malformed_token",
            },
            Self::MissingKid | Self::UnknownKey(_) => "unknown_key",
            Self::MissingScope(_) => "missing_scope",
            Self::RequestMismatch => "request_mismatch",
            Self::Digest(_) | Self::Key(_) | Self::NotAllowed => "invalid_token",
        }
    }
}

/// The `intent` claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntentClaim {
    pub category: IntentCategory,
    pub action: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource: Option<String>,
    pub confidence: f32,
}

/// JWT claims issued by this framework.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Claims {
    pub iss: String,
    pub sub: String,
    pub aud: String,
    pub iat: i64,
    pub nbf: i64,
    pub exp: i64,
    pub jti: String,
    pub intent: IntentClaim,
    /// Space-separated scopes (RFC 8693 style).
    pub scope: String,
    /// Base64url SHA-256 of the canonical tool calls.
    pub req_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sid: Option<String>,
}

impl Claims {
    pub fn scopes(&self) -> impl Iterator<Item = &str> {
        self.scope.split_ascii_whitespace()
    }

    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes().any(|s| s == scope)
    }
}

#[derive(Debug, Clone)]
pub struct IssuerConfig {
    pub issuer: String,
    /// Tool gateway the token is intended for.
    pub audience: String,
    pub max_ttl: Duration,
}

impl IssuerConfig {
    pub fn new(issuer: impl Into<String>, audience: impl Into<String>) -> Self {
        Self {
            issuer: issuer.into(),
            audience: audience.into(),
            max_ttl: DEFAULT_MAX_TTL,
        }
    }
}

/// Signs intent tokens with an Ed25519 key.
pub struct JwtIssuer {
    config: IssuerConfig,
    kid: String,
    encoding_key: EncodingKey,
    jwk: Jwk,
}

impl JwtIssuer {
    pub fn new(config: IssuerConfig, key: &KeyPair) -> Result<Self, TokenError> {
        Ok(Self {
            config,
            kid: key.kid().to_owned(),
            encoding_key: EncodingKey::from_ed_der(&key.pkcs8_der()?),
            jwk: key.public_jwk(),
        })
    }

    /// Public keys for verifiers.
    pub fn jwks(&self) -> Jwks {
        Jwks {
            keys: vec![self.jwk.clone()],
        }
    }

    fn sign(&self, claims: &Claims) -> Result<String, TokenError> {
        let mut header = Header::new(Algorithm::EdDSA);
        header.kid = Some(self.kid.clone());
        Ok(jsonwebtoken::encode(&header, claims, &self.encoding_key)?)
    }
}

impl TokenIssuer for JwtIssuer {
    fn issue(
        &self,
        request: &IntentRequest,
        intent: &Intent,
        decision: &Decision,
    ) -> intent_core::Result<IssuedToken> {
        let Decision::Allow { scopes, ttl } = decision else {
            return Err(TokenError::NotAllowed.into());
        };
        let now = unix_now();
        let ttl = (*ttl).min(self.config.max_ttl).as_secs() as i64;
        let claims = Claims {
            iss: self.config.issuer.clone(),
            sub: request.subject.clone(),
            aud: self.config.audience.clone(),
            iat: now,
            nbf: now,
            exp: now + ttl,
            jti: uuid::Uuid::now_v7().to_string(),
            intent: IntentClaim {
                category: intent.category,
                action: intent.action.clone(),
                resource: intent.resource.clone(),
                confidence: intent.confidence,
            },
            scope: scopes.join(" "),
            req_hash: encode_digest(&request_digest(request)?),
            sid: request.session_id.clone(),
        };
        let token = self.sign(&claims)?;
        Ok(IssuedToken {
            token,
            jti: claims.jti,
            expires_at: claims.exp,
        })
    }
}

impl From<TokenError> for intent_core::Error {
    fn from(e: TokenError) -> Self {
        match e {
            TokenError::Digest(inner) => inner,
            other => intent_core::Error::Token(other.to_string()),
        }
    }
}

/// Verifies intent tokens against a set of trusted public keys.
pub struct TokenVerifier {
    keys: HashMap<String, DecodingKey>,
    validation: Validation,
}

impl TokenVerifier {
    pub fn from_jwks(jwks: &Jwks, issuer: &str, audience: &str) -> Result<Self, TokenError> {
        let mut keys = HashMap::new();
        for jwk in &jwks.keys {
            if jwk.kty != "OKP" || jwk.crv != "Ed25519" || jwk.alg != "EdDSA" {
                continue;
            }
            keys.insert(jwk.kid.clone(), DecodingKey::from_ed_components(&jwk.x)?);
        }

        let mut validation = Validation::new(Algorithm::EdDSA);
        validation.set_issuer(&[issuer]);
        validation.set_audience(&[audience]);
        validation.set_required_spec_claims(&["exp", "nbf", "iat", "iss", "aud", "sub"]);
        validation.validate_nbf = true;
        validation.leeway = 5;
        Ok(Self { keys, validation })
    }

    /// Verifies signature, standard claims, an optional required scope, and
    /// that `tool_calls` are exactly what the token was issued for.
    pub fn verify(
        &self,
        token: &str,
        tool_calls: &[ToolCall],
        required_scope: Option<&str>,
    ) -> Result<Claims, TokenError> {
        let header = jsonwebtoken::decode_header(token)?;
        let kid = header.kid.ok_or(TokenError::MissingKid)?;
        let key = self
            .keys
            .get(&kid)
            .ok_or_else(|| TokenError::UnknownKey(kid.clone()))?;
        let claims = jsonwebtoken::decode::<Claims>(token, key, &self.validation)?.claims;

        if let Some(scope) = required_scope
            && !claims.has_scope(scope)
        {
            return Err(TokenError::MissingScope(scope.to_owned()));
        }
        if claims.req_hash != encode_digest(&tool_calls_digest(tool_calls)?) {
            return Err(TokenError::RequestMismatch);
        }
        Ok(claims)
    }
}

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before 1970")
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use intent_core::ClassifierSource;
    use serde_json::json;

    const ISS: &str = "intent-server";
    const AUD: &str = "tool-gateway";

    fn request() -> IntentRequest {
        IntentRequest {
            subject: "user-42".into(),
            session_id: Some("sess-1".into()),
            messages: vec![],
            tool_calls: vec![ToolCall {
                name: "read_file".into(),
                arguments: json!({"path": "/srv/report.txt"}),
            }],
            metadata: Default::default(),
        }
    }

    fn intent() -> Intent {
        Intent {
            category: IntentCategory::ReadData,
            action: "read_file".into(),
            resource: Some("/srv/report.txt".into()),
            confidence: 0.95,
            source: ClassifierSource::Rules,
        }
    }

    fn allow(ttl: u64) -> Decision {
        Decision::Allow {
            scopes: vec!["files:read".into()],
            ttl: Duration::from_secs(ttl),
        }
    }

    fn setup() -> (JwtIssuer, TokenVerifier) {
        let issuer =
            JwtIssuer::new(IssuerConfig::new(ISS, AUD), &KeyPair::generate().unwrap()).unwrap();
        let verifier = TokenVerifier::from_jwks(&issuer.jwks(), ISS, AUD).unwrap();
        (issuer, verifier)
    }

    #[test]
    fn issue_then_verify() {
        let (issuer, verifier) = setup();
        let req = request();
        let issued = issuer.issue(&req, &intent(), &allow(120)).unwrap();

        let claims = verifier
            .verify(&issued.token, &req.tool_calls, Some("files:read"))
            .unwrap();
        assert_eq!(claims.sub, "user-42");
        assert_eq!(claims.jti, issued.jti);
        assert_eq!(claims.intent.category, IntentCategory::ReadData);
        assert_eq!(claims.sid.as_deref(), Some("sess-1"));
        assert_eq!(claims.exp - claims.iat, 120);
    }

    #[test]
    fn ttl_is_capped() {
        let (issuer, verifier) = setup();
        let req = request();
        let issued = issuer.issue(&req, &intent(), &allow(86_400)).unwrap();
        let claims = verifier
            .verify(&issued.token, &req.tool_calls, None)
            .unwrap();
        assert_eq!(claims.exp - claims.iat, DEFAULT_MAX_TTL.as_secs() as i64);
    }

    #[test]
    fn refuses_non_allow_decisions() {
        let (issuer, _) = setup();
        for d in [
            Decision::Deny {
                reason: "no".into(),
            },
            Decision::StepUp {
                reason: "approve".into(),
            },
        ] {
            assert!(issuer.issue(&request(), &intent(), &d).is_err());
        }
    }

    #[test]
    fn rejects_tampered_tool_arguments() {
        let (issuer, verifier) = setup();
        let req = request();
        let issued = issuer.issue(&req, &intent(), &allow(60)).unwrap();

        let tampered = vec![ToolCall {
            name: "read_file".into(),
            arguments: json!({"path": "/etc/shadow"}),
        }];
        assert!(matches!(
            verifier.verify(&issued.token, &tampered, None),
            Err(TokenError::RequestMismatch)
        ));
    }

    #[test]
    fn rejects_missing_scope() {
        let (issuer, verifier) = setup();
        let req = request();
        let issued = issuer.issue(&req, &intent(), &allow(60)).unwrap();
        assert!(matches!(
            verifier.verify(&issued.token, &req.tool_calls, Some("files:write")),
            Err(TokenError::MissingScope(_))
        ));
    }

    #[test]
    fn rejects_wrong_audience_and_issuer() {
        let (issuer, _) = setup();
        let req = request();
        let issued = issuer.issue(&req, &intent(), &allow(60)).unwrap();

        let wrong_aud = TokenVerifier::from_jwks(&issuer.jwks(), ISS, "other-gateway").unwrap();
        assert!(matches!(
            wrong_aud.verify(&issued.token, &req.tool_calls, None),
            Err(TokenError::Jwt(_))
        ));
        let wrong_iss = TokenVerifier::from_jwks(&issuer.jwks(), "evil", AUD).unwrap();
        assert!(matches!(
            wrong_iss.verify(&issued.token, &req.tool_calls, None),
            Err(TokenError::Jwt(_))
        ));
    }

    #[test]
    fn rejects_token_signed_by_untrusted_key() {
        let (issuer, _) = setup();
        let (_, other_verifier) = setup();
        let req = request();
        let issued = issuer.issue(&req, &intent(), &allow(60)).unwrap();
        assert!(matches!(
            other_verifier.verify(&issued.token, &req.tool_calls, None),
            Err(TokenError::UnknownKey(_))
        ));
    }

    #[test]
    fn rejects_forged_signature_with_same_kid() {
        let (issuer, verifier) = setup();
        let req = request();
        let issued = issuer.issue(&req, &intent(), &allow(60)).unwrap();

        // Re-sign the same claims with an attacker key but keep the trusted kid.
        let claims = verifier
            .verify(&issued.token, &req.tool_calls, None)
            .unwrap();
        let attacker = KeyPair::generate().unwrap();
        let mut header = Header::new(Algorithm::EdDSA);
        header.kid = Some(issuer.kid.clone());
        let forged = jsonwebtoken::encode(
            &header,
            &claims,
            &EncodingKey::from_ed_der(&attacker.pkcs8_der().unwrap()),
        )
        .unwrap();
        assert!(matches!(
            verifier.verify(&forged, &req.tool_calls, None),
            Err(TokenError::Jwt(_))
        ));
    }

    #[test]
    fn rejects_expired_token() {
        let (issuer, verifier) = setup();
        let req = request();
        let now = unix_now();
        let claims = Claims {
            iss: ISS.into(),
            sub: "user-42".into(),
            aud: AUD.into(),
            iat: now - 600,
            nbf: now - 600,
            exp: now - 300,
            jti: "j".into(),
            intent: IntentClaim {
                category: IntentCategory::ReadData,
                action: "read_file".into(),
                resource: None,
                confidence: 1.0,
            },
            scope: "files:read".into(),
            req_hash: encode_digest(&request_digest(&req).unwrap()),
            sid: None,
        };
        let token = issuer.sign(&claims).unwrap();
        assert!(matches!(
            verifier.verify(&token, &req.tool_calls, None),
            Err(TokenError::Jwt(_))
        ));
    }

    #[test]
    fn rejects_alg_none_token() {
        let (issuer, verifier) = setup();
        let req = request();
        let issued = issuer.issue(&req, &intent(), &allow(60)).unwrap();
        let payload = issued.token.split('.').nth(1).unwrap();
        // {"alg":"none","kid":<trusted kid>}
        use base64::Engine as _;
        let header = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(format!(r#"{{"alg":"none","kid":"{}"}}"#, issuer.kid));
        let unsigned = format!("{header}.{payload}.");
        assert!(verifier.verify(&unsigned, &req.tool_calls, None).is_err());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;

    const ISS: &str = "intent-server";
    const AUD: &str = "tool-gateway";

    fn fixed_issuer() -> JwtIssuer {
        JwtIssuer::new(IssuerConfig::new(ISS, AUD), &KeyPair::from_seed([7u8; 32])).unwrap()
    }

    fn fixed_verifier() -> TokenVerifier {
        TokenVerifier::from_jwks(&fixed_issuer().jwks(), ISS, AUD).unwrap()
    }

    proptest! {
        #[test]
        fn verify_never_panics_or_accepts_arbitrary_strings(token in any::<String>()) {
            let result = fixed_verifier().verify(&token, &[], None);
            prop_assert!(result.is_err());
        }

        #[test]
        fn verify_never_accepts_jwt_shaped_strings(
            token in "[A-Za-z0-9_.-]{0,120}",
            scope in prop::option::of("[a-z:]{0,12}"),
        ) {
            let result = fixed_verifier().verify(&token, &[], scope.as_deref());
            prop_assert!(result.is_err());
        }

        #[test]
        fn verify_rejects_tampered_valid_token(pos in any::<usize>(), replacement in 0usize..62) {
            const ALPHABET: &[u8] =
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
            let issuer = fixed_issuer();
            let verifier = fixed_verifier();
            let call = ToolCall {
                name: "read_file".into(),
                arguments: json!({"path": "/srv/report.txt"}),
            };
            let req = IntentRequest {
                subject: "user-42".into(),
                session_id: None,
                messages: vec![],
                tool_calls: vec![call.clone()],
                metadata: Default::default(),
            };
            let intent = Intent {
                category: IntentCategory::ReadData,
                action: "read_file".into(),
                resource: None,
                confidence: 0.95,
                source: intent_core::ClassifierSource::Rules,
            };
            let decision = Decision::Allow {
                scopes: vec!["files:read".into()],
                ttl: Duration::from_secs(60),
            };
            let issued = issuer.issue(&req, &intent, &decision).unwrap();
            prop_assert!(verifier.verify(&issued.token, &req.tool_calls, Some("files:read")).is_ok());

            // Corrupt one byte of the signed part (header or payload).
            let signed_len = issued.token.rfind('.').unwrap();
            let idx = pos % signed_len;
            let mut bytes = issued.token.clone().into_bytes();
            let original = bytes[idx];
            let mut new = ALPHABET[replacement];
            if new == original {
                new = ALPHABET[(replacement + 1) % ALPHABET.len()];
            }
            bytes[idx] = new;
            let tampered = String::from_utf8(bytes).unwrap();
            prop_assert!(verifier.verify(&tampered, &req.tool_calls, Some("files:read")).is_err());
        }
    }
}
