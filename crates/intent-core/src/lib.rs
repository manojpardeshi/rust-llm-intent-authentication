//! Core types and traits for the intent-gated token framework.
//!
//! This crate performs no I/O. It defines the request/intent/decision model,
//! the pluggable classifier/policy/issuer traits, and the canonical request
//! digest that binds an issued token to the exact tool calls it authorizes.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Errors shared across the framework.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("canonicalization failed: {0}")]
    Canonicalize(String),
    #[error("classification failed: {0}")]
    Classify(String),
    #[error("policy error: {0}")]
    Policy(String),
    #[error("token error: {0}")]
    Token(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// One chat message that accompanied the LLM request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// A tool invocation the LLM wants to perform.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// The request whose intent is evaluated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntentRequest {
    /// The principal on whose behalf the LLM acts (user or service id).
    pub subject: String,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

/// Coarse intent categories, ordered from least to most restrictive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntentCategory {
    ReadData,
    WriteData,
    SendMessage,
    ExecuteCode,
    AdminConfig,
    Financial,
    Unknown,
    Malicious,
}

impl IntentCategory {
    /// Higher means more sensitive. Used to resolve disagreements between
    /// classifiers by picking the stricter result.
    pub fn restrictiveness(self) -> u8 {
        match self {
            Self::ReadData => 0,
            Self::WriteData => 1,
            Self::SendMessage => 2,
            Self::ExecuteCode => 3,
            Self::AdminConfig => 4,
            Self::Financial => 5,
            Self::Unknown => 6,
            Self::Malicious => 7,
        }
    }

    /// Returns whichever of `self` and `other` is more restrictive.
    pub fn stricter(self, other: Self) -> Self {
        if other.restrictiveness() > self.restrictiveness() {
            other
        } else {
            self
        }
    }
}

/// Which classifier produced an intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassifierSource {
    Rules,
    Llm,
    Hybrid,
}

/// The classified intent of a request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Intent {
    pub category: IntentCategory,
    /// Fine-grained action, e.g. `"read_file"`.
    pub action: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource: Option<String>,
    /// Classifier confidence in `[0.0, 1.0]`.
    pub confidence: f32,
    pub source: ClassifierSource,
}

/// Outcome of policy evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum Decision {
    Allow {
        scopes: Vec<String>,
        #[serde(with = "duration_secs")]
        ttl: Duration,
    },
    Deny {
        reason: String,
    },
    /// Requires out-of-band human approval before a token can be issued.
    StepUp {
        reason: String,
    },
}

/// A freshly issued token and its metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IssuedToken {
    pub token: String,
    pub jti: String,
    /// Unix timestamp (seconds).
    pub expires_at: i64,
}

#[async_trait]
pub trait IntentClassifier: Send + Sync {
    async fn classify(&self, request: &IntentRequest) -> Result<Intent>;
}

pub trait PolicyEngine: Send + Sync {
    fn decide(&self, request: &IntentRequest, intent: &Intent) -> Decision;
}

pub trait TokenIssuer: Send + Sync {
    /// Issues a token. Implementations must refuse anything but `Decision::Allow`.
    fn issue(
        &self,
        request: &IntentRequest,
        intent: &Intent,
        decision: &Decision,
    ) -> Result<IssuedToken>;
}

/// Result of running a request through the full pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evaluation {
    pub intent: Intent,
    #[serde(flatten)]
    pub decision: Decision,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<IssuedToken>,
}

/// Classify → decide → issue. Tokens are only issued on `Decision::Allow`.
#[derive(Clone)]
pub struct Evaluator {
    classifier: Arc<dyn IntentClassifier>,
    policy: Arc<dyn PolicyEngine>,
    issuer: Arc<dyn TokenIssuer>,
}

impl Evaluator {
    pub fn new(
        classifier: Arc<dyn IntentClassifier>,
        policy: Arc<dyn PolicyEngine>,
        issuer: Arc<dyn TokenIssuer>,
    ) -> Self {
        Self {
            classifier,
            policy,
            issuer,
        }
    }

    pub async fn evaluate(&self, request: &IntentRequest) -> Result<Evaluation> {
        let intent = self.classifier.classify(request).await?;
        let decision = self.policy.decide(request, &intent);
        let token = match decision {
            Decision::Allow { .. } => Some(self.issuer.issue(request, &intent, &decision)?),
            _ => None,
        };
        Ok(Evaluation {
            intent,
            decision,
            token,
        })
    }
}

/// SHA-256 over the RFC 8785 (JCS) canonical JSON of the tool calls.
///
/// Key order and whitespace do not affect the digest, so a verifier can
/// recompute it from the tool calls it actually receives.
pub fn tool_calls_digest(tool_calls: &[ToolCall]) -> Result<[u8; 32]> {
    let canonical =
        serde_jcs::to_vec(tool_calls).map_err(|e| Error::Canonicalize(e.to_string()))?;
    Ok(Sha256::digest(&canonical).into())
}

/// Digest binding for an [`IntentRequest`]; covers its tool calls only, since
/// those are what downstream gateways execute and can re-check.
pub fn request_digest(request: &IntentRequest) -> Result<[u8; 32]> {
    tool_calls_digest(&request.tool_calls)
}

/// Base64url (unpadded) encoding of a digest, as carried in the `req_hash` claim.
pub fn encode_digest(digest: &[u8; 32]) -> String {
    URL_SAFE_NO_PAD.encode(digest)
}

/// `*` matches everything, `prefix*` matches by prefix, anything else exactly.
pub fn glob_match(pattern: &str, value: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => value.starts_with(prefix),
        None => pattern == value,
    }
}

mod duration_secs {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(d.as_secs())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        u64::deserialize(d).map(Duration::from_secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(args: serde_json::Value) -> ToolCall {
        ToolCall {
            name: "read_file".into(),
            arguments: args,
        }
    }

    #[test]
    fn digest_ignores_key_order_and_whitespace() {
        let a: serde_json::Value = serde_json::from_str(r#"{"path":"/tmp/a","limit":10}"#).unwrap();
        let b: serde_json::Value =
            serde_json::from_str(r#"{ "limit" : 10,  "path" : "/tmp/a" }"#).unwrap();
        assert_eq!(
            tool_calls_digest(&[call(a)]).unwrap(),
            tool_calls_digest(&[call(b)]).unwrap()
        );
    }

    #[test]
    fn digest_changes_when_arguments_change() {
        let a = tool_calls_digest(&[call(json!({"path": "/tmp/a"}))]).unwrap();
        let b = tool_calls_digest(&[call(json!({"path": "/etc/passwd"}))]).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn digest_changes_when_call_order_changes() {
        let x = call(json!({"path": "a"}));
        let y = call(json!({"path": "b"}));
        assert_ne!(
            tool_calls_digest(&[x.clone(), y.clone()]).unwrap(),
            tool_calls_digest(&[y, x]).unwrap()
        );
    }

    struct FixedClassifier(IntentCategory);

    #[async_trait]
    impl IntentClassifier for FixedClassifier {
        async fn classify(&self, _: &IntentRequest) -> Result<Intent> {
            Ok(Intent {
                category: self.0,
                action: "a".into(),
                resource: None,
                confidence: 1.0,
                source: ClassifierSource::Rules,
            })
        }
    }

    struct AllowReads;

    impl PolicyEngine for AllowReads {
        fn decide(&self, _: &IntentRequest, intent: &Intent) -> Decision {
            match intent.category {
                IntentCategory::ReadData => Decision::Allow {
                    scopes: vec!["r".into()],
                    ttl: Duration::from_secs(1),
                },
                _ => Decision::Deny {
                    reason: "no".into(),
                },
            }
        }
    }

    struct StubIssuer;

    impl TokenIssuer for StubIssuer {
        fn issue(&self, _: &IntentRequest, _: &Intent, _: &Decision) -> Result<IssuedToken> {
            Ok(IssuedToken {
                token: "t".into(),
                jti: "j".into(),
                expires_at: 0,
            })
        }
    }

    fn evaluator(category: IntentCategory) -> Evaluator {
        Evaluator::new(
            Arc::new(FixedClassifier(category)),
            Arc::new(AllowReads),
            Arc::new(StubIssuer),
        )
    }

    fn empty_request() -> IntentRequest {
        IntentRequest {
            subject: "u".into(),
            session_id: None,
            messages: vec![],
            tool_calls: vec![],
            metadata: Default::default(),
        }
    }

    #[tokio::test]
    async fn evaluator_issues_only_on_allow() {
        let allowed = evaluator(IntentCategory::ReadData)
            .evaluate(&empty_request())
            .await
            .unwrap();
        assert!(allowed.token.is_some());

        let denied = evaluator(IntentCategory::WriteData)
            .evaluate(&empty_request())
            .await
            .unwrap();
        assert!(denied.token.is_none());
        let v = serde_json::to_value(&denied).unwrap();
        assert_eq!(v["decision"], "deny");
        assert!(v.get("token").is_none());
    }

    #[test]
    fn glob() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("delete_*", "delete_db"));
        assert!(!glob_match("delete_*", "read_db"));
        assert!(glob_match("read_file", "read_file"));
        assert!(!glob_match("read_file", "read_files"));
    }

    #[test]
    fn stricter_picks_more_restrictive() {
        use IntentCategory::*;
        assert_eq!(ReadData.stricter(Financial), Financial);
        assert_eq!(Malicious.stricter(ReadData), Malicious);
        assert_eq!(Unknown.stricter(AdminConfig), Unknown);
    }

    #[test]
    fn decision_serializes_with_tag_and_ttl_seconds() {
        let d = Decision::Allow {
            scopes: vec!["files:read".into()],
            ttl: Duration::from_secs(120),
        };
        let v = serde_json::to_value(&d).unwrap();
        assert_eq!(
            v,
            json!({"decision": "allow", "scopes": ["files:read"], "ttl": 120})
        );
        assert_eq!(serde_json::from_value::<Decision>(v).unwrap(), d);
    }
}
