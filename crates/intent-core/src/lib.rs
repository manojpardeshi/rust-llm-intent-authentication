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
///
/// Serialised in `snake_case` (`read_data`, `send_message`, ...), like every
/// other enum on the wire. The older PascalCase spellings (`ReadData`) are
/// still accepted on input as aliases, so existing policy and rules files
/// keep working; output always uses `snake_case`.
///
/// Non-exhaustive so categories can be added without a breaking change;
/// code outside this crate must handle unknown variants (treat them as
/// at least as restrictive as `Unknown`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum IntentCategory {
    #[serde(alias = "ReadData")]
    ReadData,
    #[serde(alias = "WriteData")]
    WriteData,
    #[serde(alias = "SendMessage")]
    SendMessage,
    #[serde(alias = "ExecuteCode")]
    ExecuteCode,
    #[serde(alias = "AdminConfig")]
    AdminConfig,
    #[serde(alias = "Financial")]
    Financial,
    #[serde(alias = "Unknown")]
    Unknown,
    #[serde(alias = "Malicious")]
    Malicious,
}

impl std::fmt::Display for IntentCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl IntentCategory {
    /// Every category, least to most restrictive.
    pub const ALL: [Self; 8] = [
        Self::ReadData,
        Self::WriteData,
        Self::SendMessage,
        Self::ExecuteCode,
        Self::AdminConfig,
        Self::Financial,
        Self::Unknown,
        Self::Malicious,
    ];

    /// The wire name, e.g. `"read_data"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReadData => "read_data",
            Self::WriteData => "write_data",
            Self::SendMessage => "send_message",
            Self::ExecuteCode => "execute_code",
            Self::AdminConfig => "admin_config",
            Self::Financial => "financial",
            Self::Unknown => "unknown",
            Self::Malicious => "malicious",
        }
    }

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

    /// Runs the pipeline. A request with several tool calls is classified and
    /// decided once per call (each call's sub-request keeps the full message
    /// history, so injection checks still see everything), and the results
    /// are combined by [`combine_decisions`]. Policy therefore sees every
    /// call's action, and a batch can never be less restricted than its
    /// strictest call.
    pub async fn evaluate(&self, request: &IntentRequest) -> Result<Evaluation> {
        let (intent, decision) = if request.tool_calls.len() <= 1 {
            let intent = self.classifier.classify(request).await?;
            let decision = self.policy.decide(request, &intent);
            (intent, decision)
        } else {
            let mut results = Vec::with_capacity(request.tool_calls.len());
            for call in &request.tool_calls {
                let sub = IntentRequest {
                    tool_calls: vec![call.clone()],
                    ..request.clone()
                };
                let intent = self.classifier.classify(&sub).await?;
                let decision = self.policy.decide(&sub, &intent);
                results.push((call.name.as_str(), intent, decision));
            }
            combine_per_call(results)
        };
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

/// Combines per-call decisions: any deny wins, else any step-up, else allow
/// with the sorted union of scopes and the shortest TTL. Deny and step-up
/// reasons from several calls are joined, each prefixed with its tool name.
pub fn combine_decisions<'a>(
    decisions: impl IntoIterator<Item = (&'a str, &'a Decision)>,
) -> Decision {
    let mut denies = Vec::new();
    let mut step_ups = Vec::new();
    let mut scopes = std::collections::BTreeSet::new();
    let mut ttl: Option<Duration> = None;
    let mut any = false;
    for (name, decision) in decisions {
        any = true;
        match decision {
            Decision::Deny { reason } => denies.push(format!("{name}: {reason}")),
            Decision::StepUp { reason } => step_ups.push(format!("{name}: {reason}")),
            Decision::Allow { scopes: s, ttl: t } => {
                scopes.extend(s.iter().cloned());
                ttl = Some(ttl.map_or(*t, |cur| cur.min(*t)));
            }
        }
    }
    if !any {
        return Decision::Deny {
            reason: "no decisions to combine".into(),
        };
    }
    if !denies.is_empty() {
        Decision::Deny {
            reason: denies.join("; "),
        }
    } else if !step_ups.is_empty() {
        Decision::StepUp {
            reason: step_ups.join("; "),
        }
    } else {
        Decision::Allow {
            scopes: scopes.into_iter().collect(),
            ttl: ttl.unwrap_or_default(),
        }
    }
}

/// Picks the reported intent for a batch (strictest category, first such
/// call, lowest confidence across calls) and combines the decisions.
fn combine_per_call(results: Vec<(&str, Intent, Decision)>) -> (Intent, Decision) {
    let decision = combine_decisions(results.iter().map(|(n, _, d)| (*n, d)));
    let min_confidence =
        results
            .iter()
            .map(|(_, i, _)| i.confidence)
            .fold(
                f32::INFINITY,
                |a, b| if b.is_nan() || b < a { b } else { a },
            );
    let mut intent = results
        .iter()
        .map(|(_, i, _)| i)
        .fold(None::<&Intent>, |best, i| match best {
            Some(b) if b.category.restrictiveness() >= i.category.restrictiveness() => Some(b),
            _ => Some(i),
        })
        .expect("at least two calls")
        .clone();
    intent.confidence = min_confidence;
    (intent, decision)
}

/// SHA-256 over the RFC 8785 (JCS) canonical JSON of the tool calls.
///
/// Key order and whitespace do not affect the digest, so a verifier can
/// recompute it from the tool calls it actually receives.
///
/// RFC 8785 serialises numbers as IEEE-754 doubles, so integers beyond
/// ±(2^53 − 1) would lose precision and two different ids could share a
/// digest. Such integers are rejected with [`Error::Canonicalize`] here, so
/// both issuing and verifying fail closed. Pass large ids as strings.
/// Non-integer floats are allowed (their JCS form is exact for the double
/// that serde_json parsed).
pub fn tool_calls_digest(tool_calls: &[ToolCall]) -> Result<[u8; 32]> {
    for call in tool_calls {
        check_safe_integers(&call.arguments)?;
    }
    let canonical =
        serde_jcs::to_vec(tool_calls).map_err(|e| Error::Canonicalize(e.to_string()))?;
    Ok(Sha256::digest(&canonical).into())
}

/// Largest integer magnitude a double represents exactly (2^53 − 1).
pub const MAX_SAFE_INTEGER: u64 = (1 << 53) - 1;

fn check_safe_integers(v: &serde_json::Value) -> Result<()> {
    match v {
        serde_json::Value::Number(n) => {
            let unsafe_int = match (n.as_u64(), n.as_i64()) {
                (Some(u), _) => u > MAX_SAFE_INTEGER,
                (None, Some(i)) => i.unsigned_abs() > MAX_SAFE_INTEGER,
                (None, None) => false, // a float
            };
            if unsafe_int {
                return Err(Error::Canonicalize(format!(
                    "integer {n} is outside ±(2^53 - 1); pass it as a string"
                )));
            }
            Ok(())
        }
        serde_json::Value::Array(a) => a.iter().try_for_each(check_safe_integers),
        serde_json::Value::Object(o) => o.values().try_for_each(check_safe_integers),
        _ => Ok(()),
    }
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
    fn digest_rejects_integers_beyond_double_precision() {
        // These two would canonicalise to the same double and collide.
        let a: serde_json::Value = serde_json::from_str(r#"{"id":9007199254740993}"#).unwrap();
        let b: serde_json::Value = serde_json::from_str(r#"{"id":9007199254740992}"#).unwrap();
        assert!(matches!(
            tool_calls_digest(&[call(a)]),
            Err(Error::Canonicalize(_))
        ));
        assert!(matches!(
            tool_calls_digest(&[call(b)]),
            Err(Error::Canonicalize(_))
        ));
        let nested = json!({"x": [{"y": -9007199254740992i64}]});
        assert!(tool_calls_digest(&[call(nested)]).is_err());
        assert!(tool_calls_digest(&[call(json!({"id": u64::MAX}))]).is_err());
        // The boundary itself and large ids as strings are fine.
        assert!(tool_calls_digest(&[call(json!({"id": MAX_SAFE_INTEGER}))]).is_ok());
        assert!(tool_calls_digest(&[call(json!({"id": -(MAX_SAFE_INTEGER as i64)}))]).is_ok());
        assert!(tool_calls_digest(&[call(json!({"id": "9007199254740993"}))]).is_ok());
        assert!(tool_calls_digest(&[call(json!({"x": 1.5e300}))]).is_ok());
    }

    #[test]
    fn category_wire_format_is_snake_case_with_legacy_aliases() {
        for c in IntentCategory::ALL {
            let json = serde_json::to_value(c).unwrap();
            assert_eq!(json, json!(c.as_str()));
            assert_eq!(c.to_string(), c.as_str());
            let back: IntentCategory = serde_json::from_value(json).unwrap();
            assert_eq!(back, c);
            // The old PascalCase spelling (the Debug name) still parses.
            let legacy: IntentCategory = serde_json::from_value(json!(format!("{c:?}"))).unwrap();
            assert_eq!(legacy, c);
        }
        assert!(serde_json::from_value::<IntentCategory>(json!("readdata")).is_err());
        assert_eq!(
            IntentCategory::ALL.map(|c| c.restrictiveness()),
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn combine_decisions_strictest_wins() {
        let allow = |s: &str, t| Decision::Allow {
            scopes: vec![s.into()],
            ttl: Duration::from_secs(t),
        };
        let deny = Decision::Deny {
            reason: "no".into(),
        };
        let step = Decision::StepUp {
            reason: "ask".into(),
        };
        let a1 = allow("b", 120);
        let a2 = allow("a", 60);
        assert_eq!(
            combine_decisions([("x", &a1), ("y", &a2)]),
            Decision::Allow {
                scopes: vec!["a".into(), "b".into()],
                ttl: Duration::from_secs(60)
            }
        );
        assert_eq!(
            combine_decisions([("x", &a1), ("y", &step)]),
            Decision::StepUp {
                reason: "y: ask".into()
            }
        );
        assert_eq!(
            combine_decisions([("x", &step), ("y", &deny), ("z", &a1)]),
            Decision::Deny {
                reason: "y: no".into()
            }
        );
        assert!(matches!(
            combine_decisions(std::iter::empty()),
            Decision::Deny { .. }
        ));
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

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::Value;

    /// Arbitrary JSON values, including nested objects and arrays.
    fn json_value() -> impl Strategy<Value = Value> {
        let leaf = prop_oneof![
            Just(Value::Null),
            any::<bool>().prop_map(Value::Bool),
            (-(MAX_SAFE_INTEGER as i64)..=MAX_SAFE_INTEGER as i64).prop_map(Value::from),
            any::<f64>()
                .prop_filter("finite", |f| f.is_finite())
                .prop_map(Value::from),
            "[ -~]{0,8}".prop_map(Value::String),
            any::<String>().prop_map(Value::String),
        ];
        leaf.prop_recursive(3, 24, 4, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..4).prop_map(Value::Array),
                prop::collection::btree_map("[a-z]{1,3}", inner, 0..4)
                    .prop_map(|m| Value::Object(m.into_iter().collect())),
            ]
        })
    }

    fn tool_calls() -> impl Strategy<Value = Vec<ToolCall>> {
        prop::collection::vec(
            ("[a-z_]{1,8}", json_value())
                .prop_map(|(name, arguments)| ToolCall { name, arguments }),
            0..4,
        )
    }

    /// FNV-1a over the key, perturbed by `seed`: gives a seed-dependent order.
    fn key_rank(key: &str, seed: u64) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64 ^ seed;
        for b in key.bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h
    }

    /// Renders `v` as JSON text with object members in a seed-dependent order.
    fn render(v: &Value, seed: u64) -> String {
        match v {
            Value::Object(map) => {
                let mut members: Vec<_> = map.iter().collect();
                members.sort_by_key(|(k, _)| key_rank(k, seed));
                let body: Vec<String> = members
                    .iter()
                    .map(|(k, x)| {
                        format!("{}:{}", serde_json::to_string(k).unwrap(), render(x, seed))
                    })
                    .collect();
                format!("{{{}}}", body.join(","))
            }
            Value::Array(items) => {
                let body: Vec<String> = items.iter().map(|x| render(x, seed)).collect();
                format!("[{}]", body.join(","))
            }
            other => other.to_string(),
        }
    }

    fn calls_text(calls: &[ToolCall], seed: u64) -> String {
        let items: Vec<String> = calls
            .iter()
            .map(|c| {
                format!(
                    "{{\"name\":{},\"arguments\":{}}}",
                    serde_json::to_string(&c.name).unwrap(),
                    render(&c.arguments, seed)
                )
            })
            .collect();
        format!("[{}]", items.join(","))
    }

    fn count_strings(v: &Value) -> usize {
        match v {
            Value::String(_) => 1,
            Value::Array(a) => a.iter().map(count_strings).sum(),
            Value::Object(o) => o.values().map(count_strings).sum(),
            _ => 0,
        }
    }

    /// Appends a marker to the `target`-th string value (depth-first).
    fn mutate_string(v: &mut Value, target: &mut usize) -> bool {
        match v {
            Value::String(s) => {
                if *target == 0 {
                    s.push('\u{1}');
                    true
                } else {
                    *target -= 1;
                    false
                }
            }
            Value::Array(a) => a.iter_mut().any(|x| mutate_string(x, target)),
            Value::Object(o) => o.values_mut().any(|x| mutate_string(x, target)),
            _ => false,
        }
    }

    fn decision() -> impl Strategy<Value = Decision> {
        prop_oneof![
            (prop::collection::vec("[a-c]:[rw]", 1..3), 1u64..300).prop_map(|(scopes, t)| {
                Decision::Allow {
                    scopes,
                    ttl: Duration::from_secs(t),
                }
            }),
            "[a-z]{1,5}".prop_map(|reason| Decision::Deny { reason }),
            "[a-z]{1,5}".prop_map(|reason| Decision::StepUp { reason }),
        ]
    }

    fn rank(d: &Decision) -> u8 {
        match d {
            Decision::Allow { .. } => 0,
            Decision::StepUp { .. } => 1,
            Decision::Deny { .. } => 2,
        }
    }

    proptest! {
        /// Across the full i64/u64 range, an integer is accepted exactly when
        /// a double represents it, and accepted ones never collide.
        #[test]
        fn digest_integer_range(a in any::<i64>(), b in any::<u64>()) {
            let da = tool_calls_digest(&[ToolCall { name: "t".into(), arguments: Value::from(a) }]);
            prop_assert_eq!(da.is_ok(), a.unsigned_abs() <= MAX_SAFE_INTEGER);
            let db = tool_calls_digest(&[ToolCall { name: "t".into(), arguments: Value::from(b) }]);
            prop_assert_eq!(db.is_ok(), b <= MAX_SAFE_INTEGER);
            if let (Ok(x), Ok(y)) = (da, db) {
                prop_assert_eq!(x == y, a >= 0 && a as u64 == b);
            }
        }

        /// Adding a call to a batch never makes the decision less restrictive.
        #[test]
        fn adding_a_call_never_relaxes_the_decision(
            base in prop::collection::vec(decision(), 1..5),
            extra in decision(),
        ) {
            let before = combine_decisions(base.iter().map(|d| ("t", d)));
            let after = combine_decisions(base.iter().chain([&extra]).map(|d| ("t", d)));
            prop_assert!(rank(&after) >= rank(&before));
            if let (Decision::Allow { ttl: a, scopes: sa }, Decision::Allow { ttl: b, scopes: sb }) = (&after, &before) {
                prop_assert!(a <= b);
                prop_assert!(sb.iter().all(|s| sa.contains(s)));
            }
        }

        #[test]
        fn digest_ignores_object_key_order(
            calls in tool_calls(),
            s1 in any::<u64>(),
            s2 in any::<u64>(),
        ) {
            let a: Vec<ToolCall> = serde_json::from_str(&calls_text(&calls, s1)).unwrap();
            let b: Vec<ToolCall> = serde_json::from_str(&calls_text(&calls, s2)).unwrap();
            prop_assert_eq!(tool_calls_digest(&a).unwrap(), tool_calls_digest(&b).unwrap());
            prop_assert_eq!(
                tool_calls_digest(&a).unwrap(),
                tool_calls_digest(&calls).unwrap()
            );
        }

        #[test]
        fn digest_ignores_insertion_order_of_shuffled_pairs(
            pairs in prop::collection::btree_map("[a-z]{1,4}", json_value(), 0..6)
                .prop_map(|m| m.into_iter().collect::<Vec<_>>())
                .prop_shuffle(),
            nested in json_value(),
        ) {
            let mut forward = serde_json::Map::new();
            for (k, v) in &pairs {
                forward.insert(k.clone(), v.clone());
            }
            let mut shuffled = serde_json::Map::new();
            for (k, v) in pairs.iter().rev() {
                shuffled.insert(k.clone(), v.clone());
            }
            let mut outer_a = serde_json::Map::new();
            outer_a.insert("args".into(), Value::Object(forward));
            outer_a.insert("nested".into(), nested.clone());
            let mut outer_b = serde_json::Map::new();
            outer_b.insert("nested".into(), nested);
            outer_b.insert("args".into(), Value::Object(shuffled));
            let call = |args: Value| vec![ToolCall { name: "t".into(), arguments: args }];
            prop_assert_eq!(
                tool_calls_digest(&call(Value::Object(outer_a))).unwrap(),
                tool_calls_digest(&call(Value::Object(outer_b))).unwrap()
            );
        }

        #[test]
        fn digest_changes_when_any_string_value_changes(
            calls in tool_calls(),
            pick in any::<usize>(),
        ) {
            let total: usize = calls.iter().map(|c| count_strings(&c.arguments)).sum();
            prop_assume!(total > 0);
            let mut idx = pick % total;
            let mut mutated = calls.clone();
            for call in &mut mutated {
                let n = count_strings(&call.arguments);
                if idx < n {
                    prop_assert!(mutate_string(&mut call.arguments, &mut idx));
                    break;
                }
                idx -= n;
            }
            prop_assert_ne!(
                tool_calls_digest(&calls).unwrap(),
                tool_calls_digest(&mutated).unwrap()
            );
        }

        #[test]
        fn glob_star_matches_everything(value in any::<String>()) {
            prop_assert!(glob_match("*", &value));
        }

        #[test]
        fn glob_exact_pattern_matches_itself(pattern in any::<String>()) {
            prop_assert!(glob_match(&pattern, &pattern));
        }

        #[test]
        fn glob_prefix_pattern_matches_prefix_plus_anything(
            prefix in any::<String>(),
            rest in any::<String>(),
        ) {
            let pattern = format!("{prefix}*");
            let value = format!("{prefix}{rest}");
            prop_assert!(glob_match(&pattern, &value));
        }
    }
}
