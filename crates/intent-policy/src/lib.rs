//! TOML policy engine mapping classified intents to decisions.
//!
//! Rules are evaluated in order and the first match wins. Anything unmatched
//! falls through to `[default]`, which should be `deny`. Two guards apply
//! regardless of configuration: `Malicious` intents are always denied, and a
//! matched rule's `min_confidence` must be met or the request is denied.

use std::path::Path;
use std::time::Duration;

use intent_core::{
    ClassifierSource, Decision, Intent, IntentCategory, IntentRequest, PolicyEngine, glob_match,
};
use serde::Deserialize;

#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    #[error("reading policy file: {0}")]
    Io(#[from] std::io::Error),
    #[error("parsing policy: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("invalid policy: {0}")]
    Invalid(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionKind {
    Allow,
    Deny,
    StepUp,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rule {
    /// Matches any category when omitted.
    #[serde(default)]
    pub category: Option<IntentCategory>,
    /// Action pattern: `*`, `prefix*`, or exact. Matches any when omitted.
    #[serde(default)]
    pub action: Option<String>,
    /// Subject patterns; `["*"]` when omitted.
    #[serde(default = "any")]
    pub subjects: Vec<String>,
    /// Classifier sources this rule accepts (`rules`, `llm`, `hybrid`). Any
    /// source when omitted. Use `["rules"]` on sensitive allows so that only a
    /// rule-derived intent, never one taken from the LLM's reading of
    /// attacker-influenced text, can earn the token.
    #[serde(default)]
    pub sources: Option<Vec<ClassifierSource>>,
    pub decision: DecisionKind,
    #[serde(default)]
    pub scopes: Vec<String>,
    #[serde(default = "default_ttl")]
    pub ttl_secs: u64,
    #[serde(default)]
    pub min_confidence: f32,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DefaultRule {
    pub decision: DecisionKind,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    #[serde(default, rename = "rule")]
    pub rules: Vec<Rule>,
    pub default: DefaultRule,
}

fn any() -> Vec<String> {
    vec!["*".into()]
}

fn default_ttl() -> u64 {
    120
}

impl Policy {
    pub fn from_toml_str(s: &str) -> Result<Self, PolicyError> {
        let policy: Self = toml::from_str(s)?;
        policy.validate()?;
        Ok(policy)
    }

    pub fn from_path(path: &Path) -> Result<Self, PolicyError> {
        Self::from_toml_str(&std::fs::read_to_string(path)?)
    }

    fn validate(&self) -> Result<(), PolicyError> {
        if self.default.decision == DecisionKind::Allow {
            return Err(PolicyError::Invalid(
                "[default] decision must not be allow".into(),
            ));
        }
        for (i, rule) in self.rules.iter().enumerate() {
            let n = i + 1;
            if !(0.0..=1.0).contains(&rule.min_confidence) {
                return Err(PolicyError::Invalid(format!(
                    "rule {n}: min_confidence must be within 0.0..=1.0"
                )));
            }
            if rule.decision == DecisionKind::Allow {
                if rule.scopes.is_empty() {
                    return Err(PolicyError::Invalid(format!(
                        "rule {n}: allow rules must grant at least one scope"
                    )));
                }
                if rule.ttl_secs == 0 {
                    return Err(PolicyError::Invalid(format!(
                        "rule {n}: ttl_secs must be > 0"
                    )));
                }
                if matches!(
                    rule.category,
                    None | Some(IntentCategory::Malicious | IntentCategory::Unknown)
                ) {
                    return Err(PolicyError::Invalid(format!(
                        "rule {n}: allow rules must name a category other than Malicious or Unknown"
                    )));
                }
            }
            if rule.sources.as_ref().is_some_and(Vec::is_empty) {
                return Err(PolicyError::Invalid(format!(
                    "rule {n}: sources must not be empty when present"
                )));
            }
            if rule.subjects.is_empty() {
                return Err(PolicyError::Invalid(format!(
                    "rule {n}: subjects must not be empty"
                )));
            }
        }
        Ok(())
    }

    fn matches(rule: &Rule, request: &IntentRequest, intent: &Intent) -> bool {
        rule.category.is_none_or(|c| c == intent.category)
            && rule
                .action
                .as_deref()
                .is_none_or(|p| glob_match(p, &intent.action))
            && rule
                .subjects
                .iter()
                .any(|p| glob_match(p, &request.subject))
            && rule
                .sources
                .as_ref()
                .is_none_or(|s| s.contains(&intent.source))
    }

    fn to_decision(kind: DecisionKind, rule: Option<&Rule>, reason: Option<&str>) -> Decision {
        match kind {
            DecisionKind::Allow => {
                let rule = rule.expect("allow decisions only come from rules");
                Decision::Allow {
                    scopes: rule.scopes.clone(),
                    ttl: Duration::from_secs(rule.ttl_secs),
                }
            }
            DecisionKind::Deny => Decision::Deny {
                reason: reason.unwrap_or("denied by policy").to_owned(),
            },
            DecisionKind::StepUp => Decision::StepUp {
                reason: reason.unwrap_or("human approval required").to_owned(),
            },
        }
    }
}

impl PolicyEngine for Policy {
    fn decide(&self, request: &IntentRequest, intent: &Intent) -> Decision {
        if intent.category == IntentCategory::Malicious {
            return Decision::Deny {
                reason: "request classified as malicious".into(),
            };
        }
        match self
            .rules
            .iter()
            .find(|r| Self::matches(r, request, intent))
        {
            // A NaN confidence counts as below the bar (fails closed).
            Some(rule) if intent.confidence.is_nan() || intent.confidence < rule.min_confidence => {
                Decision::Deny {
                    reason: format!(
                        "intent confidence {:.2} below required {:.2}",
                        intent.confidence, rule.min_confidence
                    ),
                }
            }
            Some(rule) => Self::to_decision(rule.decision, Some(rule), rule.reason.as_deref()),
            None => Self::to_decision(self.default.decision, None, self.default.reason.as_deref()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use intent_core::ClassifierSource;

    const POLICY: &str = r#"
        [[rule]]
        category = "read_data"
        decision = "allow"
        scopes = ["files:read"]
        ttl_secs = 60
        min_confidence = 0.7

        [[rule]]
        category = "write_data"
        action = "delete_*"
        decision = "deny"
        reason = "deletes are never automated"

        [[rule]]
        category = "write_data"
        subjects = ["svc-*"]
        decision = "allow"
        scopes = ["files:write"]

        [[rule]]
        category = "financial"
        decision = "step_up"

        [default]
        decision = "deny"
    "#;

    fn req(subject: &str) -> IntentRequest {
        IntentRequest {
            subject: subject.into(),
            session_id: None,
            messages: vec![],
            tool_calls: vec![],
            metadata: Default::default(),
        }
    }

    fn intent(category: IntentCategory, action: &str, confidence: f32) -> Intent {
        Intent {
            category,
            action: action.into(),
            resource: None,
            confidence,
            source: ClassifierSource::Rules,
        }
    }

    fn policy() -> Policy {
        Policy::from_toml_str(POLICY).unwrap()
    }

    #[test]
    fn allows_matching_rule_with_scopes_and_ttl() {
        let d = policy().decide(
            &req("alice"),
            &intent(IntentCategory::ReadData, "read_file", 0.9),
        );
        assert_eq!(
            d,
            Decision::Allow {
                scopes: vec!["files:read".into()],
                ttl: Duration::from_secs(60)
            }
        );
    }

    #[test]
    fn denies_below_min_confidence() {
        let d = policy().decide(
            &req("alice"),
            &intent(IntentCategory::ReadData, "read_file", 0.5),
        );
        assert!(matches!(d, Decision::Deny { reason } if reason.contains("confidence")));
    }

    #[test]
    fn legacy_pascal_case_categories_still_load() {
        let p = Policy::from_toml_str(
            "[[rule]]\ncategory = \"ReadData\"\ndecision = \"allow\"\nscopes = [\"r\"]\n[default]\ndecision = \"deny\"",
        )
        .unwrap();
        assert!(matches!(
            p.decide(
                &req("a"),
                &intent(IntentCategory::ReadData, "read_file", 1.0)
            ),
            Decision::Allow { .. }
        ));
    }

    #[test]
    fn sources_restrict_which_classifier_may_earn_an_allow() {
        let p = Policy::from_toml_str(
            r#"
            [[rule]]
            category = "write_data"
            decision = "allow"
            scopes = ["w"]
            sources = ["rules", "hybrid"]

            [default]
            decision = "deny"
            "#,
        )
        .unwrap();
        let with = |source| Intent {
            source,
            ..intent(IntentCategory::WriteData, "write_file", 1.0)
        };
        assert!(matches!(
            p.decide(&req("a"), &with(ClassifierSource::Rules)),
            Decision::Allow { .. }
        ));
        assert!(matches!(
            p.decide(&req("a"), &with(ClassifierSource::Hybrid)),
            Decision::Allow { .. }
        ));
        assert!(matches!(
            p.decide(&req("a"), &with(ClassifierSource::Llm)),
            Decision::Deny { .. }
        ));
        assert!(
            Policy::from_toml_str(
                "[[rule]]\ndecision = \"deny\"\nsources = []\n[default]\ndecision = \"deny\""
            )
            .is_err()
        );
    }

    #[test]
    fn nan_confidence_fails_closed() {
        let d = policy().decide(
            &req("alice"),
            &intent(IntentCategory::ReadData, "read_file", f32::NAN),
        );
        assert!(matches!(d, Decision::Deny { .. }));
    }

    #[test]
    fn first_match_wins_and_action_glob_applies() {
        let d = policy().decide(
            &req("svc-etl"),
            &intent(IntentCategory::WriteData, "delete_db", 1.0),
        );
        assert_eq!(
            d,
            Decision::Deny {
                reason: "deletes are never automated".into()
            }
        );
    }

    #[test]
    fn subject_glob_restricts_rule() {
        let p = policy();
        let write = intent(IntentCategory::WriteData, "write_file", 1.0);
        assert!(matches!(
            p.decide(&req("svc-etl"), &write),
            Decision::Allow { .. }
        ));
        assert!(matches!(
            p.decide(&req("alice"), &write),
            Decision::Deny { .. }
        ));
    }

    #[test]
    fn step_up_has_default_reason() {
        let d = policy().decide(
            &req("alice"),
            &intent(IntentCategory::Financial, "transfer_funds", 1.0),
        );
        assert_eq!(
            d,
            Decision::StepUp {
                reason: "human approval required".into()
            }
        );
    }

    #[test]
    fn unmatched_falls_to_default_deny() {
        let d = policy().decide(
            &req("alice"),
            &intent(IntentCategory::ExecuteCode, "run_shell", 1.0),
        );
        assert!(matches!(d, Decision::Deny { .. }));
    }

    #[test]
    fn malicious_is_always_denied() {
        let p = Policy::from_toml_str(
            r#"
            [[rule]]
            decision = "step_up"
            [default]
            decision = "step_up"
            "#,
        )
        .unwrap();
        let d = p.decide(&req("alice"), &intent(IntentCategory::Malicious, "x", 1.0));
        assert!(matches!(d, Decision::Deny { .. }));
    }

    #[test]
    fn rejects_unsafe_policies() {
        let cases = [
            // default allow
            "[default]\ndecision = \"allow\"",
            // allow without scopes
            "[[rule]]\ncategory = \"read_data\"\ndecision = \"allow\"\n[default]\ndecision = \"deny\"",
            // allow for any category
            "[[rule]]\ndecision = \"allow\"\nscopes = [\"x\"]\n[default]\ndecision = \"deny\"",
            // allow Unknown
            "[[rule]]\ncategory = \"unknown\"\ndecision = \"allow\"\nscopes = [\"x\"]\n[default]\ndecision = \"deny\"",
            // bad confidence
            "[[rule]]\ndecision = \"deny\"\nmin_confidence = 1.5\n[default]\ndecision = \"deny\"",
            // typo'd field
            "[[rule]]\ndecison = \"deny\"\n[default]\ndecision = \"deny\"",
        ];
        for c in cases {
            assert!(Policy::from_toml_str(c).is_err(), "should reject:\n{c}");
        }
    }
}

#[cfg(test)]
mod proptests {
    // Not `super::*`: the module's private `fn any()` would shadow proptest's `any`.
    use super::{Decision, Policy, PolicyEngine};
    use intent_core::{ClassifierSource, Intent, IntentCategory, IntentRequest};
    use proptest::prelude::*;

    /// A valid policy with allow rules for several categories.
    const VALID: &str = r#"
        [[rule]]
        category = "read_data"
        decision = "allow"
        scopes = ["files:read"]
        min_confidence = 0.5

        [[rule]]
        category = "write_data"
        action = "delete_*"
        decision = "deny"

        [[rule]]
        category = "write_data"
        subjects = ["svc-*"]
        decision = "allow"
        scopes = ["files:write"]

        [[rule]]
        action = "*"
        decision = "step_up"

        [default]
        decision = "deny"
    "#;

    /// Fragments that look like policy TOML, so random docs reach deeper parsing.
    const FRAGMENTS: &[&str] = &[
        "[[rule]]\n",
        "[default]\n",
        "decision = \"allow\"\n",
        "decision = \"deny\"\n",
        "decision = \"step_up\"\n",
        "category = \"read_data\"\n",
        "category = \"malicious\"\n",
        "scopes = [\"x\"]\n",
        "min_confidence = 1.5\n",
        "min_confidence = 0.5\n",
        "ttl_secs = 0\n",
        "subjects = []\n",
        "subjects = [\"svc-*\"]\n",
        "action = \"*\"\n",
        "reason = \"r\"\n",
        "decison = \"deny\"\n",
        "junk ===\n",
        "\"\n",
        "[\n",
    ];

    fn category() -> impl Strategy<Value = IntentCategory> {
        prop::sample::select(vec![
            IntentCategory::ReadData,
            IntentCategory::WriteData,
            IntentCategory::SendMessage,
            IntentCategory::ExecuteCode,
            IntentCategory::AdminConfig,
            IntentCategory::Financial,
            IntentCategory::Unknown,
            IntentCategory::Malicious,
        ])
    }

    proptest! {
        #[test]
        fn parsing_arbitrary_text_never_panics(s in any::<String>()) {
            let _ = Policy::from_toml_str(&s);
        }

        #[test]
        fn parsing_policy_like_text_never_panics(
            parts in prop::collection::vec(prop::sample::select(FRAGMENTS), 0..16),
        ) {
            let _ = Policy::from_toml_str(&parts.concat());
        }

        #[test]
        fn valid_policy_never_allows_malicious_or_unknown(
            cat in category(),
            action in ".*",
            subject in ".*",
            confidence in any::<f32>(),
        ) {
            let policy = Policy::from_toml_str(VALID).unwrap();
            let request = IntentRequest {
                subject,
                session_id: None,
                messages: vec![],
                tool_calls: vec![],
                metadata: Default::default(),
            };
            let intent = Intent {
                category: cat,
                action,
                resource: None,
                confidence,
                source: ClassifierSource::Llm,
            };
            let decision = policy.decide(&request, &intent);
            if matches!(cat, IntentCategory::Malicious | IntentCategory::Unknown) {
                prop_assert!(!matches!(decision, Decision::Allow { .. }), "got {decision:?}");
            }
        }
    }
}
