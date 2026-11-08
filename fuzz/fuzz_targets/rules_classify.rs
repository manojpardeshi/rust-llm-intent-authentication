#![no_main]

use std::sync::OnceLock;

use arbitrary::Arbitrary;
use intent_classify::RuleClassifier;
use intent_core::{IntentRequest, Message, ToolCall};
use libfuzzer_sys::fuzz_target;
use serde_json::Value;

const EXAMPLE_RULES: &str = include_str!("../../config/rules.example.toml");

/// Structured fuzz input: an arbitrary request shape. Tool arguments are
/// parsed as JSON when valid, otherwise wrapped as a string, so both nested
/// structures and odd strings reach the classifier.
#[derive(Debug, Arbitrary)]
struct Input {
    subject: String,
    messages: Vec<(String, String)>,
    tool_calls: Vec<(String, Vec<u8>)>,
}

impl Input {
    fn into_request(self) -> IntentRequest {
        IntentRequest {
            subject: self.subject,
            session_id: None,
            messages: self
                .messages
                .into_iter()
                .map(|(role, content)| Message { role, content })
                .collect(),
            tool_calls: self
                .tool_calls
                .into_iter()
                .map(|(name, args)| ToolCall {
                    name,
                    arguments: serde_json::from_slice(&args)
                        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&args).into())),
                })
                .collect(),
            metadata: Default::default(),
        }
    }
}

fuzz_target!(|input: Input| {
    static CLASSIFIER: OnceLock<RuleClassifier> = OnceLock::new();
    let classifier = CLASSIFIER
        .get_or_init(|| RuleClassifier::from_toml_str(EXAMPLE_RULES).expect("example rules are valid"));

    let intent = classifier.classify_sync(&input.into_request());
    assert!(
        (0.0..=1.0).contains(&intent.confidence),
        "confidence out of range: {}",
        intent.confidence
    );
});
