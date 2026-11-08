#![no_main]

use intent_core::{ClassifierSource, Intent, IntentCategory, IntentRequest, PolicyEngine};
use intent_policy::Policy;
use libfuzzer_sys::fuzz_target;

const CATEGORIES: [IntentCategory; 8] = [
    IntentCategory::ReadData,
    IntentCategory::WriteData,
    IntentCategory::SendMessage,
    IntentCategory::ExecuteCode,
    IntentCategory::AdminConfig,
    IntentCategory::Financial,
    IntentCategory::Unknown,
    IntentCategory::Malicious,
];

// Arbitrary UTF-8 as a policy file. If it parses, decide() must not panic on a
// fixed set of intents, and Malicious must never be allowed.
fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let Ok(policy) = Policy::from_toml_str(text) else {
        return;
    };
    let request = IntentRequest {
        subject: "svc-fuzz".into(),
        session_id: None,
        messages: vec![],
        tool_calls: vec![],
        metadata: Default::default(),
    };
    for category in CATEGORIES {
        for action in ["read_file", "delete_db", "*", ""] {
            for confidence in [0.0f32, 0.5, 1.0] {
                let intent = Intent {
                    category,
                    action: action.into(),
                    resource: None,
                    confidence,
                    source: ClassifierSource::Rules,
                };
                let decision = policy.decide(&request, &intent);
                if category == IntentCategory::Malicious {
                    assert!(
                        !matches!(decision, intent_core::Decision::Allow { .. }),
                        "Malicious intent was allowed"
                    );
                }
            }
        }
    }
});
