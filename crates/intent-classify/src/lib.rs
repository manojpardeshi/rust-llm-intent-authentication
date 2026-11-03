//! Intent classifiers: deterministic rules, and a hybrid that escalates
//! low-confidence results to a fallback (LLM) classifier.

mod hybrid;
#[cfg(feature = "llm")]
mod llm;
mod rules;

pub use hybrid::{DEFAULT_THRESHOLD, HybridClassifier};
#[cfg(feature = "llm")]
pub use llm::{LlmClassifier, LlmConfig};
pub use rules::{DEFAULT_INJECTION_PATTERNS, RuleClassifier, RulesConfig, ToolRule};

#[derive(Debug, thiserror::Error)]
pub enum ClassifyError {
    #[error("invalid classifier config: {0}")]
    Config(String),
}
