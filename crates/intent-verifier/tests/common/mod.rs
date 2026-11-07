#![allow(dead_code)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use intent_core::{
    ClassifierSource, Decision, Intent, IntentCategory, IntentRequest, TokenIssuer, ToolCall,
};
use intent_token::{IssuerConfig, Jwks, JwtIssuer, KeyPair};
use serde_json::json;
use wiremock::{Request, Respond, ResponseTemplate};

pub const ISS: &str = "intent-server";
pub const AUD: &str = "tool-gateway";

pub fn calls() -> Vec<ToolCall> {
    vec![ToolCall {
        name: "read_file".into(),
        arguments: json!({"path": "/srv/report.txt"}),
    }]
}

pub fn tampered() -> Vec<ToolCall> {
    vec![ToolCall {
        name: "read_file".into(),
        arguments: json!({"path": "/etc/shadow"}),
    }]
}

pub fn issuer(key: &KeyPair) -> JwtIssuer {
    JwtIssuer::new(IssuerConfig::new(ISS, AUD), key).unwrap()
}

/// Mints a token for `calls()` granting `scopes`.
pub fn mint(key: &KeyPair, scopes: &[&str]) -> String {
    let request = IntentRequest {
        subject: "user-42".into(),
        session_id: Some("sess-1".into()),
        messages: vec![],
        tool_calls: calls(),
        metadata: Default::default(),
    };
    let intent = Intent {
        category: IntentCategory::ReadData,
        action: "read_file".into(),
        resource: Some("/srv/report.txt".into()),
        confidence: 0.95,
        source: ClassifierSource::Rules,
    };
    let decision = Decision::Allow {
        scopes: scopes.iter().map(|s| s.to_string()).collect(),
        ttl: Duration::from_secs(120),
    };
    issuer(key)
        .issue(&request, &intent, &decision)
        .unwrap()
        .token
}

pub fn jwks(key: &KeyPair) -> Jwks {
    issuer(key).jwks()
}

/// What the mock JWKS endpoint currently serves.
#[derive(Clone)]
pub enum Mode {
    Keys(Jwks),
    Fail,
    Raw(String),
}

#[derive(Clone)]
pub struct Dynamic(pub Arc<Mutex<Mode>>);

impl Dynamic {
    pub fn new(mode: Mode) -> Self {
        Self(Arc::new(Mutex::new(mode)))
    }
    pub fn set(&self, mode: Mode) {
        *self.0.lock().unwrap() = mode;
    }
}

impl Respond for Dynamic {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        match &*self.0.lock().unwrap() {
            Mode::Keys(j) => ResponseTemplate::new(200).set_body_json(j),
            Mode::Fail => ResponseTemplate::new(500),
            Mode::Raw(s) => ResponseTemplate::new(200).set_body_string(s.clone()),
        }
    }
}
