//! Server configuration (TOML) and API key handling.

use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq as _;

use crate::Error;

/// Environment variable holding comma-separated API keys.
pub const API_KEYS_ENV: &str = "INTENT_API_KEYS";

/// On-disk server configuration. API keys are deliberately not part of it.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default = "default_bind")]
    pub bind: SocketAddr,
    pub key_path: PathBuf,
    pub policy_path: PathBuf,
    pub rules_path: PathBuf,
    #[serde(default = "default_issuer")]
    pub issuer: String,
    #[serde(default = "default_audience")]
    pub audience: String,
    /// Rule confidence at/above which the LLM fallback is skipped.
    #[serde(default)]
    pub threshold: Option<f32>,
    /// Enable the LLM fallback classifier.
    #[serde(default)]
    pub llm: bool,
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
    /// Allow starting without any API key. Dev only.
    #[serde(default)]
    pub allow_unauthenticated: bool,
    /// Public JWKS files (`{"keys":[...]}`) of rotated-out signing keys. Their
    /// keys stay valid for `/v1/verify` and are published in the JWKS, but
    /// are never used for signing. Reloadable.
    #[serde(default)]
    pub retired_keys_paths: Vec<PathBuf>,
    /// Watch `policy_path` and `rules_path` and reload on change.
    #[serde(default)]
    pub watch: bool,
    /// Per-client rate limit for /v1/evaluate and /v1/verify. Absent = off.
    #[serde(default)]
    pub rate_limit: Option<RateLimitConfig>,
    /// Subject patterns each API key may request, keyed by the hex SHA-256
    /// of the key (so no secret lives in this file). A key listed here gets
    /// 403 for any other `subject`.
    #[serde(default)]
    pub subject_bindings: BTreeMap<String, Vec<String>>,
    /// Reject keys that have no entry in `subject_bindings` (403). Without
    /// it, unbound keys may still request any subject.
    #[serde(default)]
    pub require_subject_binding: bool,
    /// Make `/v1/verify` single-use: the first successful verification of a
    /// token consumes its `jti`, later ones get 401 `replayed`. In-memory and
    /// per process. Off by default (the endpoint is stateless).
    #[serde(default)]
    pub verify_single_use: bool,
    /// Maximum unexpired token ids remembered when `verify_single_use` is on.
    /// When full, verification fails closed with 503.
    #[serde(default = "default_replay_capacity")]
    pub verify_replay_capacity: usize,
    /// Require an API key for `/metrics`. Off by default: metrics are meant
    /// for a private network or a scraper on the same host.
    #[serde(default)]
    pub metrics_require_auth: bool,
}

fn default_replay_capacity() -> usize {
    100_000
}

/// Token-bucket rate limit: `burst` requests at once, refilled at
/// `per_second`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RateLimitConfig {
    pub per_second: u32,
    pub burst: u32,
}

impl RateLimitConfig {
    pub fn validate(&self) -> Result<(), Error> {
        if self.per_second == 0 || self.burst == 0 {
            return Err(Error::Config(
                "rate_limit.per_second and rate_limit.burst must be greater than 0".into(),
            ));
        }
        Ok(())
    }
}

fn default_bind() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 8080))
}

fn default_issuer() -> String {
    "intent-server".into()
}

fn default_audience() -> String {
    "tool-gateway".into()
}

fn default_max_body_bytes() -> usize {
    256 * 1024
}

impl ServerConfig {
    pub fn from_toml_str(s: &str) -> Result<Self, Error> {
        let config: Self = toml::from_str(s).map_err(|e| Error::Config(e.to_string()))?;
        if let Some(rl) = &config.rate_limit {
            rl.validate()?;
        }
        Ok(config)
    }

    pub fn from_path(path: &Path) -> Result<Self, Error> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("reading {}: {e}", path.display())))?;
        Self::from_toml_str(&text)
    }
}

/// SHA-256 digests of the accepted API keys, plus optional per-key subject
/// bindings.
#[derive(Debug, Clone, Default)]
pub struct ApiKeys {
    digests: Vec<[u8; 32]>,
    bindings: HashMap<[u8; 32], Vec<String>>,
    require_binding: bool,
}

impl ApiKeys {
    /// Parses a comma-separated key list; blank entries are ignored.
    pub fn from_csv(csv: &str) -> Self {
        Self::from_keys(csv.split(',').map(str::trim).filter(|k| !k.is_empty()))
    }

    pub fn from_keys<'a>(keys: impl IntoIterator<Item = &'a str>) -> Self {
        Self {
            digests: keys.into_iter().map(digest).collect(),
            ..Self::default()
        }
    }

    /// Binds keys (by hex SHA-256 digest) to the subject patterns they may
    /// request. Every digest must belong to a configured key, so a typo
    /// fails loudly instead of leaving a key unbound.
    pub fn with_subject_bindings(
        mut self,
        bindings: &BTreeMap<String, Vec<String>>,
        require_binding: bool,
    ) -> Result<Self, Error> {
        for (hex, patterns) in bindings {
            let d = parse_hex_digest(hex).ok_or_else(|| {
                Error::Config(format!(
                    "subject_bindings: {hex:?} is not a 64-character hex SHA-256 digest"
                ))
            })?;
            if !self.digests.contains(&d) {
                return Err(Error::Config(format!(
                    "subject_bindings: {hex} does not match any key in {API_KEYS_ENV}"
                )));
            }
            if patterns.is_empty() {
                return Err(Error::Config(format!(
                    "subject_bindings: {hex} must list at least one subject pattern"
                )));
            }
            self.bindings.insert(d, patterns.clone());
        }
        self.require_binding = require_binding;
        Ok(self)
    }

    /// Reads [`API_KEYS_ENV`]; unset means no keys.
    pub fn from_env() -> Self {
        std::env::var(API_KEYS_ENV)
            .map(|v| Self::from_csv(&v))
            .unwrap_or_default()
    }

    pub fn is_empty(&self) -> bool {
        self.digests.is_empty()
    }

    /// Constant-time check of a presented key against every stored digest.
    pub fn check(&self, presented: &str) -> bool {
        self.authenticate(presented).is_some()
    }

    /// Like [`check`](Self::check), returning the key's digest on success.
    pub fn authenticate(&self, presented: &str) -> Option<[u8; 32]> {
        let d = digest(presented);
        let mut ok = subtle::Choice::from(0);
        for stored in &self.digests {
            ok |= stored.ct_eq(&d);
        }
        bool::from(ok).then_some(d)
    }

    /// Whether the key with this digest may request `subject`.
    pub fn subject_allowed(&self, key: &[u8; 32], subject: &str) -> bool {
        match self.bindings.get(key) {
            Some(patterns) => patterns.iter().any(|p| intent_core::glob_match(p, subject)),
            None => !self.require_binding,
        }
    }
}

fn parse_hex_digest(hex: &str) -> Option<[u8; 32]> {
    let bytes = hex.as_bytes();
    if bytes.len() != 64 {
        return None;
    }
    let nibble = |c: u8| (c as char).to_digit(16).map(|v| v as u8);
    let mut out = [0u8; 32];
    for (i, pair) in bytes.chunks(2).enumerate() {
        out[i] = nibble(pair[0])? << 4 | nibble(pair[1])?;
    }
    Some(out)
}

pub(crate) fn digest(key: &str) -> [u8; 32] {
    Sha256::digest(key.as_bytes()).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_parse_and_check() {
        let keys = ApiKeys::from_csv(" a ,b,, ");
        assert!(keys.check("a") && keys.check("b"));
        assert!(!keys.check("c") && !keys.check(""));
        assert!(ApiKeys::from_csv("").is_empty());
    }

    #[test]
    fn subject_bindings() {
        let hex = |k: &str| {
            digest(k)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        };
        let mut map = BTreeMap::new();
        map.insert(hex("svc"), vec!["svc-*".to_string()]);
        let keys = ApiKeys::from_csv("svc,other")
            .with_subject_bindings(&map, false)
            .unwrap();
        let svc = keys.authenticate("svc").unwrap();
        let other = keys.authenticate("other").unwrap();
        assert!(keys.subject_allowed(&svc, "svc-billing"));
        assert!(!keys.subject_allowed(&svc, "user-42"));
        assert!(keys.subject_allowed(&other, "anyone"));

        let strict = ApiKeys::from_csv("svc,other")
            .with_subject_bindings(&map, true)
            .unwrap();
        assert!(!strict.subject_allowed(&other, "anyone"));

        // Typos and unknown keys fail loudly.
        let mut bad = BTreeMap::new();
        bad.insert("zz".to_string(), vec!["*".to_string()]);
        assert!(
            ApiKeys::from_csv("svc")
                .with_subject_bindings(&bad, false)
                .is_err()
        );
        let mut unknown = BTreeMap::new();
        unknown.insert(hex("nope"), vec!["*".to_string()]);
        assert!(
            ApiKeys::from_csv("svc")
                .with_subject_bindings(&unknown, false)
                .is_err()
        );
        let mut empty = BTreeMap::new();
        empty.insert(hex("svc"), vec![]);
        assert!(
            ApiKeys::from_csv("svc")
                .with_subject_bindings(&empty, false)
                .is_err()
        );
    }

    #[test]
    fn rejects_unknown_fields_such_as_api_keys() {
        let toml = "key_path='k'\npolicy_path='p'\nrules_path='r'\napi_keys=['x']";
        assert!(ServerConfig::from_toml_str(toml).is_err());
    }

    #[test]
    fn defaults_apply() {
        let c =
            ServerConfig::from_toml_str("key_path='k'\npolicy_path='p'\nrules_path='r'").unwrap();
        assert_eq!(c.bind.to_string(), "127.0.0.1:8080");
        assert_eq!(c.max_body_bytes, 256 * 1024);
        assert!(!c.llm && !c.allow_unauthenticated && !c.watch);
        assert!(c.retired_keys_paths.is_empty() && c.rate_limit.is_none());
    }

    #[test]
    fn rate_limit_is_validated() {
        let base = "key_path='k'\npolicy_path='p'\nrules_path='r'\n";
        let ok =
            ServerConfig::from_toml_str(&format!("{base}[rate_limit]\nper_second=5\nburst=10"))
                .unwrap();
        assert_eq!(ok.rate_limit.unwrap().burst, 10);
        for bad in [
            "per_second=0\nburst=1",
            "per_second=1\nburst=0",
            "per_second=1",
        ] {
            assert!(
                ServerConfig::from_toml_str(&format!("{base}[rate_limit]\n{bad}")).is_err(),
                "{bad}"
            );
        }
    }
}
