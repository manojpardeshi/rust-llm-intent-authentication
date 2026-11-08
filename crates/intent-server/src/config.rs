//! Server configuration (TOML) and API key handling.

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

/// SHA-256 digests of the accepted API keys.
#[derive(Debug, Clone, Default)]
pub struct ApiKeys {
    digests: Vec<[u8; 32]>,
}

impl ApiKeys {
    /// Parses a comma-separated key list; blank entries are ignored.
    pub fn from_csv(csv: &str) -> Self {
        Self::from_keys(csv.split(',').map(str::trim).filter(|k| !k.is_empty()))
    }

    pub fn from_keys<'a>(keys: impl IntoIterator<Item = &'a str>) -> Self {
        Self {
            digests: keys.into_iter().map(digest).collect(),
        }
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
        let d = digest(presented);
        let mut ok = subtle::Choice::from(0);
        for stored in &self.digests {
            ok |= stored.ct_eq(&d);
        }
        ok.into()
    }
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
