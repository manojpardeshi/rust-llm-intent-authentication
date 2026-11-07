//! Verify intent tokens inside a tool or API gateway.
//!
//! [`Verifier`] checks a token's signature and claims against the issuer's
//! JWKS (static or fetched and refreshed over HTTP) and that the tool calls
//! being executed are exactly those the token was issued for. With the `axum`
//! feature (default), [`IntentLayer`] wires this into an axum router.

mod error;
mod remote;

#[cfg(feature = "axum")]
mod middleware;

use std::sync::{Arc, RwLock};
use std::time::Instant;

use intent_core::ToolCall;
use intent_token::{Claims, Jwks, TokenError, TokenVerifier};
use tokio::sync::Mutex;

pub use error::VerifyError;
pub use remote::RemoteJwksOptions;

#[cfg(feature = "axum")]
pub use middleware::{IntentLayer, IntentService, VerifiedIntent, VerifiedToolCalls};

/// Verifies intent tokens. Cheap to clone; clones share key state.
#[derive(Clone)]
pub struct Verifier {
    inner: Arc<Inner>,
}

enum Inner {
    Static(TokenVerifier),
    Remote(Remote),
}

struct KeyState {
    verifier: Arc<TokenVerifier>,
    fetched_at: Instant,
    /// Last fetch attempt, successful or not; throttles refreshes.
    last_attempt: Instant,
}

struct Remote {
    issuer: String,
    audience: String,
    url: reqwest::Url,
    client: reqwest::Client,
    opts: RemoteJwksOptions,
    state: RwLock<KeyState>,
    /// Serializes refreshes so concurrent callers do not all fetch.
    refresh: Mutex<()>,
}

impl Verifier {
    /// Verifier with a fixed key set that is never refreshed.
    pub fn from_jwks(jwks: &Jwks, issuer: &str, audience: &str) -> Result<Self, VerifyError> {
        let verifier = TokenVerifier::from_jwks(jwks, issuer, audience)
            .map_err(|e| VerifyError::Config(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(Inner::Static(verifier)),
        })
    }

    /// Verifier that fetches the JWKS from `url` now (failing if that fetch
    /// fails) and keeps it fresh: periodically, and when a token names an
    /// unknown `kid` (rate limited by `min_refresh_interval`).
    pub async fn from_jwks_url(
        url: &str,
        issuer: &str,
        audience: &str,
        opts: RemoteJwksOptions,
    ) -> Result<Self, VerifyError> {
        let url = remote::parse_url(url, &opts)?;
        let client = remote::build_client(&opts)?;
        let jwks = remote::fetch_jwks(&client, &url, opts.max_body_bytes).await?;
        let verifier = TokenVerifier::from_jwks(&jwks, issuer, audience)
            .map_err(|e| VerifyError::Config(e.to_string()))?;
        let now = Instant::now();
        Ok(Self {
            inner: Arc::new(Inner::Remote(Remote {
                issuer: issuer.to_owned(),
                audience: audience.to_owned(),
                url,
                client,
                opts,
                state: RwLock::new(KeyState {
                    verifier: Arc::new(verifier),
                    fetched_at: now,
                    last_attempt: now,
                }),
                refresh: Mutex::new(()),
            })),
        })
    }

    /// Verifies signature, standard claims, an optional required scope, and
    /// that `tool_calls` are exactly what the token was issued for.
    pub async fn verify(
        &self,
        token: &str,
        tool_calls: &[ToolCall],
        required_scope: Option<&str>,
    ) -> Result<Claims, VerifyError> {
        match &*self.inner {
            Inner::Static(v) => Ok(v.verify(token, tool_calls, required_scope)?),
            Inner::Remote(r) => r.verify(token, tool_calls, required_scope).await,
        }
    }
}

impl Remote {
    fn current(&self) -> Arc<TokenVerifier> {
        self.state.read().expect("poisoned").verifier.clone()
    }

    async fn verify(
        &self,
        token: &str,
        tool_calls: &[ToolCall],
        required_scope: Option<&str>,
    ) -> Result<Claims, VerifyError> {
        self.refresh_if_stale().await;
        let seen = self.current();
        match seen.verify(token, tool_calls, required_scope) {
            Err(TokenError::UnknownKey(_)) => {
                self.refresh_for_unknown_key(&seen).await;
                Ok(self.current().verify(token, tool_calls, required_scope)?)
            }
            other => Ok(other?),
        }
    }

    fn stale(&self) -> bool {
        let s = self.state.read().expect("poisoned");
        s.fetched_at.elapsed() >= self.opts.refresh_interval
            && s.last_attempt.elapsed() >= self.opts.min_refresh_interval
    }

    async fn refresh_if_stale(&self) {
        if !self.stale() {
            return;
        }
        let _guard = self.refresh.lock().await;
        if self.stale() {
            self.fetch().await;
        }
    }

    async fn refresh_for_unknown_key(&self, seen: &Arc<TokenVerifier>) {
        let _guard = self.refresh.lock().await;
        let throttled = {
            let s = self.state.read().expect("poisoned");
            // Someone else already refreshed while we waited for the lock.
            if !Arc::ptr_eq(&s.verifier, seen) {
                return;
            }
            s.last_attempt.elapsed() < self.opts.min_refresh_interval
        };
        if !throttled {
            self.fetch().await;
        }
    }

    /// Fetches and swaps in a new key set. Must be called with `refresh` held.
    /// A failure keeps the old keys.
    async fn fetch(&self) {
        let result = remote::fetch_jwks(&self.client, &self.url, self.opts.max_body_bytes)
            .await
            .and_then(|jwks| {
                TokenVerifier::from_jwks(&jwks, &self.issuer, &self.audience)
                    .map_err(|e| VerifyError::Config(e.to_string()))
            });
        let mut s = self.state.write().expect("poisoned");
        s.last_attempt = Instant::now();
        match result {
            Ok(v) => {
                s.verifier = Arc::new(v);
                s.fetched_at = s.last_attempt;
            }
            Err(e) => {
                tracing::warn!(target: "intent::gateway", error = %e, "JWKS refresh failed; keeping previous keys");
            }
        }
    }
}
