//! HTTP service exposing intent evaluation, token verification and JWKS.
//!
//! Audit events are emitted on the `intent::audit` tracing target. They carry
//! identifiers and classification results only, never message content, tool
//! arguments, API keys or tokens.

mod config;
mod metrics;
mod ratelimit;
mod reload;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use axum::extract::{ConnectInfo, DefaultBodyLimit, Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use intent_classify::{DEFAULT_THRESHOLD, HybridClassifier, RuleClassifier};
use intent_core::{
    Decision, Evaluator, IntentClassifier, IntentRequest, ToolCall, encode_digest, request_digest,
};
use intent_policy::Policy;
use intent_token::{IssuerConfig, Jwk, Jwks, JwtIssuer, KeyPair, TokenError, TokenVerifier};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use ratelimit::{KeyedLimiter, RateKey};

pub use config::{API_KEYS_ENV, ApiKeys, RateLimitConfig, ServerConfig};
pub use metrics::Metrics;
pub use reload::{ReloadGuard, spawn_reload_triggers};

/// Per-request timeout.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Errors while building the server state.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("config: {0}")]
    Config(String),
    #[error("signing key: {0}")]
    Key(#[from] TokenError),
    #[error("policy: {0}")]
    Policy(#[from] intent_policy::PolicyError),
    #[error("classifier: {0}")]
    Classifier(String),
    #[error(
        "no API keys configured: set {API_KEYS_ENV} or, for development only, allow_unauthenticated = true"
    )]
    NoApiKeys,
}

/// Everything derived from files on disk. Immutable; replaced as a whole on
/// reload so a request never sees a mix of old and new parts.
struct Snapshot {
    evaluator: Evaluator,
    verifier: TokenVerifier,
    jwks: Jwks,
    /// Active signing `kid`, for logging.
    kid: String,
}

impl Snapshot {
    /// Builds and fully validates a snapshot from the config's files.
    fn from_config(config: &ServerConfig) -> Result<Self, Error> {
        let key = KeyPair::load(&config.key_path)?;
        let retired = load_retired_keys(&config.retired_keys_paths)?;
        let issuer = JwtIssuer::new(
            IssuerConfig::new(config.issuer.clone(), config.audience.clone()),
            &key,
        )?
        .with_retired_keys(retired)?;
        let policy = Policy::from_path(&config.policy_path)?;
        let rules = RuleClassifier::from_path(&config.rules_path)
            .map_err(|e| Error::Classifier(e.to_string()))?;
        let fallback = if config.llm {
            Some(llm_fallback()?)
        } else {
            None
        };
        let classifier = HybridClassifier::new(rules, fallback)
            .with_threshold(config.threshold.unwrap_or(DEFAULT_THRESHOLD));

        let jwks = issuer.jwks();
        let kid = issuer.kid().to_owned();
        Ok(Self {
            evaluator: Evaluator::new(Arc::new(classifier), Arc::new(policy), Arc::new(issuer)),
            verifier: TokenVerifier::from_jwks(&jwks, &config.issuer, &config.audience)?,
            jwks,
            kid,
        })
    }
}

/// Reads public-only JWKS files. Each path is a JWKS file or a directory
/// whose `*.json` files are all read (in name order), so a rotation only
/// needs a new file dropped in plus a reload. A file containing private key
/// material (`d`) is rejected rather than silently trusted.
fn load_retired_keys(paths: &[PathBuf]) -> Result<Vec<Jwk>, Error> {
    let mut files = Vec::new();
    for path in paths {
        if path.is_dir() {
            let mut found = Vec::new();
            let entries = std::fs::read_dir(path)
                .map_err(|e| Error::Config(format!("reading {}: {e}", path.display())))?;
            for entry in entries {
                let p = entry
                    .map_err(|e| Error::Config(format!("reading {}: {e}", path.display())))?
                    .path();
                if p.is_file() && p.extension().is_some_and(|x| x == "json") {
                    found.push(p);
                }
            }
            found.sort();
            files.extend(found);
        } else {
            files.push(path.clone());
        }
    }
    let mut out = Vec::new();
    for path in &files {
        let bytes = std::fs::read(path)
            .map_err(|e| Error::Config(format!("reading {}: {e}", path.display())))?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Config(format!("parsing {}: {e}", path.display())))?;
        let has_private = value["keys"]
            .as_array()
            .is_some_and(|ks| ks.iter().any(|k| k.get("d").is_some()));
        if has_private {
            return Err(Error::Config(format!(
                "{} contains private key material; retired keys must be public JWKS only",
                path.display()
            )));
        }
        let jwks: Jwks = serde_json::from_value(value)
            .map_err(|e| Error::Config(format!("parsing {}: {e}", path.display())))?;
        out.extend(jwks.keys);
    }
    Ok(out)
}

/// Shared, cheaply clonable server state.
///
/// File-derived parts live in a [`Snapshot`] swapped atomically by
/// [`AppState::reload`]. Bind address, API keys, body limit and rate limits
/// are fixed at construction and are not reloadable.
#[derive(Clone)]
pub struct AppState {
    snapshot: Arc<ArcSwap<Snapshot>>,
    /// Config to rebuild snapshots from; `None` for [`AppState::new`].
    reload_config: Option<Arc<ServerConfig>>,
    api_keys: Arc<ApiKeys>,
    /// When false, the auth middleware lets everything through.
    require_auth: bool,
    max_body_bytes: usize,
    rate_limiter: Option<Arc<KeyedLimiter>>,
    metrics: Option<Metrics>,
}

impl AppState {
    /// Builds state from a config, loading the key, policy and rules files.
    ///
    /// Fails with [`Error::NoApiKeys`] when `api_keys` is empty and the config
    /// does not set `allow_unauthenticated`.
    pub fn from_config(
        config: &ServerConfig,
        api_keys: ApiKeys,
        metrics: Option<Metrics>,
    ) -> Result<Self, Error> {
        let require_auth = !api_keys.is_empty();
        if !require_auth {
            if !config.allow_unauthenticated {
                return Err(Error::NoApiKeys);
            }
            tracing::warn!(
                "running WITHOUT authentication (allow_unauthenticated = true); \
                 anyone who can reach this port can obtain tokens"
            );
        }
        if let Some(rl) = &config.rate_limit {
            rl.validate()?;
        }
        let snapshot = Snapshot::from_config(config)?;
        Ok(Self {
            snapshot: Arc::new(ArcSwap::from_pointee(snapshot)),
            reload_config: Some(Arc::new(config.clone())),
            api_keys: Arc::new(api_keys),
            require_auth,
            max_body_bytes: config.max_body_bytes,
            rate_limiter: config.rate_limit.map(|c| Arc::new(KeyedLimiter::new(c))),
            metrics,
        })
    }

    /// Assembles state from prebuilt parts. Authentication is required iff
    /// `api_keys` is non-empty; the body limit defaults to 256 KiB.
    ///
    /// Such state has no config to rebuild from, so [`AppState::reload`]
    /// always fails with [`Error::Config`] and leaves the state untouched.
    pub fn new(
        evaluator: Evaluator,
        jwks: &Jwks,
        issuer: &str,
        audience: &str,
        api_keys: ApiKeys,
        metrics: Option<Metrics>,
    ) -> Result<Self, Error> {
        let snapshot = Snapshot {
            evaluator,
            verifier: TokenVerifier::from_jwks(jwks, issuer, audience)?,
            jwks: jwks.clone(),
            kid: jwks.keys.first().map(|k| k.kid.clone()).unwrap_or_default(),
        };
        Ok(Self {
            snapshot: Arc::new(ArcSwap::from_pointee(snapshot)),
            reload_config: None,
            require_auth: !api_keys.is_empty(),
            api_keys: Arc::new(api_keys),
            max_body_bytes: 256 * 1024,
            rate_limiter: None,
            metrics,
        })
    }

    pub fn with_max_body_bytes(mut self, max: usize) -> Self {
        self.max_body_bytes = max;
        self
    }

    /// Enables per-client rate limiting on /v1/evaluate and /v1/verify.
    pub fn with_rate_limit(mut self, config: RateLimitConfig) -> Result<Self, Error> {
        config.validate()?;
        self.rate_limiter = Some(Arc::new(KeyedLimiter::new(config)));
        Ok(self)
    }

    /// Rebuilds the signing key, retired keys, policy, rules, evaluator,
    /// verifier and JWKS from the config files and swaps them in atomically.
    ///
    /// On any error the previous snapshot keeps serving unchanged.
    pub fn reload(&self) -> Result<(), Error> {
        let result = self.try_reload();
        match &result {
            Ok(()) => {
                tracing::info!(
                    target: "intent::audit",
                    kid = %self.snapshot.load().kid,
                    "configuration reloaded"
                );
            }
            Err(e) => {
                tracing::error!(
                    target: "intent::server",
                    error = %e,
                    "configuration reload failed; keeping previous configuration"
                );
            }
        }
        if let Some(m) = &self.metrics {
            m.record_reload(if result.is_ok() { "ok" } else { "error" });
        }
        result
    }

    fn try_reload(&self) -> Result<(), Error> {
        let config = self
            .reload_config
            .as_ref()
            .ok_or_else(|| Error::Config("no config to reload from".into()))?;
        let snapshot = Snapshot::from_config(config)?;
        self.snapshot.store(Arc::new(snapshot));
        Ok(())
    }

    /// The config this state reloads from, if any.
    pub fn config(&self) -> Option<&ServerConfig> {
        self.reload_config.as_deref()
    }
}

#[cfg(feature = "llm")]
fn llm_fallback() -> Result<Arc<dyn IntentClassifier>, Error> {
    use intent_classify::{LlmClassifier, LlmConfig};
    let config = LlmConfig::from_env().map_err(|e| Error::Classifier(e.to_string()))?;
    let classifier = LlmClassifier::new(config).map_err(|e| Error::Classifier(e.to_string()))?;
    Ok(Arc::new(classifier))
}

#[cfg(not(feature = "llm"))]
fn llm_fallback() -> Result<Arc<dyn IntentClassifier>, Error> {
    Err(Error::Classifier(
        "this build was compiled without the llm feature".into(),
    ))
}

/// Builds the HTTP router.
pub fn router(state: AppState) -> Router {
    let protected = Router::new()
        .route("/v1/evaluate", post(evaluate))
        .route("/v1/verify", post(verify))
        // Layers added later run first: authenticate, then rate limit, so
        // unauthenticated traffic never consumes a valid key's bucket.
        .route_layer(middleware::from_fn_with_state(state.clone(), rate_limit))
        .route_layer(middleware::from_fn_with_state(state.clone(), authenticate));

    Router::new()
        .merge(protected)
        .route("/.well-known/jwks.json", get(jwks))
        .route("/healthz", get(healthz))
        .route("/metrics", get(metrics_text))
        .layer(DefaultBodyLimit::max(state.max_body_bytes))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            REQUEST_TIMEOUT,
        ))
        // Default span records method and URI only, never headers or bodies.
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

// --- auth ---------------------------------------------------------------

fn presented_key(headers: &HeaderMap) -> Option<&str> {
    if let Some(v) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        let (scheme, rest) = v.split_once(' ')?;
        return scheme.eq_ignore_ascii_case("bearer").then_some(rest.trim());
    }
    headers.get("x-api-key").and_then(|v| v.to_str().ok())
}

async fn authenticate(State(state): State<AppState>, req: Request, next: Next) -> Response {
    if state.require_auth && !presented_key(req.headers()).is_some_and(|k| state.api_keys.check(k))
    {
        return ApiError::Unauthorized.into_response();
    }
    next.run(req).await
}

async fn rate_limit(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let Some(limiter) = &state.rate_limiter else {
        return next.run(req).await;
    };
    // Authentication already succeeded (or is disabled) at this point.
    let key = match presented_key(req.headers()) {
        Some(k) if state.require_auth => RateKey::ApiKey(config::digest(k)),
        _ => req
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .map_or(RateKey::Shared, |c| RateKey::Ip(c.0.ip())),
    };
    match limiter.check(&key) {
        Ok(()) => next.run(req).await,
        Err(wait) => {
            if let Some(m) = &state.metrics {
                m.record_rate_limited();
            }
            ApiError::RateLimited(wait).into_response()
        }
    }
}

// --- errors -------------------------------------------------------------

enum ApiError {
    Unauthorized,
    RateLimited(Duration),
    Internal,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            Self::Unauthorized => {
                let mut res = (
                    StatusCode::UNAUTHORIZED,
                    Json(json!({"error": "unauthorized"})),
                )
                    .into_response();
                res.headers_mut().insert(
                    header::WWW_AUTHENTICATE,
                    header::HeaderValue::from_static("Bearer"),
                );
                res
            }
            Self::RateLimited(wait) => {
                let secs = wait.as_secs() + u64::from(wait.subsec_nanos() > 0);
                let mut res = (
                    StatusCode::TOO_MANY_REQUESTS,
                    Json(json!({"error": "rate_limited"})),
                )
                    .into_response();
                res.headers_mut()
                    .insert(header::RETRY_AFTER, secs.max(1).into());
                res
            }
            Self::Internal => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "internal error"})),
            )
                .into_response(),
        }
    }
}

// --- handlers -----------------------------------------------------------

async fn evaluate(
    State(state): State<AppState>,
    Json(req): Json<IntentRequest>,
) -> Result<Response, ApiError> {
    let started = Instant::now();
    let snap = state.snapshot.load_full();
    let evaluation = snap.evaluator.evaluate(&req).await.map_err(|e| {
        tracing::error!(target: "intent::server", error = %e, "evaluation failed");
        ApiError::Internal
    })?;
    let req_hash = request_digest(&req)
        .map(|d| encode_digest(&d))
        .map_err(|e| {
            tracing::error!(target: "intent::server", error = %e, "request digest failed");
            ApiError::Internal
        })?;

    let (name, status, reason) = match &evaluation.decision {
        Decision::Allow { .. } => ("allow", StatusCode::OK, None),
        Decision::Deny { reason } => ("deny", StatusCode::FORBIDDEN, Some(reason.as_str())),
        Decision::StepUp { reason } => ("step_up", StatusCode::ACCEPTED, Some(reason.as_str())),
    };
    let intent = &evaluation.intent;
    tracing::info!(
        target: "intent::audit",
        subject = %req.subject,
        session_id = req.session_id.as_deref(),
        category = ?intent.category,
        action = %intent.action,
        resource = intent.resource.as_deref(),
        confidence = intent.confidence,
        source = ?intent.source,
        decision = name,
        reason,
        jti = evaluation.token.as_ref().map(|t| t.jti.as_str()),
        req_hash = %req_hash,
        "evaluate"
    );
    if let Some(m) = &state.metrics {
        m.record_evaluation(name, format!("{:?}", intent.category), started.elapsed());
    }

    let mut body = serde_json::to_value(&evaluation).map_err(|_| ApiError::Internal)?;
    body["req_hash"] = Value::String(req_hash);
    Ok((status, Json(body)).into_response())
}

#[derive(Deserialize)]
struct VerifyRequest {
    token: String,
    tool_calls: Vec<ToolCall>,
    #[serde(default)]
    scope: Option<String>,
}

#[derive(Serialize)]
struct VerifyError {
    valid: bool,
    error: &'static str,
}

async fn verify(State(state): State<AppState>, Json(req): Json<VerifyRequest>) -> Response {
    let snap = state.snapshot.load();
    match snap
        .verifier
        .verify(&req.token, &req.tool_calls, req.scope.as_deref())
    {
        Ok(claims) => {
            tracing::info!(
                target: "intent::audit",
                result = "valid",
                jti = %claims.jti,
                sub = %claims.sub,
                "verify"
            );
            if let Some(m) = &state.metrics {
                m.record_verification("valid");
            }
            Json(json!({"valid": true, "claims": claims})).into_response()
        }
        Err(e) => {
            let error = e.category();
            tracing::info!(target: "intent::audit", result = "invalid", error, "verify");
            if let Some(m) = &state.metrics {
                m.record_verification("invalid");
            }
            (
                StatusCode::UNAUTHORIZED,
                Json(VerifyError {
                    valid: false,
                    error,
                }),
            )
                .into_response()
        }
    }
}

async fn jwks(State(state): State<AppState>) -> Json<Jwks> {
    Json(state.snapshot.load().jwks.clone())
}

async fn healthz() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn metrics_text(State(state): State<AppState>) -> Response {
    match &state.metrics {
        Some(m) => (
            [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
            m.render(),
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}
