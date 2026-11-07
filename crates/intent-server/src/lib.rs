//! HTTP service exposing intent evaluation, token verification and JWKS.
//!
//! Audit events are emitted on the `intent::audit` tracing target. They carry
//! identifiers and classification results only, never message content, tool
//! arguments, API keys or tokens.

mod config;
mod metrics;

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::{DefaultBodyLimit, Request, State};
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
use intent_token::{IssuerConfig, Jwks, JwtIssuer, KeyPair, TokenError, TokenVerifier};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

pub use config::{API_KEYS_ENV, ApiKeys, ServerConfig};
pub use metrics::Metrics;

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

/// Shared, cheaply clonable server state.
#[derive(Clone)]
pub struct AppState {
    evaluator: Evaluator,
    verifier: Arc<TokenVerifier>,
    jwks: Arc<Jwks>,
    api_keys: Arc<ApiKeys>,
    /// When false, the auth middleware lets everything through.
    require_auth: bool,
    max_body_bytes: usize,
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

        let key = KeyPair::load(&config.key_path)?;
        let issuer = JwtIssuer::new(
            IssuerConfig::new(config.issuer.clone(), config.audience.clone()),
            &key,
        )?;
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
        let mut state = Self::new(
            Evaluator::new(Arc::new(classifier), Arc::new(policy), Arc::new(issuer)),
            &jwks,
            &config.issuer,
            &config.audience,
            api_keys,
            metrics,
        )?;
        state.require_auth = require_auth;
        state.max_body_bytes = config.max_body_bytes;
        Ok(state)
    }

    /// Assembles state from prebuilt parts. Authentication is required iff
    /// `api_keys` is non-empty; the body limit defaults to 256 KiB.
    pub fn new(
        evaluator: Evaluator,
        jwks: &Jwks,
        issuer: &str,
        audience: &str,
        api_keys: ApiKeys,
        metrics: Option<Metrics>,
    ) -> Result<Self, Error> {
        Ok(Self {
            evaluator,
            verifier: Arc::new(TokenVerifier::from_jwks(jwks, issuer, audience)?),
            jwks: Arc::new(jwks.clone()),
            require_auth: !api_keys.is_empty(),
            api_keys: Arc::new(api_keys),
            max_body_bytes: 256 * 1024,
            metrics,
        })
    }

    pub fn with_max_body_bytes(mut self, max: usize) -> Self {
        self.max_body_bytes = max;
        self
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

// --- errors -------------------------------------------------------------

enum ApiError {
    Unauthorized,
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
    let evaluation = state.evaluator.evaluate(&req).await.map_err(|e| {
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
    match state
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
    Json((*state.jwks).clone())
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
