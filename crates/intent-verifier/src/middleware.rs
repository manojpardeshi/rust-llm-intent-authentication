//! tower/axum middleware that gates routes on a valid intent token.

use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::body::{Body, to_bytes};
use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode, header};
use axum::response::{IntoResponse, Response};
use intent_core::ToolCall;
use intent_token::Claims;
use serde_json::{Value, json};
use tower::{Layer, Service};

use crate::Verifier;

const DEFAULT_MAX_BODY: usize = 256 * 1024;
const INTENT_TOKEN_HEADER: &str = "x-intent-token";

/// Claims of the token that authorized this request. Inserted into request
/// extensions by [`IntentLayer`]; also usable as an axum extractor.
#[derive(Debug, Clone)]
pub struct VerifiedIntent(pub Claims);

impl<S: Send + Sync> FromRequestParts<S> for VerifiedIntent {
    type Rejection = (StatusCode, &'static str);

    async fn from_request_parts(parts: &mut Parts, _: &S) -> Result<Self, Self::Rejection> {
        parts.extensions.get::<VerifiedIntent>().cloned().ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            "IntentLayer is not installed on this route",
        ))
    }
}

/// The tool calls the token was verified against. Handlers should execute
/// these rather than re-parsing the body, so what runs is exactly what was
/// authorized. Inserted by [`IntentLayer`]; also usable as an axum extractor.
#[derive(Debug, Clone)]
pub struct VerifiedToolCalls(pub Vec<ToolCall>);

impl<S: Send + Sync> FromRequestParts<S> for VerifiedToolCalls {
    type Rejection = (StatusCode, &'static str);

    async fn from_request_parts(parts: &mut Parts, _: &S) -> Result<Self, Self::Rejection> {
        parts.extensions.get::<VerifiedToolCalls>().cloned().ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            "IntentLayer is not installed on this route",
        ))
    }
}

/// Layer that verifies the intent token against the tool call(s) in the body.
///
/// The token comes from `Authorization: Bearer <jwt>` or `x-intent-token`.
/// The body may be a single tool call object, `{"tool_calls": [...]}` or a
/// JSON array of tool calls. Objects may not carry extra keys, so a handler
/// that parses the body differently cannot see a call that was not verified.
#[derive(Clone)]
pub struct IntentLayer {
    verifier: Verifier,
    scope: Option<Arc<str>>,
    max_body_bytes: usize,
}

impl IntentLayer {
    pub fn new(verifier: Verifier) -> Self {
        Self {
            verifier,
            scope: None,
            max_body_bytes: DEFAULT_MAX_BODY,
        }
    }

    /// Requires the token to grant `scope`.
    pub fn require_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(Arc::from(scope.into()));
        self
    }

    /// Maximum request body size (default 256 KiB); larger gets a 413.
    pub fn max_body_bytes(mut self, max: usize) -> Self {
        self.max_body_bytes = max;
        self
    }
}

impl<S> Layer<S> for IntentLayer {
    type Service = IntentService<S>;

    fn layer(&self, inner: S) -> IntentService<S> {
        IntentService {
            inner,
            cfg: self.clone(),
        }
    }
}

#[derive(Clone)]
pub struct IntentService<S> {
    inner: S,
    cfg: IntentLayer,
}

impl<S> Service<Request<Body>> for IntentService<S>
where
    S: Service<Request<Body>, Response = Response, Error = Infallible> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Response;
    type Error = Infallible;
    type Future = Pin<Box<dyn Future<Output = Result<Response, Infallible>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        // Use the instance that was driven to readiness, leave a clone behind.
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);
        let cfg = self.cfg.clone();
        Box::pin(async move {
            match authorize(&cfg, req).await {
                Ok(req) => inner.call(req).await,
                Err(res) => Ok(*res),
            }
        })
    }
}

async fn authorize(cfg: &IntentLayer, req: Request<Body>) -> Result<Request<Body>, Box<Response>> {
    let (mut parts, body) = req.into_parts();

    let Some(token) = extract_token(&parts.headers) else {
        return Err(reject(Failure::Unauthorized("missing_token")).into());
    };

    let bytes = to_bytes(body, cfg.max_body_bytes).await.map_err(|e| {
        if is_length_limit(&e) {
            Box::new(reject(Failure::TooLarge))
        } else {
            Box::new(reject(Failure::BadRequest("unreadable request body")))
        }
    })?;
    let tool_calls =
        parse_tool_calls(&bytes).map_err(|m| Box::new(reject(Failure::BadRequest(m))))?;

    match cfg
        .verifier
        .verify(&token, &tool_calls, cfg.scope.as_deref())
        .await
    {
        Ok(claims) => {
            tracing::info!(
                target: "intent::gateway",
                result = "valid",
                jti = %claims.jti,
                sub = %claims.sub,
                "intent token verified"
            );
            parts.extensions.insert(VerifiedIntent(claims));
            parts.extensions.insert(VerifiedToolCalls(tool_calls));
            Ok(Request::from_parts(parts, Body::from(bytes)))
        }
        Err(e) => {
            let error = e.category();
            tracing::info!(target: "intent::gateway", result = "invalid", error, "intent token rejected");
            Err(Box::new(reject(Failure::Unauthorized(error))))
        }
    }
}

fn extract_token(headers: &HeaderMap) -> Option<String> {
    if let Some((scheme, rest)) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split_once(' '))
        && scheme.eq_ignore_ascii_case("bearer")
        && !rest.trim().is_empty()
    {
        return Some(rest.trim().to_owned());
    }
    headers
        .get(INTENT_TOKEN_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_owned)
}

fn parse_tool_calls(bytes: &[u8]) -> Result<Vec<ToolCall>, &'static str> {
    const SHAPE: &str = "body must be a tool call, an array or {\"tool_calls\": [...]}";
    let value: Value = serde_json::from_slice(bytes).map_err(|_| "body is not valid JSON")?;
    let calls: Vec<Value> = match value {
        Value::Array(calls) => calls,
        Value::Object(mut o) if o.contains_key("tool_calls") => {
            if o.len() != 1 {
                return Err(SHAPE);
            }
            match o.remove("tool_calls") {
                Some(Value::Array(calls)) => calls,
                _ => return Err(SHAPE),
            }
        }
        call @ Value::Object(_) => vec![call],
        _ => return Err(SHAPE),
    };
    if calls.is_empty() {
        return Err("no tool calls in body");
    }
    calls.into_iter().map(strict_tool_call).collect()
}

/// A tool call object with no keys besides `name` and `arguments`.
fn strict_tool_call(call: Value) -> Result<ToolCall, &'static str> {
    const INVALID: &str = "body does not contain valid tool calls";
    match &call {
        Value::Object(o) if o.keys().all(|k| k == "name" || k == "arguments") => {}
        _ => return Err(INVALID),
    }
    serde_json::from_value(call).map_err(|_| INVALID)
}

fn is_length_limit(e: &axum::Error) -> bool {
    let mut cur: Option<&(dyn std::error::Error + 'static)> = Some(e);
    while let Some(err) = cur {
        if err.is::<http_body_util::LengthLimitError>() {
            return true;
        }
        cur = err.source();
    }
    false
}

enum Failure {
    Unauthorized(&'static str),
    BadRequest(&'static str),
    TooLarge,
}

fn reject(f: Failure) -> Response {
    match f {
        Failure::Unauthorized(error) => {
            let mut res = (
                StatusCode::UNAUTHORIZED,
                axum::Json(json!({"error": error})),
            )
                .into_response();
            res.headers_mut().insert(
                header::WWW_AUTHENTICATE,
                HeaderValue::from_static(r#"Bearer error="invalid_token""#),
            );
            res
        }
        Failure::BadRequest(msg) => {
            tracing::info!(target: "intent::gateway", result = "invalid", error = "bad_request", "request rejected");
            (StatusCode::BAD_REQUEST, axum::Json(json!({"error": msg}))).into_response()
        }
        Failure::TooLarge => {
            tracing::info!(target: "intent::gateway", result = "invalid", error = "too_large", "request rejected");
            (
                StatusCode::PAYLOAD_TOO_LARGE,
                axum::Json(json!({"error": "request body too large"})),
            )
                .into_response()
        }
    }
}
