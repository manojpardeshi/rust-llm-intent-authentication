//! Fetching a JWKS over HTTP.

use std::net::IpAddr;
use std::time::Duration;

use intent_token::Jwks;

use crate::VerifyError;

/// Tuning for a verifier that follows a remote JWKS endpoint.
#[derive(Debug, Clone)]
pub struct RemoteJwksOptions {
    /// Keys older than this are refreshed before verifying.
    pub refresh_interval: Duration,
    /// Minimum time between fetch attempts, so unknown key ids cannot be used
    /// to hammer the issuer.
    pub min_refresh_interval: Duration,
    /// Timeout for each fetch.
    pub timeout: Duration,
    /// Maximum accepted response body size.
    pub max_body_bytes: usize,
    /// Allow plain `http://` to non-loopback hosts. Off by default.
    pub allow_insecure_http: bool,
}

impl Default for RemoteJwksOptions {
    fn default() -> Self {
        Self {
            refresh_interval: Duration::from_secs(300),
            min_refresh_interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            max_body_bytes: 64 * 1024,
            allow_insecure_http: false,
        }
    }
}

pub(crate) fn parse_url(url: &str, opts: &RemoteJwksOptions) -> Result<reqwest::Url, VerifyError> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|_| VerifyError::Config("JWKS URL is not a valid URL".into()))?;
    match parsed.scheme() {
        "https" => {}
        "http" => {
            if !opts.allow_insecure_http && !is_loopback(&parsed) {
                return Err(VerifyError::Config(
                    "JWKS URL must use https (http is only allowed for loopback hosts \
                     or with allow_insecure_http)"
                        .into(),
                ));
            }
        }
        other => {
            return Err(VerifyError::Config(format!(
                "unsupported JWKS URL scheme {other:?}"
            )));
        }
    }
    if parsed.host_str().is_none() {
        return Err(VerifyError::Config("JWKS URL has no host".into()));
    }
    Ok(parsed)
}

fn is_loopback(url: &reqwest::Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.trim_start_matches('[')
        .trim_end_matches(']')
        .parse::<IpAddr>()
        .is_ok_and(|ip| ip.is_loopback())
}

pub(crate) fn build_client(opts: &RemoteJwksOptions) -> Result<reqwest::Client, VerifyError> {
    reqwest::Client::builder()
        .timeout(opts.timeout)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| VerifyError::Config(format!("building HTTP client: {}", e.without_url())))
}

pub(crate) async fn fetch_jwks(
    client: &reqwest::Client,
    url: &reqwest::Url,
    max_body_bytes: usize,
) -> Result<Jwks, VerifyError> {
    let fetch =
        |what: &str, e: reqwest::Error| VerifyError::Fetch(format!("{what}: {}", e.without_url()));
    let mut res = client
        .get(url.clone())
        .header(reqwest::header::ACCEPT, "application/json")
        .send()
        .await
        .map_err(|e| fetch("request failed", e))?;
    if !res.status().is_success() {
        return Err(VerifyError::Fetch(format!("status {}", res.status())));
    }
    let too_big = || VerifyError::Fetch(format!("response exceeds {max_body_bytes} bytes"));
    if res
        .content_length()
        .is_some_and(|n| n > max_body_bytes as u64)
    {
        return Err(too_big());
    }
    // Content-Length may be absent or wrong, so also count what is received.
    let mut body = Vec::new();
    while let Some(chunk) = res
        .chunk()
        .await
        .map_err(|e| fetch("reading body failed", e))?
    {
        if body.len() + chunk.len() > max_body_bytes {
            return Err(too_big());
        }
        body.extend_from_slice(&chunk);
    }
    let jwks: Jwks = serde_json::from_slice(&body)
        .map_err(|e| VerifyError::Fetch(format!("response is not a JWKS: {e}")))?;
    if jwks.keys.is_empty() {
        return Err(VerifyError::Fetch("JWKS contains no keys".into()));
    }
    Ok(jwks)
}
