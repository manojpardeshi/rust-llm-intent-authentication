use intent_token::TokenError;

/// Errors from building or using a [`crate::Verifier`].
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    /// Invalid verifier configuration (bad URL, unusable key set, ...).
    #[error("config: {0}")]
    Config(String),
    /// Fetching the JWKS failed. The message never contains secrets.
    #[error("fetching JWKS: {0}")]
    Fetch(String),
    /// The token (or the request it is presented with) was rejected.
    #[error(transparent)]
    Token(#[from] TokenError),
}

impl VerifyError {
    /// Short, token-free category, matching intent-server's `/v1/verify`.
    pub fn category(&self) -> &'static str {
        match self {
            Self::Token(e) => e.category(),
            Self::Config(_) | Self::Fetch(_) => "invalid_token",
        }
    }
}
