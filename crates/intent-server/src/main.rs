use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use intent_server::{ApiKeys, AppState, Metrics, ServerConfig, router};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "intent-server",
    about = "Intent-gated token HTTP service",
    version
)]
struct Args {
    /// Path to the server TOML config.
    #[arg(long)]
    config: PathBuf,
    /// Emit logs as JSON lines.
    #[arg(long)]
    log_json: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let builder = tracing_subscriber::fmt().with_env_filter(filter);
    if args.log_json {
        builder.json().init();
    } else {
        builder.init();
    }

    let config = ServerConfig::from_path(&args.config)?;
    let state = AppState::from_config(&config, ApiKeys::from_env(), Some(Metrics::new()))?;
    let listener = tokio::net::TcpListener::bind(config.bind)
        .await
        .with_context(|| format!("binding {}", config.bind))?;
    tracing::info!(bind = %config.bind, "intent-server listening");
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await?;
    Ok(())
}
