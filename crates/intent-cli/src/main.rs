use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use intent_classify::{DEFAULT_THRESHOLD, HybridClassifier, RuleClassifier};
#[cfg(feature = "llm")]
use intent_classify::{LlmClassifier, LlmConfig};
use intent_core::{
    ClassifierSource, Decision, Evaluator, Intent, IntentCategory, IntentClassifier, IntentRequest,
    TokenIssuer, ToolCall,
};
use intent_policy::Policy;
use intent_token::{IssuerConfig, Jwks, JwtIssuer, KeyPair, TokenVerifier};

#[derive(Parser)]
#[command(name = "intent", about = "Intent-gated token framework CLI", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a new Ed25519 signing key file.
    Keygen {
        #[arg(long, default_value = "signing.key.json")]
        out: PathBuf,
    },
    /// Print the public JWK set for a signing key.
    Jwks {
        #[arg(long)]
        key: PathBuf,
    },
    /// Classify a request, apply policy, and issue a token if allowed.
    ///
    /// Exit code: 0 allow, 2 deny, 3 step-up. The LLM fallback is opt-in via
    /// --llm, so low-confidence results are left to policy by default.
    Evaluate {
        #[arg(long)]
        key: PathBuf,
        #[arg(long, default_value = "config/policy.example.toml")]
        policy: PathBuf,
        #[arg(long, default_value = "config/rules.example.toml")]
        rules: PathBuf,
        /// JSON file containing an IntentRequest.
        #[arg(long)]
        request: PathBuf,
        #[arg(long, default_value_t = DEFAULT_THRESHOLD)]
        threshold: f32,
        /// Use Claude (ANTHROPIC_API_KEY) as fallback for low-confidence rule results
        #[arg(long)]
        llm: bool,
        #[command(flatten)]
        ids: Ids,
    },
    /// Validate policy and/or classifier rule files.
    Lint {
        #[arg(long)]
        policy: Option<PathBuf>,
        #[arg(long)]
        rules: Option<PathBuf>,
    },
    /// DEV ONLY: issue a token for a request with a manually supplied intent,
    /// bypassing classification and policy.
    Issue {
        #[arg(long)]
        key: PathBuf,
        /// JSON file containing an IntentRequest.
        #[arg(long)]
        request: PathBuf,
        #[arg(long, value_parser = parse_category)]
        category: IntentCategory,
        #[arg(long)]
        action: String,
        #[arg(long)]
        resource: Option<String>,
        #[arg(long = "scope", required = true)]
        scopes: Vec<String>,
        #[arg(long, default_value_t = 120)]
        ttl: u64,
        #[command(flatten)]
        ids: Ids,
    },
    /// Verify a token against a JWK set and the tool calls about to run.
    Verify {
        /// JWKS JSON file.
        #[arg(long)]
        jwks: PathBuf,
        #[arg(long)]
        token: String,
        /// JSON file with either a tool call array or a full IntentRequest.
        #[arg(long)]
        tool_calls: PathBuf,
        #[arg(long)]
        scope: Option<String>,
        #[command(flatten)]
        ids: Ids,
    },
}

#[derive(clap::Args)]
struct Ids {
    #[arg(long, default_value = "intent-server")]
    issuer: String,
    #[arg(long, default_value = "tool-gateway")]
    audience: String,
}

fn parse_category(s: &str) -> Result<IntentCategory, String> {
    serde_json::from_value(serde_json::Value::String(s.to_owned()))
        .map_err(|_| format!("unknown category {s:?}"))
}

#[cfg(feature = "llm")]
fn llm_fallback() -> Result<Arc<dyn IntentClassifier>> {
    let config = LlmConfig::from_env().context("configuring LLM fallback")?;
    let classifier = LlmClassifier::new(config).context("configuring LLM fallback")?;
    Ok(Arc::new(classifier))
}

#[cfg(not(feature = "llm"))]
fn llm_fallback() -> Result<Arc<dyn IntentClassifier>> {
    anyhow::bail!("this build was compiled without the llm feature")
}

fn read_json<T: serde::de::DeserializeOwned>(path: &PathBuf) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    match cli.command {
        Command::Evaluate {
            key,
            policy,
            rules,
            request,
            threshold,
            llm,
            ids,
        } => {
            let key = KeyPair::load(&key)?;
            let issuer = JwtIssuer::new(IssuerConfig::new(ids.issuer, ids.audience), &key)?;
            let policy = Policy::from_path(&policy)
                .with_context(|| format!("loading {}", policy.display()))?;
            let rules = RuleClassifier::from_path(&rules)
                .with_context(|| format!("loading {}", rules.display()))?;
            let fallback = if llm { Some(llm_fallback()?) } else { None };
            let classifier = HybridClassifier::new(rules, fallback).with_threshold(threshold);
            let evaluator =
                Evaluator::new(Arc::new(classifier), Arc::new(policy), Arc::new(issuer));
            let request: IntentRequest = read_json(&request)?;

            let evaluation = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(evaluator.evaluate(&request))?;
            println!("{}", serde_json::to_string_pretty(&evaluation)?);
            return Ok(match evaluation.decision {
                Decision::Allow { .. } => ExitCode::SUCCESS,
                Decision::Deny { .. } => ExitCode::from(2),
                Decision::StepUp { .. } => ExitCode::from(3),
            });
        }
        Command::Lint { policy, rules } => {
            anyhow::ensure!(
                policy.is_some() || rules.is_some(),
                "pass --policy and/or --rules"
            );
            if let Some(p) = policy {
                let parsed = Policy::from_path(&p).with_context(|| format!("{}", p.display()))?;
                eprintln!("{}: ok ({} rules)", p.display(), parsed.rules.len());
            }
            if let Some(r) = rules {
                RuleClassifier::from_path(&r).with_context(|| format!("{}", r.display()))?;
                eprintln!("{}: ok", r.display());
            }
        }
        Command::Keygen { out } => {
            let key = KeyPair::generate()?;
            key.save(&out)?;
            eprintln!("wrote {} (kid {})", out.display(), key.kid());
        }
        Command::Jwks { key } => {
            let key = KeyPair::load(&key)?;
            let jwks = Jwks {
                keys: vec![key.public_jwk()],
            };
            println!("{}", serde_json::to_string_pretty(&jwks)?);
        }
        Command::Issue {
            key,
            request,
            category,
            action,
            resource,
            scopes,
            ttl,
            ids,
        } => {
            let key = KeyPair::load(&key)?;
            let issuer = JwtIssuer::new(IssuerConfig::new(ids.issuer, ids.audience), &key)?;
            let request: IntentRequest = read_json(&request)?;
            let intent = Intent {
                category,
                action,
                resource,
                confidence: 1.0,
                source: ClassifierSource::Rules,
            };
            let decision = Decision::Allow {
                scopes,
                ttl: Duration::from_secs(ttl),
            };
            let issued = issuer.issue(&request, &intent, &decision)?;
            println!("{}", serde_json::to_string_pretty(&issued)?);
        }
        Command::Verify {
            jwks,
            token,
            tool_calls,
            scope,
            ids,
        } => {
            let jwks: Jwks = read_json(&jwks)?;
            let value: serde_json::Value = read_json(&tool_calls)?;
            let calls: Vec<ToolCall> = if value.is_array() {
                serde_json::from_value(value)?
            } else {
                serde_json::from_value::<IntentRequest>(value)?.tool_calls
            };
            let verifier = TokenVerifier::from_jwks(&jwks, &ids.issuer, &ids.audience)?;
            let claims = verifier.verify(token.trim(), &calls, scope.as_deref())?;
            println!("{}", serde_json::to_string_pretty(&claims)?);
        }
    }
    Ok(ExitCode::SUCCESS)
}
