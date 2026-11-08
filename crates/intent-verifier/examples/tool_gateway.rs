//! Self-contained demo: an intent-server, a tool gateway protected by
//! `IntentLayer`, and a client walking through allowed and rejected calls.
//!
//! Run with `cargo run -p intent-verifier --example tool_gateway`. Exits
//! non-zero if any outcome differs from what is expected.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use axum::routing::post;
use axum::{Json, Router};
use intent_server::{ApiKeys, AppState, ServerConfig, router};
use intent_token::KeyPair;
use intent_verifier::{
    InMemoryReplayStore, IntentLayer, RemoteJwksOptions, VerifiedIntent, VerifiedToolCalls,
    Verifier,
};
use serde_json::{Value, json};
use tokio::net::TcpListener;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn short(token: &str) -> String {
    format!("{}…", token.chars().take(12).collect::<String>())
}

/// Fake tool: never touches the filesystem.
async fn read_file(
    VerifiedIntent(claims): VerifiedIntent,
    VerifiedToolCalls(calls): VerifiedToolCalls,
) -> Json<Value> {
    // Execute only the calls the token was verified against.
    let name = calls[0].arguments["path"].as_str().unwrap_or("?");
    Json(json!({
        "contents": format!("<fake contents of {name}>"),
        "authorized_for": claims.sub,
    }))
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<bool, Box<dyn std::error::Error>> {
    // 1. Issuing service on 127.0.0.1:0. The key file lives in a temp dir that
    //    is removed as soon as the server has loaded it.
    let root = repo_root();
    let dir = std::env::temp_dir().join(format!("intent-gateway-demo-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let key_path = dir.join("signing.key.json");
    KeyPair::generate()?.save(&key_path)?;
    let config = ServerConfig::from_toml_str(&format!(
        "key_path = {:?}\npolicy_path = {:?}\nrules_path = {:?}\nllm = false",
        key_path,
        root.join("config/policy.example.toml"),
        root.join("config/rules.example.toml"),
    ))?;
    // Random-looking API key generated at runtime (a fresh key's public part).
    let api_key = KeyPair::generate()?.public_x();
    let state = AppState::from_config(&config, ApiKeys::from_keys([api_key.as_str()]), None)?;
    std::fs::remove_dir_all(&dir).ok();

    let server = TcpListener::bind("127.0.0.1:0").await?;
    let server_addr = server.local_addr()?;
    let server_app = router(state);
    tokio::spawn(async move { axum::serve(server, server_app).await });
    println!("intent-server on http://{server_addr}");

    // 2. Gateway protected by IntentLayer, keys fetched from the server's JWKS.
    let verifier = Verifier::from_jwks_url(
        &format!("http://{server_addr}/.well-known/jwks.json"),
        &config.issuer,
        &config.audience,
        RemoteJwksOptions::default(),
    )
    .await?
    // Tokens are single-use at this gateway.
    .with_replay_store(Arc::new(InMemoryReplayStore::default()));
    let gateway_app = Router::new().route(
        "/tools/read_file",
        post(read_file).layer(IntentLayer::new(verifier).require_scope("data:read")),
    );
    let gateway = TcpListener::bind("127.0.0.1:0").await?;
    let gateway_addr = gateway.local_addr()?;
    tokio::spawn(async move { axum::serve(gateway, gateway_app).await });
    println!("tool gateway on http://{gateway_addr}\n");

    // 3. Client flow.
    let client = reqwest::Client::new();
    let scenario = |name: &str| -> Result<Value, Box<dyn std::error::Error>> {
        Ok(serde_json::from_str(&std::fs::read_to_string(
            root.join(format!("examples/scenarios/{name}.json")),
        )?)?)
    };
    let evaluate = |body: Value| {
        client
            .post(format!("http://{server_addr}/v1/evaluate"))
            .bearer_auth(&api_key)
            .json(&body)
            .send()
    };
    let gateway_url = format!("http://{gateway_addr}/tools/read_file");
    let mut ok = true;
    let mut check = |label: &str, got: String, want: &str| {
        let pass = got == want;
        ok &= pass;
        println!("[{}] {label}: {got}", if pass { "ok" } else { "FAIL" });
    };

    let allow = scenario("allow_read")?;
    let res = evaluate(allow.clone()).await?;
    let status = res.status();
    let body: Value = res.json().await?;
    check("evaluate allow_read", status.to_string(), "200 OK");
    let token = body["token"]["token"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    println!("     token {}", short(&token));
    let call = allow["tool_calls"][0].clone();

    let res = client
        .post(&gateway_url)
        .bearer_auth(&token)
        .json(&call)
        .send()
        .await?;
    let status = res.status();
    let body: Value = res.json().await?;
    check("gateway, matching call", status.to_string(), "200 OK");
    println!("     {body}");

    let tampered = json!({"name": "read_file", "arguments": {"path": "/etc/shadow"}});
    let res = client
        .post(&gateway_url)
        .bearer_auth(&token)
        .json(&tampered)
        .send()
        .await?;
    let status = res.status();
    let body: Value = res.json().await?;
    check(
        "gateway, tampered args",
        format!("{status} {}", body["error"].as_str().unwrap_or("")),
        "401 Unauthorized request_mismatch",
    );

    let res = client
        .post(&gateway_url)
        .bearer_auth(&token)
        .json(&call)
        .send()
        .await?;
    let status = res.status();
    let body: Value = res.json().await?;
    check(
        "gateway, same token re-sent",
        format!("{status} {}", body["error"].as_str().unwrap_or("")),
        "401 Unauthorized replayed",
    );

    let res = client.post(&gateway_url).json(&call).send().await?;
    check(
        "gateway, no token",
        res.status().to_string(),
        "401 Unauthorized",
    );

    let res = evaluate(scenario("deny_delete")?).await?;
    let status = res.status();
    let body: Value = res.json().await?;
    check("evaluate deny_delete", status.to_string(), "403 Forbidden");
    check(
        "deny_delete issues no token",
        body["token"].is_null().to_string(),
        "true",
    );

    println!(
        "\n{}",
        if ok {
            "all outcomes as expected"
        } else {
            "UNEXPECTED OUTCOME"
        }
    );
    Ok(ok)
}
