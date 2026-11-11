# rust-llm-intent-authentication

An intent-gated token framework for LLM agents, written in Rust.

Before an agent's tool call runs, this framework **classifies what the call is
trying to do**, checks that intent against a policy, and, only if the policy
allows it, issues a **short-lived EdDSA JWT bound to that exact tool call**.
The tool or API gateway verifies the token before executing. A prompt-injected
or misaligned request can't get a token for an action the policy forbids, and a
token issued for one call can't be reused for a different one.

```
LLM request ──► /v1/evaluate ──► rule classifier ──(confident?)──► policy ──► signed JWT
                                       │ no                          │ deny / step-up
                                       ▼                             ▼
                              LLM classifier (optional)        403 + reason + audit log

Tool gateway ──► intent-verifier (JWKS) ──► signature, scope, req_hash, replay ──► execute / reject
```

## How it works

1. **Classify.** Rules (tool-name matches, argument regexes, known
   prompt-injection phrases) assign an intent category and a confidence. If the
   confidence is below a threshold, Claude can optionally be asked as a
   fallback. The LLM can only make a result stricter: when the two disagree
   the more restrictive category wins, agreement keeps the rule's confidence,
   and an intent that only the LLM produced is labelled `llm` so policy can
   refuse it. If the LLM call fails or runs out of time, the request is denied
   (fail closed).
2. **Decide.** A TOML policy maps the intent to `allow` (with scopes and a TTL),
   `deny`, or `step_up` (needs a human). The first matching rule wins, the
   default is deny, and `malicious` intents are always denied. A request with
   several tool calls is classified and decided **per call**: any deny wins,
   then any step-up, otherwise the token gets the union of scopes and the
   shortest TTL.
3. **Issue.** On allow, the server signs an Ed25519 JWT whose `req_hash` claim
   is the SHA-256 of the RFC 8785 canonical JSON of the tool calls. Lifetime is
   capped at 300 s. Integers beyond ±(2^53 − 1) are rejected, because JSON
   canonicalisation would round them and two ids could share a hash; pass
   large ids as strings.
4. **Verify.** The gateway checks the signature (via JWKS), `iss`, `aud`,
   expiry, the required scope, and recomputes `req_hash` from the call it is
   about to run. Tokens can optionally be single-use (`jti` replay store).

Intent categories, least to most restrictive: `read_data`, `write_data`,
`send_message`, `execute_code`, `admin_config`, `financial`, `unknown`,
`malicious`. This is how they appear in policy and rules files, API
responses and tokens. The older PascalCase spellings (`ReadData`) are still
accepted in policy and rules files.

## What a token does and does not guarantee

A valid token means: *this server's policy allowed these exact tool calls, for
this subject, within the last few minutes.* Specifically:

- **Guaranteed:** the tool calls (names and arguments) are byte-for-byte the
  ones that were evaluated; the token was signed by a key in the JWKS; it has
  not expired; it carries the scopes policy granted; with a replay store, it
  has not been used before.
- **Not guaranteed:** that the classification was right. Rules and the LLM
  are heuristics, and policy can only be as good as the intent it is given.
  Write rules as allow-lists (see the `run_sql` rules in
  `config/rules.example.toml`), keep `min_confidence` meaningful, and use
  `sources = ["rules", "hybrid"]` on sensitive allows.
- **Not guaranteed:** who the end user is. `subject` comes from the caller.
  Bind API keys to the subjects they may request with `subject_bindings` in
  the server config.
- **Out of scope:** what the tool does with valid arguments, and anything the
  agent does without calling a gated tool.

## Threat model

The framework assumes an attacker who can influence the LLM's input (prompt
injection through documents, web pages, tool output) and therefore the tool
calls it emits, and possibly holds a valid API key for the server. It aims to
stop that attacker from:

- running a tool call that policy forbids, including by batching it with
  allowed calls or disguising it (for example SQL writes inside a `SELECT`);
- reusing a token for different arguments, a different tool, or (with a
  replay store) a second time;
- talking the LLM fallback into granting more than rules would;
- requesting tokens for subjects their key is not bound to.

It does not defend against a compromised server or signing key, a gateway
that skips verification, or tools that are themselves unsafe for arguments
policy allows.

## Crates

| Crate | Purpose |
|---|---|
| [`intent-core`](crates/intent-core) | Core types (`IntentRequest`, `Intent`, `Decision`), traits, and `request_digest` |
| [`intent-classify`](crates/intent-classify) | Rule, LLM (feature `llm`) and hybrid classifiers |
| [`intent-policy`](crates/intent-policy) | TOML policy loading and evaluation |
| [`intent-token`](crates/intent-token) | Ed25519 keys, JWT issue/verify, JWKS, key rotation |
| [`intent-verifier`](crates/intent-verifier) | Gateway-side verifier: static or remote JWKS, axum/tower `IntentLayer`, replay store |
| [`intent-server`](crates/intent-server) | axum HTTP service: evaluate, verify, JWKS, health, metrics |
| [`intent-cli`](crates/intent-cli) | The `intent` CLI: keygen, jwks, evaluate, verify, lint |

## Quickstart

Requires a recent stable Rust toolchain (edition 2024).

```sh
cargo build --workspace
cargo test --workspace
```

### CLI

```sh
# Generate a signing key (*.key.json is git-ignored) and publish its public half
cargo run -p intent-cli -- keygen --out signing.key.json
cargo run -p intent-cli -- jwks --key signing.key.json > jwks.json

# Check the example policy and rules
cargo run -p intent-cli -- lint --policy config/policy.example.toml --rules config/rules.example.toml

# Evaluate a request: exit code 0 = allow, 2 = deny, 3 = step-up
cargo run -p intent-cli -- evaluate --key signing.key.json --request examples/request.json > eval.json

# Verify the token against the tool calls about to run
TOKEN=$(jq -r .token.token eval.json)
cargo run -p intent-cli -- verify --jwks jwks.json --token "$TOKEN" \
  --tool-calls examples/request.json --scope data:read

# The same token against different arguments is rejected
cargo run -p intent-cli -- verify --jwks jwks.json --token "$TOKEN" \
  --tool-calls examples/tampered_calls.json
# error: tool calls do not match the request the token was issued for
```

More requests to try are in [`examples/scenarios/`](examples/scenarios):
an allowed read, a denied delete, step-ups for SQL writes and fund transfers,
a prompt-injection attempt, and an unknown tool.

### HTTP server

```sh
cargo run -p intent-cli -- keygen --out signing.key.json   # if not done above
INTENT_API_KEYS="dev-key-123" cargo run -p intent-server -- --config config/server.example.toml
```

```sh
curl -s localhost:8080/v1/evaluate \
  -H 'Authorization: Bearer dev-key-123' -H 'content-type: application/json' \
  --data @examples/request.json
```

| Endpoint | Auth | Description |
|---|---|---|
| `POST /v1/evaluate` | API key | Body: an `IntentRequest`. Returns `decision`, `intent`, and on allow `scopes`, `ttl`, `token` |
| `POST /v1/verify` | API key | Body: `{"token", "tool_calls", "scope"?}`. For gateways that aren't written in Rust |
| `GET /.well-known/jwks.json` | none | Public signing keys (active and retired) |
| `GET /healthz` | none | Liveness |
| `GET /metrics` | none | Prometheus metrics |

API keys are passed as `Authorization: Bearer <key>` or `x-api-key: <key>` and
are configured only via the `INTENT_API_KEYS` environment variable
(comma-separated; only their SHA-256 digests are kept in memory).

An `IntentRequest` looks like:

```json
{
  "subject": "user-42",
  "session_id": "sess-1",
  "messages": [{"role": "user", "content": "Summarise /srv/report.txt"}],
  "tool_calls": [{"name": "read_file", "arguments": {"path": "/srv/report.txt"}}]
}
```

### Protecting a gateway

```sh
cargo run -p intent-verifier --example tool_gateway
```

[The example](crates/intent-verifier/examples/tool_gateway.rs) runs the server
and a tool gateway protected by `IntentLayer` in one process. It shows an
allowed call, a tampered call (401), a replayed token (401), a missing token
(401), and a denied intent (403).

## Configuration

- [`config/server.example.toml`](config/server.example.toml): bind address,
  key and policy paths, issuer/audience, LLM fallback, body limit, hot reload,
  key rotation (with a step-by-step procedure), and rate limiting.
- [`config/policy.example.toml`](config/policy.example.toml): policy rules.
- [`config/rules.example.toml`](config/rules.example.toml): classifier rules.

The LLM fallback is off by default. Enable it with `llm = true` in the server
config or `--llm` on the CLI, and set `ANTHROPIC_API_KEY`. Build without it
using `--no-default-features`.

Policy, rules, and signing keys hot-reload on `SIGHUP`, and policy and rules
also reload on file change when `watch = true`. A file that fails to load
leaves the previous configuration serving.

## Security notes

- Signing keys are never logged. Keep `*.key.json` out of version control.
- Audit events (subject, intent, decision, `jti`, `req_hash`) go to the
  `intent::audit` tracing target. Raw prompts are not logged.
- `/v1/verify` is stateless unless `verify_single_use = true`. Both that and
  intent-verifier's `InMemoryReplayStore` are per process: run a single
  instance, or implement the `ReplayStore` trait over shared storage, if
  tokens must be single-use across instances.
- `/metrics` is unauthenticated unless `metrics_require_auth = true`.
- The LLM classifier has been tested against a mocked API only.

## Docker

```sh
docker build -t intent-server .
docker run --rm -p 8080:8080 -e INTENT_API_KEYS=change-me \
  -v "$PWD/deploy:/etc/intent:ro" intent-server
```

`deploy/` must hold `server.toml` (with `bind = "0.0.0.0:8080"` and paths under
`/etc/intent`), the signing key, policy and rules. The image runs as non-root
and drains in-flight requests on SIGTERM.

## Development

Requires Rust 1.89 or newer (declared as `rust-version`).

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```

Property tests use `proptest`. Fuzz targets live in a separate workspace under
[`fuzz/`](fuzz); see its [README](fuzz/README.md) (requires nightly and
`cargo-fuzz`).

## License

MIT. See [LICENSE](LICENSE).
