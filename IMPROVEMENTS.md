# Review findings and proposed fixes

Review date: 2026-11-10, against commit `8fd5195` (`main`).

State at review time: `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo fmt --all -- --check` and all 148 tests pass. `cargo audit` was not run
(not installed), so dependencies are unchecked for advisories.

Status (2026-11-10): everything is fixed except the enum casing decision.
Each item notes what changed. After the fixes: fmt and clippy (all features
and no default features) clean, 164 tests pass, every fuzz target ran 20 s
with no crashes.

## 1. Confirmed security issues

Each of these was reproduced with `intent evaluate` against
`config/policy.example.toml` and `config/rules.example.toml`.

### 1.1 Batching tool calls bypasses policy rules

- [x] Fixed. `Evaluator::evaluate` (`intent-core`) classifies and decides each call separately and combines the results with `combine_decisions`. Tests: `batching_cannot_bypass_per_call_policy` (server), `combine_decisions_strictest_wins`, proptest `adding_a_call_never_relaxes_the_decision`. Note: with the LLM fallback on, a batch makes up to one LLM call per tool call.

**Where:** `crates/intent-classify/src/rules.rs` (`classify_sync`, the loop
over `request.tool_calls`), `crates/intent-policy/src/lib.rs` (`decide`).

**Problem:** For a multi-call request the classifier keeps only the strictest
call's category and action. Policy sees that single intent, so rules that
match on the action of any other call never fire. The issued token's
`req_hash` still covers every call.

**Reproduction:**

| Request | Result |
| --- | --- |
| `delete_records` alone | deny ("deletes are never automated") |
| `delete_records` + `send_email` | allow, scope `messages:send` |
| `run_sql "drop table users"` alone | step-up |
| `run_sql "drop table users"` + `send_email` | allow, scope `messages:send` |

**Fix:** Classify and evaluate policy per tool call, then combine: any deny →
deny, else any step-up → step-up, else allow with the union of scopes and the
shortest TTL. Keep the injection check request-wide.

**Tests to add:** the four rows above as policy/classifier tests, plus a
property test that adding a call to a request never makes the decision less
restrictive.

### 1.2 SQL writes classify as reads

- [x] Fixed. The example rules allow-list a single plain `SELECT` (new `args_not_match` rejects `;`, comments, `INTO`, locking reads and a few dangerous functions); every other `run_sql` is `sql_write`. An inspected argument that is present but not a string makes the call `Unknown`. Tests in `rules.rs`.

**Where:** `config/rules.example.toml` (`run_sql` rules),
`crates/intent-classify/src/rules.rs` (`CompiledToolRule::matches`).

**Problem:** The write rule is a prefix regex
(`^\s*(insert|update|delete|drop|...)`). Anything that does not match falls
through to the `sql_read` rule at confidence 0.7, which exactly meets the
read rule's `min_confidence = 0.7`. `args_match` also only matches string
arguments, so a non-string `query` skips the write rule as well.

**Reproduction:** all of these get a `data:read` token:

- `WITH x AS (SELECT 1) DELETE FROM users`
- `select 1; drop table users`
- `/**/drop table users`

**Fix:**

- Invert the example rules: only a positive match on a single `SELECT`
  statement (no `;`, no comments) is `sql_read`; every other `run_sql` call
  is `sql_write` or `Unknown`.
- Lower the fallthrough confidence below the read rule's `min_confidence`,
  so an unconfirmed read is denied without the LLM.
- When an `args_match` argument is present but not a string, treat the call
  as `Unknown` rather than trying the next rule.

**Tests to add:** the three queries above, plus a non-string `query`.

### 1.3 Request digest collides on large integers

- [x] Fixed. `tool_calls_digest` rejects integers outside ±(2^53 − 1) (`MAX_SAFE_INTEGER`); `/v1/evaluate` returns 400 `invalid_request`. Non-integer floats stay allowed. Unit test, proptest `digest_integer_range`, and the `digest` fuzz target covers the full i64/u64 range.

**Where:** `crates/intent-core/src/lib.rs` (`tool_calls_digest`).

**Problem:** RFC 8785 canonicalisation serialises numbers as IEEE-754
doubles, so integers beyond ±2^53 lose precision before hashing.
`"id": 9007199254740993` and `"id": 9007199254740992` produce the same
`req_hash`, so a token issued for one record id verifies for the other.

**Fix:** Reject tool-call arguments containing integers outside ±(2^53 − 1)
with `Error::Canonicalize`, in `tool_calls_digest` itself so both issuing and
verifying fail closed. Callers that need large ids pass them as strings.
Decide separately whether non-integer floats should be allowed.

**Tests to add:** unit test for the pair above; extend the digest property
tests and the `digest` fuzz target to cover the full `i64`/`u64` range.

## 2. Design gaps

### 2.1 Subject is caller-asserted

- [x] Fixed. `subject_bindings` (hex SHA-256 of key → subject patterns) and `require_subject_binding` in server config; mismatches get 403 `subject_not_allowed`. Unknown digests fail at startup.

Any API key holder can put any `subject` in the request body, yet policy
rules match on subject (for example `svc-*`). Bind each API key to the
subject patterns it may request (for example a keyed table in server config)
and reject mismatches with 403 in `crates/intent-server/src/lib.rs`
(`evaluate`).

### 2.2 LLM fallback can raise trust

- [x] Fixed. Agreement keeps the rule confidence; an LLM-only intent is labelled `llm`; policy rules accept an optional `sources` list, used on the example WriteData and SendMessage allows.

In `crates/intent-classify/src/hybrid.rs`, an `Unknown` rule result takes the
LLM's answer wholesale, and when categories agree the higher of the two
confidences wins. The LLM reads attacker-influenced text, so it should only
be able to make a result stricter. Options:

- Add an optional `sources = ["rules"]` matcher to policy rules so sensitive
  allows can require a rule-derived intent.
- On agreement, keep the rule confidence unless explicitly configured
  otherwise.

### 2.3 `/v1/verify` is not single-use

- [x] Fixed. `verify_single_use` (and `verify_replay_capacity`) in server config. The replay store moved to `intent-token` and is re-exported by `intent-verifier`.

Replay protection lives only in `intent-verifier`. Gateways that call the
server's HTTP endpoint get no `jti` consumption. Either add an optional
replay store to the server's verify handler or document clearly that the
endpoint is stateless.

## 3. Operational

- [x] **SIGTERM:** done (`shutdown_signal` in `main.rs`, checked by hand). `crates/intent-server/src/main.rs` shuts down gracefully
  on Ctrl-C only. Also wait on `SignalKind::terminate()` so containers drain
  in-flight requests.
- [x] **Timeout mismatch:** `/v1/evaluate` now has a 25 s budget (`EVALUATE_BUDGET`, below `REQUEST_TIMEOUT`); running out gives a fail-closed deny. Test `slow_classification_fails_closed_within_budget`. the LLM retry budget (3 × 10s plus backoff) can
  exceed `REQUEST_TIMEOUT` (30s), giving a 408 instead of a fail-closed
  deny. Cap the total classify time below the request timeout.
- [x] **Key hygiene:** `zeroize` feature on, seed/DER/key-file buffers wrapped in `Zeroizing`, `sync_all` in `save`. zeroise the seed and encoded `d` in
  `crates/intent-token/src/keys.rs` (enable `ed25519-dalek`'s `zeroize`
  feature) and `sync_all` in `KeyPair::save`.
- [x] **Leeway constant:** `intent_token::LEEWAY_SECS`, used by the verifier and the replay store. the 5s leeway is hard-coded in both
  `intent-token` (`validation.leeway`) and
  `crates/intent-verifier/src/replay.rs` (`LEEWAY`). Export one constant
  from `intent-token`.
- [ ] **Enum casing:** not changed; needs a decision (breaking wire-format change). `IntentCategory` serialises as PascalCase while every
  other enum is snake_case. Settle the wire format (and consider
  `#[non_exhaustive]`) before 1.0.
- [x] **`/metrics` is unauthenticated:** documented, plus optional `metrics_require_auth`. acceptable behind a private
  network; document it or put it behind the API key check.

## 4. Repo hygiene

- [x] Root `README.md`: what the protocol is, threat model, flow diagram,
  quickstart, and what the token does and does not guarantee.
- [x] `LICENSE` file: MIT, and `Cargo.toml` changed to match.
- [x] CI workflow: fmt, clippy, test, `cargo audit` or `cargo deny`, and a
  short fuzz smoke run.
- [x] `rust-toolchain.toml` and a declared MSRV (`rust-version = "1.89"`, the highest any dependency declares; verified only by the CI `msrv` job).
- [x] Dockerfile for `intent-server` (not built locally; built by the CI `docker` job).

## Suggested order

1. 1.1 batching bypass
2. 1.3 digest collision
3. 1.2 SQL rules
4. 2.1 subject binding, then 2.2 and 2.3
5. SIGTERM, README, LICENSE, CI
6. Remaining operational items
