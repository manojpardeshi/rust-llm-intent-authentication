# Fuzzing

cargo-fuzz (libFuzzer) targets for the intent framework. This directory is a
separate Cargo workspace, so `cargo test --workspace` at the repository root
does not build it.

## Setup

```sh
rustup toolchain install nightly --profile minimal
cargo install cargo-fuzz --locked
```

## Running

Run from this directory (`fuzz/`):

```sh
cargo +nightly fuzz run digest -- -max_total_time=30
cargo +nightly fuzz run request_json -- -max_total_time=30
cargo +nightly fuzz run policy_toml -- -max_total_time=30
cargo +nightly fuzz run rules_classify -- -max_total_time=30
cargo +nightly fuzz run token_verify -- -max_total_time=30
```

List targets with `cargo fuzz list`. Crashing inputs are written to
`artifacts/<target>/`; reproduce one with
`cargo +nightly fuzz run <target> artifacts/<target>/<file>`.

## Targets

| Target           | Input                          | Checks                                                                     |
|------------------|--------------------------------|----------------------------------------------------------------------------|
| `digest`         | bytes as JSON `Vec<ToolCall>`  | no panic; digest is unchanged by a serialize/parse round trip              |
| `request_json`   | bytes as JSON `IntentRequest`  | no panic in `request_digest`                                               |
| `policy_toml`    | UTF-8 as policy TOML           | no panic in `decide` on a fixed intent set; `Malicious` is never allowed   |
| `rules_classify` | structured (`arbitrary`)       | no panic; confidence in `[0, 1]`, using `config/rules.example.toml`        |
| `token_verify`   | UTF-8 as a bearer token        | no panic; a token without the signing key is never accepted                |

Corpus and artifact directories are git-ignored.
