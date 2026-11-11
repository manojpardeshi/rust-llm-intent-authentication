# syntax=docker/dockerfile:1
#
# intent-server container image.
#
#   docker build -t intent-server .
#   docker run --rm -p 8080:8080 \
#     -e INTENT_API_KEYS=change-me \
#     -v "$PWD/deploy:/etc/intent:ro" \
#     intent-server
#
# /etc/intent must contain server.toml, the signing key, policy and rules,
# with paths in server.toml pointing inside /etc/intent and
# bind = "0.0.0.0:8080". The image runs as a non-root user and handles
# SIGTERM by draining in-flight requests.

FROM rust:1.99-slim-bookworm AS build
WORKDIR /src
COPY . .
# Use the image's own toolchain rather than the channel in rust-toolchain.toml.
RUN RUSTUP_TOOLCHAIN="$(rustup default | cut -d' ' -f1)" \
    cargo build --release --locked -p intent-server \
 && cp target/release/intent-server /intent-server

FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=build /intent-server /usr/local/bin/intent-server
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/usr/local/bin/intent-server"]
CMD ["--config", "/etc/intent/server.toml", "--log-json"]
