FROM rust:1.89.0 as base

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN cargo install sccache
RUN cargo install cargo-chef

ENV RUSTC_WRAPPER=sccache SCCACHE_DIR=/sccache

FROM base AS planner


WORKDIR /app

COPY . .

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo chef prepare --recipe-path recipe.json


FROM base as builder

WORKDIR /app

COPY --from=planner /app/recipe.json recipe.json

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo chef cook --release --recipe-path recipe.json

COPY . .

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo build -r --bin rust-service

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

COPY --from=builder /app/target/release/rust-service /app/rust-service

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 3000

ENTRYPOINT ["/app/rust-service"]
CMD ["--config", "/app/config.toml", "--port", "3000", "--inference-url", "http://text-embeddings-inference:80/embed"]
