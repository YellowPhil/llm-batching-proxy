use std::{sync::Arc, time::Duration};

use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use clap::Parser;
use serde::{Deserialize, Serialize};

mod batching;
mod config;
mod errors;

use config::Config;

use crate::{batching::BatchProcessor, errors::ProxyError};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    #[arg(short, long, default_value = "3000")]
    port: u16,

    #[arg(long, default_value = "http://127.0.0.1:8080")]
    inference_url: String,

    #[arg(long)]
    debug: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbedRequest {
    inputs: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SingleRequest {
    input: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SingleResponse {
    embedding: Vec<f32>,
}

#[derive(Clone)]
struct AppState {
    batch_processor: Arc<BatchProcessor>,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let args = Args::parse();
    let config = Config::load(&args.config, args.debug)?;

    tracing::info!("Starting Server on port {}", args.port);
    tracing::info!("Configuration: {:?}", config);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .pool_max_idle_per_host(10)
        .build()?;

    let batch_processor = Arc::new(
        BatchProcessor::new(args.inference_url.clone(), config.clone(), Arc::new(client)).await?,
    );

    let app_state = AppState {
        batch_processor,
        // start_time: tokio::time::Instant::now(),
        // config: Arc::new(config),
    };

    let app = Router::new()
        .route("/embed", post(embed_single))
        .with_state(app_state);
    // .layer(CorsLayer::permissive())
    // .layer(TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", args.port)).await?;

    tracing::info!("Server running on http://0.0.0.0:{}", args.port);
    tracing::info!("Ready to accept requests!");

    axum::serve(listener, app).await?;
    Ok(())
}

#[tracing::instrument(skip(state))]
async fn embed_single(
    State(state): State<AppState>,
    Json(payload): Json<SingleRequest>,
) -> Result<Json<SingleResponse>, ProxyError> {
    tracing::debug!("Received single embed request for: {}", payload.input);

    let start = tokio::time::Instant::now();

    match state.batch_processor.process_single(payload.input).await {
        Ok(embedding) => {
            let duration = start.elapsed();
            tracing::debug!("Single request processed in {:?}", duration);
            Ok(Json(SingleResponse { embedding }))
        }
        Err(e) => {
            tracing::error!("Failed to process single request: {}", e);
            Err(e)
        }
    }
}
