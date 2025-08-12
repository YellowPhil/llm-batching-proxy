use std::{sync::Arc, time::Duration};

use axum::{
    Router,
    routing::{get, post},
};
use clap::Parser;

mod batching;
mod config;
mod controller;
mod errors;

use config::Config;
//use controller::{AppState, embed_single, embed_batch, health_check};
use batching::BatchProcessor;

use crate::controller::{AppState, embed_single, health_check};

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

    let app_state = AppState::new(batch_processor);

    let app = Router::new()
        .route("/embed", post(embed_single))
        .route("/health", get(health_check))
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", args.port)).await?;

    tracing::info!("Server running on http://0.0.0.0:{}", args.port);
    tracing::info!("Ready to accept requests!");

    axum::serve(listener, app).await?;
    Ok(())
}
