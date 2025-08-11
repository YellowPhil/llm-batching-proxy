use clap::Parser;
use eyre::Context;
use serde::{Deserialize, Serialize};

mod config;
mod errors;

use config::Config;

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

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let args = Args::parse();
    let config = Config::load(&args.config, args.debug).wrap_err("Failed to load config from file")?;

    tracing::info!("Config loaded: {}", config);
    tracing::info!("Starting server on port {}", args.port);

    Ok(())
}
