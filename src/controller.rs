use axum::{Json, extract::State};
use serde::{Deserialize, Serialize};
use tracing;

use crate::{batching::BatchProcessor, errors::ProxyError};

#[async_trait::async_trait]
pub trait BatchProcessorTrait: Send + Sync {
    async fn process_single(&self, input: String) -> Result<Vec<f32>, ProxyError>;
}

#[async_trait::async_trait]
impl BatchProcessorTrait for BatchProcessor {
    async fn process_single(&self, input: String) -> Result<Vec<f32>, ProxyError> {
        self.process_single(input).await
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub inputs: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SingleRequest {
    pub input: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SingleResponse {
    pub embedding: Vec<f32>,
}

#[derive(Clone)]
pub struct AppState {
    pub batch_processor: std::sync::Arc<dyn BatchProcessorTrait>,
}

impl AppState {
    pub fn new(batch_processor: std::sync::Arc<dyn BatchProcessorTrait>) -> Self {
        Self { batch_processor }
    }
}

#[tracing::instrument(skip(state))]
pub async fn embed_single(
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

pub async fn health_check() -> Result<Json<serde_json::Value>, ProxyError> {
    let status = serde_json::json!({
        "status": "healthy",
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        "service": "embedding-proxy"
    });

    Ok(Json(status))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct MockBatchProcessor;

    #[async_trait::async_trait]
    impl BatchProcessorTrait for MockBatchProcessor {
        async fn process_single(
            &self,
            _input: String,
        ) -> Result<Vec<f32>, crate::errors::ProxyError> {
            Ok(vec![0.1, 0.2, 0.3])
        }
    }

    #[tokio::test]
    async fn test_embed_single() {
        let mock_processor = Arc::new(MockBatchProcessor);
        let app_state = AppState::new(mock_processor);

        let request = SingleRequest {
            input: "test input".to_string(),
        };

        let result = embed_single(axum::extract::State(app_state), axum::Json(request)).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.0.embedding, vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn test_health_check() {
        let result = health_check().await;

        assert!(result.is_ok());
        let response = result.unwrap();
        let status = response.0;

        assert_eq!(status["status"], "healthy");
        assert_eq!(status["service"], "embedding-proxy");
        assert!(status["timestamp"].is_number());
    }
}
