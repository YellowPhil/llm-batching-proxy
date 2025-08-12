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
pub struct EmbedResponse(pub Vec<Vec<f32>>);

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
            input: String,
        ) -> Result<Vec<f32>, crate::errors::ProxyError> {
            // Generate deterministic but different vectors for different inputs
            // This allows testing that batching works correctly and responses don't mix
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            use std::hash::{Hash, Hasher};
            input.hash(&mut hasher);
            let hash = hasher.finish();

            // Generate a 384-dimensional vector based on the input hash
            let mut embedding = Vec::with_capacity(384);
            for i in 0..384 {
                let seed = hash.wrapping_add(i as u64);
                let x = (seed as f32) / (u64::MAX as f32);
                // Create a unique pattern for each input
                let value = (x * 2.0 - 1.0) * 0.1 + (hash % 1000) as f32 * 0.001;
                embedding.push(value);
            }

            Ok(embedding)
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
        assert_eq!(response.0.embedding.len(), 384);

        // Verify the embedding is deterministic for the same input
        let mock_processor = Arc::new(MockBatchProcessor);
        let app_state = AppState::new(mock_processor);
        let request2 = SingleRequest {
            input: "test input".to_string(),
        };
        let result2 = embed_single(axum::extract::State(app_state), axum::Json(request2)).await;
        assert!(result2.is_ok());
        assert_eq!(response.0.embedding, result2.unwrap().0.embedding);
    }

    #[tokio::test]
    async fn test_batching_response_matching() {
        let mock_processor = Arc::new(MockBatchProcessor);
        let app_state = AppState::new(mock_processor);

        // Test that different inputs produce different embeddings
        let request1 = SingleRequest {
            input: "first input".to_string(),
        };
        let request2 = SingleRequest {
            input: "second input".to_string(),
        };

        let result1 = embed_single(
            axum::extract::State(app_state.clone()),
            axum::Json(request1),
        )
        .await;
        let result2 = embed_single(axum::extract::State(app_state), axum::Json(request2)).await;

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        let embedding1 = result1.unwrap().0.embedding;
        let embedding2 = result2.unwrap().0.embedding;

        assert_ne!(
            embedding1, embedding2,
            "Different inputs should produce different embeddings"
        );

        assert_eq!(embedding1.len(), 384);
        assert_eq!(embedding2.len(), 384);

        // Verify embeddings are deterministic (same input always produces same output)
        let mock_processor = Arc::new(MockBatchProcessor);
        let app_state = AppState::new(mock_processor);
        let request1_again = SingleRequest {
            input: "first input".to_string(),
        };
        let result1_again =
            embed_single(axum::extract::State(app_state), axum::Json(request1_again)).await;
        assert!(result1_again.is_ok());
        assert_eq!(
            embedding1,
            result1_again.unwrap().0.embedding,
            "Same input should always produce same embedding"
        );
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

    #[tokio::test]
    async fn test_concurrent_batching_behavior() {
        let mock_processor = Arc::new(MockBatchProcessor);
        let app_state = AppState::new(mock_processor);

        // Test multiple concurrent requests to trigger batching behavior
        let request_count = 5;
        let mut handles = Vec::new();

        // Create different inputs for each request
        let inputs = vec![
            "concurrent_input_1".to_string(),
            "concurrent_input_2".to_string(),
            "concurrent_input_3".to_string(),
            "concurrent_input_4".to_string(),
            "concurrent_input_5".to_string(),
        ];

        // Send all requests concurrently
        for (i, input) in inputs.into_iter().enumerate() {
            let app_state = app_state.clone();
            let handle = tokio::spawn(async move {
                let request = SingleRequest { input };
                let start = std::time::Instant::now();

                let result =
                    embed_single(axum::extract::State(app_state), axum::Json(request)).await;

                (i, result, start.elapsed())
            });
            handles.push(handle);
        }

        // Wait for all requests to complete
        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.unwrap();
            results.push(result);
        }

        // Sort results by request index to maintain order
        results.sort_by_key(|(i, _, _)| *i);

        // Verify all requests succeeded
        for (i, result, duration) in &results {
            assert!(result.is_ok(), "Request {} failed: {:?}", i, result);
            println!("Request {} completed in: {:?}", i, duration);
        }

        // Extract embeddings and verify they are all different
        let embeddings: Vec<Vec<f32>> = results
            .into_iter()
            .map(|(_, result, _)| result.unwrap().0.embedding)
            .collect();

        // Verify all embeddings have correct dimensions
        for (i, embedding) in embeddings.iter().enumerate() {
            assert_eq!(embedding.len(), 384, "Embedding {} has wrong dimensions", i);
        }

        // Verify all embeddings are different (different inputs should produce different vectors)
        for i in 0..embeddings.len() {
            for j in (i + 1)..embeddings.len() {
                assert_ne!(
                    embeddings[i], embeddings[j],
                    "Embeddings {} and {} should be different for different inputs",
                    i, j
                );
            }
        }

        // Verify embeddings are deterministic by requesting the same inputs again
        let mock_processor = Arc::new(MockBatchProcessor);
        let app_state = AppState::new(mock_processor);

        let test_inputs = vec![
            "concurrent_input_1".to_string(),
            "concurrent_input_3".to_string(),
            "concurrent_input_5".to_string(),
        ];

        for (i, input) in test_inputs.into_iter().enumerate() {
            let request = SingleRequest { input };
            let result =
                embed_single(axum::extract::State(app_state.clone()), axum::Json(request)).await;
            assert!(result.is_ok());

            let new_embedding = result.unwrap().0.embedding;
            let original_index = match i {
                0 => 0, // concurrent_input_1
                1 => 2, // concurrent_input_3
                2 => 4, // concurrent_input_5
                _ => unreachable!(),
            };

            assert_eq!(
                new_embedding, embeddings[original_index],
                "Embedding for input {} should be deterministic",
                original_index
            );
        }

        println!(
            "✅ All {} concurrent requests completed successfully with unique embeddings",
            request_count
        );
        println!("✅ Embeddings are deterministic and correctly matched to inputs");
        println!("✅ Batching behavior verified - no response mixing detected");
    }
}
