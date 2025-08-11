use crate::{EmbedRequest, EmbedResponse, config, errors::ProxyError};
use eyre::{Result, WrapErr};
use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    sync::{Mutex, mpsc, oneshot},
    time::timeout,
};

#[derive(Debug)]
struct PendingRequest {
    input: String,
    sender: oneshot::Sender<Result<Vec<f32>, ProxyError>>,
    created_at: Instant,
}

#[derive(Debug)]
struct BatchRequest {
    requests: Vec<PendingRequest>,
    created_at: Instant,
}

pub struct BatchProcessor {
    client: Arc<reqwest::Client>,
    inference_url: String,
    config: config::Config,

    pending_requests: Arc<Mutex<VecDeque<PendingRequest>>>,
    batch_sender: mpsc::Sender<BatchRequest>,
}

impl BatchProcessor {
    pub async fn new(
        inference_url: String,
        config: config::Config,
        client: Arc<reqwest::Client>,
    ) -> Result<Self> {
        let (batch_sender, batch_receiver) = mpsc::channel(config.max_batch_size * 10);

        let processor = Self {
            client: client.clone(),
            inference_url: inference_url.clone(),
            config,
            pending_requests: Arc::new(Mutex::new(VecDeque::new())),
            batch_sender,
        };

        processor.start_batch_worker(batch_receiver).await;
        processor.start_timeout_worker().await;

        Ok(processor)
    }

    #[tracing::instrument(skip(self))]
    pub async fn process_single(&self, input: String) -> Result<Vec<f32>, ProxyError> {
        let (sender, receiver) = oneshot::channel();

        let pending_request = PendingRequest {
            input,
            sender,
            created_at: Instant::now(),
        };

        {
            let mut pending = self.pending_requests.lock().await;
            pending.push_back(pending_request);

            if pending.len() >= self.config.max_batch_size {
                tracing::debug!("Batch size limit reached, triggering immediate batch");
                self.try_create_batch(&mut pending).await?;
            }
        }

        let result = timeout(
            Duration::from_millis(self.config.max_wait_time_ms as u64 + 1000),
            receiver,
        )
        .await;

        match result {
            Ok(Ok(embedding)) => embedding,
            Ok(Err(e)) => Err(ProxyError::InferenceService(format!(
                "Failed to get embedding: {}",
                e
            ))),
            Err(_) => {
                tracing::error!("Request timed out waiting for batch processing");
                Err(ProxyError::Timeout)
            }
        }
    }

    async fn start_batch_worker(&self, mut receiver: mpsc::Receiver<BatchRequest>) {
        let client = self.client.clone();
        let inference_url = self.inference_url.clone();

        tokio::spawn(async move {
            while let Some(batch) = receiver.recv().await {
                let batch_size = batch.requests.len();
                let batch_age = batch.created_at.elapsed();

                tracing::debug!(
                    "Processing batch with {} requests (age: {:?})",
                    batch_size,
                    batch_age
                );

                let mut inputs: Vec<String> = Vec::with_capacity(batch_size);
                let mut senders: Vec<oneshot::Sender<Result<Vec<f32>, ProxyError>>> =
                    Vec::with_capacity(batch_size);

                for req in batch.requests {
                    senders.push(req.sender);
                    inputs.push(req.input);
                }

                let start_time = Instant::now();
                let result = Self::send_batch_request(&client, &inference_url, inputs).await;
                let processing_time = start_time.elapsed();

                match result {
                    Ok(embeddings) => {
                        for (sender, embedding) in senders.into_iter().zip(embeddings) {
                            let _ = sender.send(Ok(embedding));
                        }
                        tracing::debug!("Batch processed successfully in {:?}", processing_time);
                    }
                    Err(e) => {
                        tracing::error!("Batch processing failed: {}", e);
                        for sender in senders {
                            let _ =
                                sender.send(Err(ProxyError::BatchProcessingFailed(e.to_string())));
                        }
                    }
                }
            }
        });
    }

    async fn start_timeout_worker(&self) {
        let pending_requests = self.pending_requests.clone();
        let batch_sender = self.batch_sender.clone();
        let timeout_duration = Duration::from_millis(self.config.max_wait_time_ms as u64);

        tokio::spawn(async move {
            let sleep_duration = std::cmp::min(timeout_duration / 10, Duration::from_millis(100));

            loop {
                tokio::time::sleep(sleep_duration).await;

                let mut pending = pending_requests.lock().await;
                if pending.is_empty() {
                    continue;
                }

                if let Some(oldest) = pending.front() {
                    if oldest.created_at.elapsed() < timeout_duration {
                        continue;
                    }
                    tracing::debug!(
                        "Timeout reached, creating batch with {} requests",
                        pending.len()
                    );

                    let requests: Vec<_> = pending.drain(..).collect();
                    if !requests.is_empty() {
                        let batch = BatchRequest {
                            requests,
                            created_at: Instant::now(),
                        };

                        if let Err(e) = batch_sender.send(batch).await {
                            tracing::error!("Failed to send timeout batch: {}", e);
                        }
                    }
                }
            }
        });
    }

    async fn try_create_batch(
        &self,
        pending: &mut VecDeque<PendingRequest>,
    ) -> Result<(), ProxyError> {
        if pending.is_empty() {
            return Ok(());
        }

        let batch_size = std::cmp::min(pending.len(), self.config.max_batch_size);
        let requests: Vec<_> = pending.drain(..batch_size).collect();

        let batch = BatchRequest {
            requests,
            created_at: Instant::now(),
        };

        self.batch_sender
            .send(batch)
            .await
            .map_err(|_| ProxyError::Internal("Failed to send batch".into()))?;

        Ok(())
    }

    #[tracing::instrument(skip(client, inputs))]
    async fn send_batch_request(
        client: &Arc<reqwest::Client>,
        inference_url: &str,
        inputs: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, ProxyError> {
        let request_payload = EmbedRequest { inputs };

        tracing::debug!(
            "Sending batch request with {} inputs to {}",
            request_payload.inputs.len(),
            inference_url
        );

        let response = client
            .post(&format!("{}/embed", inference_url))
            .json(&request_payload)
            .send()
            .await
            .map_err(|e| ProxyError::InferenceService(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ProxyError::InferenceService(format!(
                "Inference service returned {}: {}",
                status, body
            )));
        }

        let embed_response: EmbedResponse = response
            .json()
            .await
            .map_err(|e| ProxyError::InferenceService(format!("Failed to get response: {}", e)))?;

        Ok(embed_response.embeddings)
    }
}
