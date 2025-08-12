use crate::{
    config,
    controller::{EmbedRequest, EmbedResponse},
    errors::ProxyError,
};
use crossbeam::channel;
use eyre::Result;
use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    sync::{Mutex, oneshot},
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
    batch_sender: channel::Sender<BatchRequest>,
}

impl BatchProcessor {
    pub async fn new(
        inference_url: String,
        config: config::Config,
        client: Arc<reqwest::Client>,
    ) -> Result<Self> {
        let (batch_sender, batch_receiver) = channel::unbounded::<BatchRequest>();

        let processor = Self {
            client: client.clone(),
            inference_url: inference_url.clone(),
            config,
            pending_requests: Arc::new(Mutex::new(VecDeque::new())),
            batch_sender,
        };

        processor.start_worker_pool(batch_receiver).await;
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

        let should_create_batch = {
            let mut pending = self.pending_requests.lock().await;
            pending.push_back(pending_request);
            pending.len() >= self.config.max_batch_size
        };

        if should_create_batch {
            self.try_create_batch().await?;
        }

        let result = timeout(
            Duration::from_millis(
                self.config.max_wait_time_ms as u64 + self.config.max_batch_size as u64 / 10,
            ),
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

    async fn start_worker_pool(&self, batch_receiver: channel::Receiver<BatchRequest>) {
        let client = self.client.clone();
        let inference_url = self.inference_url.clone();
        let max_workers = self.config.max_concurrent_workers;

        tracing::info!("Starting worker pool with {} workers", max_workers);

        for worker_id in 0..max_workers {
            let worker_client = client.clone();
            let worker_inference_url = inference_url.clone();
            let worker_receiver = batch_receiver.clone();

            tokio::spawn(async move {
                Self::run_worker(
                    worker_id,
                    worker_client,
                    worker_inference_url,
                    worker_receiver,
                )
                .await;
            });
        }
    }

    async fn run_worker(
        worker_id: usize,
        client: Arc<reqwest::Client>,
        inference_url: String,
        receiver: channel::Receiver<BatchRequest>,
    ) {
        while let Ok(batch) = receiver.recv() {
            Self::process_batch(worker_id, &client, &inference_url, batch).await;
        }

        tracing::debug!("Worker {} shutting down", worker_id);
    }

    async fn process_batch(
        worker_id: usize,
        client: &Arc<reqwest::Client>,
        inference_url: &str,
        batch: BatchRequest,
    ) {
        let batch_size = batch.requests.len();
        let batch_age = batch.created_at.elapsed();

        tracing::debug!(
            "Worker {} processing batch with {} requests (age: {:?})",
            worker_id,
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
        let result = Self::send_batch_request(client, inference_url, inputs).await;
        let processing_time = start_time.elapsed();

        match result {
            Ok(embeddings) => {
                for (sender, embedding) in senders.into_iter().zip(embeddings) {
                    let _ = sender.send(Ok(embedding));
                }
                tracing::debug!(
                    "Worker {} completed batch in {:?}",
                    worker_id,
                    processing_time
                );
            }
            Err(e) => {
                tracing::error!("Worker {} batch processing failed: {}", worker_id, e);
                for sender in senders {
                    let _ = sender.send(Err(ProxyError::BatchProcessingFailed(e.to_string())));
                }
            }
        }
    }

    async fn start_timeout_worker(&self) {
        let pending_requests = self.pending_requests.clone();
        let timeout_duration = Duration::from_millis(self.config.max_wait_time_ms as u64);
        let batch_sender = self.batch_sender.clone();

        tokio::spawn(async move {
            loop {
                tokio::time::sleep(timeout_duration).await;

                let should_create_batch = {
                    let pending = pending_requests.lock().await;
                    pending
                        .front()
                        .map(|oldest| oldest.created_at.elapsed() >= timeout_duration)
                        .unwrap_or(false)
                };

                if should_create_batch {
                    let requests: Vec<_> = {
                        let mut pending = pending_requests.lock().await;
                        pending.drain(..).collect()
                    };

                    if !requests.is_empty() {
                        let batch = BatchRequest {
                            requests,
                            created_at: Instant::now(),
                        };

                        tracing::debug!(
                            "Timeout reached, creating batch with {} requests",
                            batch.requests.len()
                        );

                        if let Err(e) = batch_sender.send(batch) {
                            tracing::error!("Failed to send timeout batch: {}", e);
                        }
                    }
                }
            }
        });
    }

    async fn try_create_batch(&self) -> Result<(), ProxyError> {
        let requests: Vec<_> = {
            let mut pending = self.pending_requests.lock().await;
            let batch_size = std::cmp::min(pending.len(), self.config.max_batch_size);
            pending.drain(..batch_size).collect()
        };

        if !requests.is_empty() {
            let batch = BatchRequest {
                requests,
                created_at: Instant::now(),
            };

            self.batch_sender
                .send(batch)
                .map_err(|_| ProxyError::Internal("Failed to send batch".into()))?;
        }

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
            .post(inference_url)
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
