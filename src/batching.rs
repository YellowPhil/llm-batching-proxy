use crate::{config, controller::EmbedRequest, errors::ProxyError};
use crossbeam::channel;
use eyre::Result;
use std::{
    collections::VecDeque,
    result,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{Notify, RwLock};
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

    request_queue: Arc<RwLock<VecDeque<PendingRequest>>>,
    batch_sender: channel::Sender<BatchRequest>,
    request_notify: Arc<Notify>,
}

impl BatchProcessor {
    pub async fn new(
        inference_url: String,
        config: config::Config,
        client: Arc<reqwest::Client>,
    ) -> Result<Self> {
        let (batch_sender, batch_receiver) =
            channel::bounded::<BatchRequest>(config.max_batch_size * 10);

        let processor = Self {
            client: client.clone(),
            inference_url,
            config,
            request_queue: Arc::new(RwLock::new(VecDeque::new())),
            batch_sender,
            request_notify: Arc::new(Notify::new()),
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
            let mut pending = self.request_queue.write().await;
            pending.push_back(pending_request);
            pending.len() >= self.config.max_batch_size
        };

        if should_create_batch {
            tracing::info!("Creating batch because of max batch size");
            let requests = {
                let mut pending = self.request_queue.write().await;
                let requests: Vec<PendingRequest> = pending.drain(..).collect();
                requests
            };

            let batch = BatchRequest {
                requests,
                created_at: Instant::now(),
            };
            if let Err(e) = self.batch_sender.send(batch) {
                tracing::error!("Failed to send batch: {:?}", e);
                return Err(ProxyError::BatchProcessingFailed(e.to_string()));
            }
        }
        self.request_notify.notify_one();

        tracing::debug!("Waiting for response");
        let result = timeout(
            Duration::from_millis(self.config.max_wait_time_ms as u64 * 20),
            receiver,
        )
        .await;

        match result {
            Ok(Ok(embedding)) => {
                tracing::debug!("Response received: {:?}", embedding);
                embedding
            }
            Ok(Err(e)) => {
                tracing::error!("Batch processing failed: {:?}", e);
                Err(ProxyError::BatchProcessingFailed(e.to_string()))
            }
            Err(e) => {
                tracing::error!("Request timed out: {:?}", e);
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
                tracing::debug!("Worker {} starting", worker_id);
                Self::run_worker(
                    worker_id,
                    worker_client,
                    worker_inference_url,
                    worker_receiver,
                )
                .await;
                tracing::debug!("Worker {} finished", worker_id);
            });
        }

        tracing::info!("Worker pool started with {} workers", max_workers);
    }

    async fn run_worker(
        worker_id: usize,
        client: Arc<reqwest::Client>,
        inference_url: String,
        receiver: channel::Receiver<BatchRequest>,
    ) {
        while let Ok(batch) = receiver.recv() {
            tracing::debug!(
                "Worker {} received batch with {} requests",
                worker_id,
                batch.requests.len()
            );
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

        let (inputs, senders): (
            Vec<String>,
            Vec<oneshot::Sender<Result<Vec<f32>, ProxyError>>>,
        ) = batch
            .requests
            .into_iter()
            .map(|req| (req.input, req.sender))
            .unzip();

        let start_time = Instant::now();
        let result = Self::send_batch_request(client, inference_url, inputs).await;
        let processing_time = start_time.elapsed();

        match result {
            Ok(embeddings) => {
                //sanity check
                if embeddings.len() != senders.len() {
                    tracing::error!(
                        "Number of embeddings and senders do not match: {} != {}",
                        embeddings.len(),
                        senders.len()
                    );
                    return;
                }

                for (sender, embedding) in senders.into_iter().zip(embeddings) {
                    match sender.send(Ok(embedding)) {
                        Ok(()) => tracing::debug!("Send response successful"),
                        Err(_) => {
                            tracing::warn!("Send response failed, receiver already gone")
                        }
                    }
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
                    if let Err(e) =
                        sender.send(Err(ProxyError::BatchProcessingFailed(e.to_string())))
                    {
                        tracing::warn!("Failed to send error response to sender: {:?}", e);
                    }
                }
            }
        }
    }

    async fn start_timeout_worker(&self) {
        let pending_requests = self.request_queue.clone();
        let timeout_duration = Duration::from_millis(self.config.max_wait_time_ms as u64);
        let batch_sender = self.batch_sender.clone();
        let request_notify = self.request_notify.clone();

        tokio::spawn(async move {
            loop {
                request_notify.notified().await;
                tokio::time::sleep(timeout_duration).await;
                tracing::debug!("Timeout worker woke up");

                let should_create_batch = {
                    pending_requests
                        .read()
                        .await
                        .front()
                        .map(|oldest| oldest.created_at.elapsed() >= timeout_duration)
                        .unwrap_or(false)
                };

                if should_create_batch {
                    let requests = {
                        let mut pending = pending_requests.write().await;
                        let requests: Vec<PendingRequest> = pending.drain(..).collect();
                        requests
                    };

                    let batch = BatchRequest {
                        requests,
                        created_at: Instant::now(),
                    };
                    if let Err(e) = batch_sender.send(batch) {
                        tracing::error!("Failed to send batch: {:?}", e);
                    }
                }
                tracing::debug!("No batch created by timer worker");
            }
        });
    }

    #[tracing::instrument(skip(client, inputs))]
    async fn send_batch_request(
        client: &Arc<reqwest::Client>,
        inference_url: &str,
        inputs: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, ProxyError> {
        let request_payload = EmbedRequest { inputs };

        tracing::info!(
            "Sending batch request with {} len",
            request_payload.inputs.len(),
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

        let embed_response = response
            .json::<Vec<Vec<f32>>>()
            .await
            .map_err(|e| ProxyError::InferenceService(format!("Failed to get response: {}", e)))?;

        tracing::debug!("Embed response: {:?}", embed_response);

        Ok(embed_response)
    }
}
