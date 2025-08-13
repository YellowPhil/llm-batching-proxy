# Auto-Batching Proxy Service

A high-performance, intelligent auto-batching proxy service for ML inference workloads, built in Rust. This service automatically batches individual inference requests to maximize GPU utilization and throughput while maintaining a simple API for clients.

## Benchmarking

![benchmark.png](./bench.png)

### Performance Analysis & Batching Benefits

Benchmarking demonstrates that the auto-batching solution effectively reduces latency throughput compared to individual requests.  While the primary performance bottleneck remains CPU usage during model inference, the benefits scale exponentially with higher QPS. Additionally, smaller token counts benefit more from batching due to lower per-request processing overhead and faster batch formation, with short text (10-50 tokens)

## 🚀 Key Features

- **Configurable Batching Strategy**: Tunable batch size and timeout parameters
- **High Performance**: Async Rust implementation with minimal latency overhead  
- **Production Ready**: Full error handling, logging, health checks, and Docker support

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client    │───▶│  Auto-Batching   │───▶│   Inference     │
│  Requests   │    │  Proxy Service   │    │   Service       │
└─────────────┘    └──────────────────┘    └─────────────────┘
                           │
                           ▼
                   ┌──────────────┐
                   │   Metrics &  │
                   │  Monitoring  │
                   └──────────────┘
```

### Core Components

- **BatchProcessor**: Intelligent batching engine with timeout and size-based triggering
- **Configuration System**: Flexible, file-based configuration management
- **Error Handling**: Robust error handling with proper HTTP status codes


### Measures


## 📋 Requirements

- **Docker & Docker Compose** (for easy deployment)
- **Text Embeddings Inference Service** (or compatible inference API)

## 🚀 Quick Start

### 1. Clone and Build

```bash
git clone <your-repo-url>
cd auto-batching-proxy
cargo build --release
```

### 2. Configuration

Create `config.toml`:

```toml
# Maximum number of requests in a single batch
max_batch_size = 32

# Maximum wait time in milliseconds before forcing a batch
max_wait_time_ms = 100

# Maximum number of workers
max_concurrent_workers = 20
```

### 3. Run with Docker Compose

```bash
# Start both inference service and proxy
docker-compose up --build -d

# Check health
curl http://localhost:3000/health
```

### 4. Manual Setup (Development)

```bash
# Start the inference service
docker run --rm -it -p 8080:80 --pull always \
  ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
  --model-id Qwen/Qwen3-Embedding-0.6B

# Start the proxy service
cargo run -- --port 3000 --inference-url http://127.0.0.1:8080
```

## 📡 API Endpoints

### Individual Request (Auto-batched)
```bash
curl -X POST http://localhost:3000/embed \
  -H "Content-Type: application/json" \
  -d '{"input": "What is vector search?"}'
```

Response:
```json
{
  "embedding": [0.1, 0.2, 0.3, ...]
}
```
## 🔧 Configuration Options

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `max_batch_size` | Maximum requests per batch | 32 | 1-1000 |
| `max_wait_time_ms` | Maximum wait time for batching | 100ms | 1-10000ms |

### Environment Variables

- `INFERENCE_URL`: URL of the inference service
- `RUST_LOG`: Log level configuration
- `PORT`: Service port (default: 3000)

## 📊 Batching Strategy

1. **Batch Size Limit**: Number of pending requests reaches `max_batch_size`
2. **Timeout**: Oldest request has been waiting for `max_wait_time_ms` 
3. **Intelligent Timing**: Optimizes for both latency and throughput

### Batching Flow

```
Request 1 ──┐
Request 2 ──┤
Request 3 ──┤──▶ Batch Formation ──▶ Inference API ──▶ Response Distribution
Request 4 ──┤     (Size/Timeout)
Request N ──┘
```

### Expected Performance Improvements

Based on testing with typical workloads:

- **Latency Improvement**
- **Throughput Improvement**
- **GPU Utilization**
- **Memory Efficiency**