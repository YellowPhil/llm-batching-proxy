# This is a Rust Service Developer interview challenge

This is an interview challenge for a Rust Service Developer position at Qdrant.

We develop a number of in-cloud services that work closely with the Qdrant search engine and our cloud platform.


This challenge proposes to implement a simplified version of a typical service.


## Service description

For this challenge, we ask you to implement a simple **auto-batching proxy** service.

The idea behind this service is based on the fact that batching inference requests together in a single batch request is more efficient (especially for GPU-based inference).
However, individual users might not have enough data to fill a batch.

The auto-batching proxy service should resolve this problem by automatically batching inference requests from multiple independent users, so that for users the interface looks like individual requests, but internally it is handled as a batch request.


## Requirements

* Proxy server should provide a REST API wrapper around some inference service.

We recommend to wrap https://github.com/huggingface/text-embeddings-inference
as it has a simple REST API and is easy to use.

Example:

Run inference service in a container:

```bash
docker run --rm -it -p 8080:80 --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-latest --model-id nomic-ai/nomic-embed-text-v1.5
```

Make a batch request to inference service:

```bash
curl 127.0.0.1:8080/embed -X POST -d '{"inputs":["What is Vector Search?", "Hello, world!"]}' -H 'Content-Type: application/json'
```


* Proxy server should be configured with following parameters:

    * Max Wait Time - maximal time user request can wait for other requests to be accumulated in a batch
    * Max Batch Size - maximal number of requests that can be accumulated in a batch.

* Submission should include

    * Git repository with the code, including README with the description of the solution.
    * Instructions on how to run the service (ideally docker compose file).
    * Benchmark results of individual requests and batched requests.

Please feel free to use any tools/libraries/frameworks you find suitable for this task.

Usage of AI is allowed, but we would expect you to explain any part of the code and the reasoning behind it.