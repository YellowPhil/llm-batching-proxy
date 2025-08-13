import time
import random
import string
import threading
from statistics import mean
import asyncio
import aiohttp
import matplotlib.pyplot as plt
import psutil  # pip install psutil

# =========================
# CONFIGURATION
# =========================
DIRECT_URL = "http://localhost:8080/embed"   # Direct TEI
BATCH_URL  = "http://localhost:4000/embed"   # Batching backend
TOTAL_TEXTS = 7000                          # Number of test inputs
CPU_MONITOR_INTERVAL = 0.1                  # Seconds between CPU usage samples
CONCURRENCY = 100 # Max concurrent requests

# =========================
# TEST DATA GENERATION
# =========================
def generate_random_word(min_len=10, max_len=100):
    return ''.join(random.choices(string.ascii_lowercase, k=random.randint(min_len, max_len)))

def generate_dataset(n=TOTAL_TEXTS):
    return [generate_random_word() for _ in range(n)]

# =========================
# CPU UTILIZATION MONITOR
# =========================
class CPUMonitor(threading.Thread):
    def __init__(self, interval=CPU_MONITOR_INTERVAL):
        super().__init__()
        self.interval = interval
        self.running = False
        self.samples = []
        self.timestamps = []

    def run(self):
        self.running = True
        start = time.time()
        while self.running:
            util = psutil.cpu_percent(interval=None)
            self.samples.append(util)
            self.timestamps.append(time.time() - start)
            time.sleep(self.interval)

    def stop(self):
        self.running = False

    def average_utilization(self):
        return mean(self.samples) if self.samples else 0

# =========================
# ASYNC BENCHMARK FUNCTIONS
# =========================
async def fetch(session, url, payload, latencies, sem):
    async with sem:
        start = time.time()
        async with session.post(url, json=payload) as resp:
            await resp.read()
        latencies.append(time.time() - start)

async def run_async_benchmark(texts, url, payload_format):
    sem = asyncio.Semaphore(CONCURRENCY)
    latencies = []

    monitor = CPUMonitor()
    monitor.start()

    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for t in texts:
            if payload_format == "direct":
                payload = {"inputs": [t]}
            else:  # batch backend
                payload = {"input": t}
            tasks.append(fetch(session, url, payload, latencies, sem))
        await asyncio.gather(*tasks)

    total_time = time.time() - start_time
    monitor.stop()

    return {
        "total_time": total_time,
        "throughput": len(texts) / total_time,
        "avg_latency": mean(latencies),
        "latencies": latencies,
        "cpu_util": monitor.samples,
        "cpu_timestamps": monitor.timestamps
    }

# =========================
# HISTOGRAM PLOTTING
# =========================
def plot_histograms(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Latency histograms
    axes[0, 0].hist(results[0]['latencies'], bins=50, alpha=0.7, label="Direct TEI")
    axes[0, 0].hist(results[1]['latencies'], bins=50, alpha=0.7, label="Batch Backend")
    axes[0, 0].set_title("Latency Histogram")
    axes[0, 0].set_xlabel("Latency (seconds)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()

    # CPU utilization histograms
    axes[0, 1].hist(results[0]['cpu_util'], bins=20, alpha=0.7, label="Direct TEI")
    axes[0, 1].hist(results[1]['cpu_util'], bins=20, alpha=0.7, label="Batch Backend")
    axes[0, 1].set_title("CPU Utilization Histogram")
    axes[0, 1].set_xlabel("CPU Utilization (%)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()

    # CPU utilization over time
    axes[1, 0].plot(results[0]['cpu_timestamps'], results[0]['cpu_util'], label="Direct TEI")
    axes[1, 0].plot(results[1]['cpu_timestamps'], results[1]['cpu_util'], label="Batch Backend")
    axes[1, 0].set_title("CPU Utilization Over Time")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("CPU Utilization (%)")
    axes[1, 0].legend()

    # Throughput bar chart
    axes[1, 1].bar(["Direct TEI", "Batch Backend"],
                   [results[0]['throughput'], results[1]['throughput']],
                   alpha=0.7)
    axes[1, 1].set_title("Throughput Comparison")
    axes[1, 1].set_ylabel("Requests/sec")

    plt.tight_layout()
    plt.savefig("benchmark_results.png")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    dataset = generate_dataset()

    print("Running Direct TEI benchmark...")
    direct_results = asyncio.run(run_async_benchmark(dataset, DIRECT_URL, "direct"))

    print("Running Batch Backend benchmark...")
    batch_results = asyncio.run(run_async_benchmark(dataset, BATCH_URL, "batch"))

    print("\n=== Benchmark Results ===")
    print(f"Direct TEI: total_time={direct_results['total_time']:.2f}s, "
          f"throughput={direct_results['throughput']:.2f} req/s, "
          f"avg_latency={direct_results['avg_latency']:.4f}s, "
          f"avg_cpu={mean(direct_results['cpu_util']):.1f}%")

    print(f"Batch Backend: total_time={batch_results['total_time']:.2f}s, "
          f"throughput={batch_results['throughput']:.2f} req/s, "
          f"avg_latency={batch_results['avg_latency']:.4f}s, "
          f"avg_cpu={mean(batch_results['cpu_util']):.1f}%")

    plot_histograms([direct_results, batch_results])