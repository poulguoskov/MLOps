"""Benchmark Cloud Run APIs with 1000 requests."""

import argparse
import statistics
import subprocess
import time

import requests


def get_auth_token():
    """Get GCP identity token."""
    result = subprocess.run(["gcloud", "auth", "print-identity-token"], capture_output=True, text=True)
    return result.stdout.strip()


def benchmark_api(url: str, n_requests: int = 1000, auth: bool = True):
    """Benchmark API with n requests."""
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {get_auth_token()}"

    payload = {"text": "You Will NEVER Believe What Happened Next!"}

    times = []
    errors = 0

    print(f"Benchmarking {url}")
    print(f"Running {n_requests} requests...")

    start_total = time.time()

    for i in range(n_requests):
        start = time.time()
        try:
            response = requests.post(f"{url}/classify", json=payload, headers=headers)
            if response.status_code == 200:
                times.append(time.time() - start)
            else:
                errors += 1
        except Exception:
            errors += 1

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_requests} completed...")

    total_time = time.time() - start_total

    if times:
        print(f"\n{'=' * 50}")
        print(f"Results for {url}")
        print(f"{'=' * 50}")
        print(f"Total requests:    {n_requests}")
        print(f"Successful:        {len(times)}")
        print(f"Errors:            {errors}")
        print(f"Total time:        {total_time:.2f}s")
        print(f"Requests/sec:      {len(times) / total_time:.2f}")
        print(f"Mean latency:      {statistics.mean(times) * 1000:.1f}ms")
        print(f"Median latency:    {statistics.median(times) * 1000:.1f}ms")
        print(f"P95 latency:       {sorted(times)[int(len(times) * 0.95)] * 1000:.1f}ms")
        print(f"P99 latency:       {sorted(times)[int(len(times) * 0.99)] * 1000:.1f}ms")
        print(f"Min latency:       {min(times) * 1000:.1f}ms")
        print(f"Max latency:       {max(times) * 1000:.1f}ms")

    return times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--no-auth", action="store_true")
    args = parser.parse_args()

    benchmark_api(args.url, args.requests, auth=not args.no_auth)
