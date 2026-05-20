"""Throw NSL-KDD events at the live Logstash input and measure how long
they take to land in Elasticsearch (p50/p95/p99). Needs the Docker stack
running.
"""

import argparse
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOGSTASH_URL = "http://localhost:8080"
ES_URL = "http://localhost:9200"
LOGSTASH_API = "http://localhost:9600"


def load_sample_events(n: int) -> list[dict]:
    """Pull n rows from KDDTest+ keeping only the 15 features the model wants."""
    csv = PROJECT_DIR / "data" / "nsl_kdd" / "KDDTest+.txt"
    if not csv.exists():
        csv = PROJECT_DIR / "data" / "nsl_kdd" / "KDDTrain+.txt"
    if not csv.exists():
        raise FileNotFoundError(
            "Run scripts/01_train_nsl_kdd.py first to fetch the dataset"
        )

    # the 15 columns selected by mutual info in script 01,
    # in the order the trained model expects
    SELECTED_IDX = [2, 3, 4, 5, 11, 22, 24, 25, 28, 29, 32, 33, 34, 37, 38]

    rows = pd.read_csv(csv, header=None, nrows=n).iloc[:, :41].fillna(0)
    for col in rows.select_dtypes(include="object").columns:
        rows[col] = rows[col].astype("category").cat.codes
    rows = rows.iloc[:, SELECTED_IDX].astype(np.float32)
    # plain floats so json.dumps doesn't choke on numpy types
    return [{"features": [float(x) for x in r], "synthetic": True} for r in rows.to_numpy()]


def send_one(event: dict, run_id: str, _session_unused=None) -> dict:
    event = dict(event)
    event["run_id"] = run_id
    event["client_ts_ms"] = int(time.time() * 1000)
    # fresh Session per call; requests.Session is not thread-safe enough for this
    try:
        r = requests.post(LOGSTASH_URL, json=event, timeout=10,
                          headers={"Content-Type": "application/json"})
        return {"status": r.status_code, "ts": event["client_ts_ms"]}
    except Exception as e:
        return {"status": "error", "error": str(e), "ts": event["client_ts_ms"]}


def run_load(events: list[dict], target_rate: int, duration: int, run_id: str) -> dict:
    print(f"Sending {len(events):,} events at target {target_rate}/s for {duration}s...")
    interval = 1.0 / target_rate
    sent = 0
    errors = 0
    started = time.perf_counter()
    deadline = started + duration

    sample_errors: list[str] = []

    def _drain(futures):
        nonlocal sent, errors
        for f in as_completed(futures):
            r = f.result()
            sent += 1
            if r["status"] != 200:
                errors += 1
                if len(sample_errors) < 5:
                    sample_errors.append(str(r))

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures: list = []
        idx = 0
        next_send = started
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if now < next_send:
                time.sleep(next_send - now)
            futures.append(pool.submit(send_one, events[idx % len(events)], run_id))
            idx += 1
            next_send += interval
            if len(futures) >= 500:
                _drain(futures)
                futures = []
        _drain(futures)

    if sample_errors:
        print("Sample failures:")
        for e in sample_errors:
            print(f"  {e}")

    elapsed = time.perf_counter() - started
    return {
        "sent": sent,
        "errors": errors,
        "elapsed_s": elapsed,
        "achieved_rate_eps": sent / elapsed if elapsed > 0 else 0.0,
    }


def fetch_latencies(run_id: str, expected: int, max_wait_s: int = 60) -> list[int]:
    """Pull every doc with our run_id from ES and compute ingest-to-index latency."""
    print(f"Waiting up to {max_wait_s}s for ES to index {expected:,} events...")
    deadline = time.time() + max_wait_s
    latencies: list[int] = []

    while time.time() < deadline:
        body = {
            "size": 10000,
            "query": {"term": {"run_id": run_id}},
            "_source": ["client_ts_ms", "ingest_ts_ms", "infer_ts_ms", "@timestamp"],
        }
        r = requests.post(
            f"{ES_URL}/siemai-events-v2-*/_search",
            json=body,
            timeout=10,
        )
        if r.status_code != 200:
            time.sleep(2)
            continue
        hits = r.json().get("hits", {}).get("hits", [])
        latencies = []
        for h in hits:
            src = h["_source"]
            client = src.get("client_ts_ms")
            infer = src.get("infer_ts_ms")
            if client and infer:
                latencies.append(int(infer) - int(client))
        print(f"  indexed {len(hits):,}/{expected:,}")
        if len(hits) >= expected * 0.95:
            return latencies
        time.sleep(2)
    return latencies


def fetch_logstash_throughput() -> dict:
    try:
        r = requests.get(f"{LOGSTASH_API}/_node/stats/events", timeout=5)
        return r.json().get("events", {})
    except Exception:
        return {}


def summarize(latencies_ms: list[int]) -> dict:
    if not latencies_ms:
        return {"count": 0}
    arr = np.array(latencies_ms)
    return {
        "count": int(arr.size),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "max_ms": int(arr.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=int, default=1000, help="target events/s")
    ap.add_argument("--duration", type=int, default=60, help="seconds")
    ap.add_argument("--n-events", type=int, default=5000, help="distinct events to cycle through")
    args = ap.parse_args()

    run_id = f"bench-{uuid.uuid4().hex[:8]}"
    print(f"Run id: {run_id}")

    events = load_sample_events(args.n_events)

    load_stats = run_load(events, args.rate, args.duration, run_id)
    print(f"\nLoad complete: {load_stats}")

    latencies = fetch_latencies(run_id, expected=load_stats["sent"])
    latency_stats = summarize(latencies)
    print(f"\nLatency stats: {latency_stats}")

    throughput = fetch_logstash_throughput()
    print(f"\nLogstash throughput: {throughput}")

    result = {
        "run_id": run_id,
        "target_rate_eps": args.rate,
        "duration_s": args.duration,
        "load_stats": load_stats,
        "ingestion_to_alert_latency_ms": latency_stats,
        "logstash_throughput": throughput,
    }
    out = RESULTS_DIR / "streaming_benchmark.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
