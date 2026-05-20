"""Ablation: how much does each stage of the pipeline actually cost?

  A. Just /predict (pure inference)
  B. Logstash -> ES with ML scoring switched off
  C. Full pipeline (B + ML)

Diff between B and C is the marginal cost of inserting ML.
"""

import argparse
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

import numpy as np
import requests

ML_URL = "http://localhost:5001/predict"
LOGSTASH_URL = "http://localhost:8080"
ES_URL = "http://localhost:9200"

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# 15-feature NSL-KDD-shaped event used as a fixed payload
SAMPLE = [0.0, 9.0, 491.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 150.0, 1.0, 0.0, 0.0, 0.0]


def variant_a_ml_only(n: int = 1000) -> dict:
    """Hit /predict directly, n times, and time it."""
    print(f"[A] Direct ML inference, {n} requests ...")
    latencies = []
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            r = requests.post(ML_URL, json={"features": SAMPLE}, timeout=5)
            if r.status_code == 200:
                latencies.append((time.perf_counter() - t0) * 1000)
        except Exception:
            pass
    return summarize(latencies)


def variant_b_pipeline_no_ml(n: int = 1000, rate: int = 100, duration: int = 15) -> dict:
    """Same pipeline, but send events without `features` so Logstash skips the
    ML http filter (see logstash/pipeline/siem.conf:
    `if [features] { http {...} }`). Just measures ingest-to-index.
    """
    print(f"[B] Pipeline (no ML) at {rate}/s for {duration}s ...")
    run_id = f"abl-noml-{uuid.uuid4().hex[:8]}"
    return run_load(rate, duration, run_id, with_features=False)


def variant_c_pipeline_with_ml(n: int = 1000, rate: int = 100, duration: int = 15) -> dict:
    """Full pipeline (same as scripts/04)."""
    print(f"[C] Pipeline (with ML) at {rate}/s for {duration}s ...")
    run_id = f"abl-ml-{uuid.uuid4().hex[:8]}"
    return run_load(rate, duration, run_id, with_features=True)


def run_load(rate: int, duration: int, run_id: str, with_features: bool) -> dict:
    started = time.perf_counter()
    deadline = started + duration
    interval = 1.0 / rate

    def send_one(_):
        event = {"run_id": run_id, "client_ts_ms": int(time.time() * 1000),
                 "synthetic": True, "ablation": "with_ml" if with_features else "no_ml"}
        if with_features:
            event["features"] = SAMPLE
        try:
            r = requests.post(LOGSTASH_URL, json=event, timeout=10,
                              headers={"Content-Type": "application/json"})
            return r.status_code
        except Exception:
            return None

    sent, errs, next_send = 0, 0, started
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = []
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if now < next_send:
                time.sleep(next_send - now)
            futs.append(pool.submit(send_one, None))
            sent += 1
            next_send += interval
            if len(futs) >= 500:
                for f in as_completed(futs):
                    if f.result() != 200:
                        errs += 1
                futs = []
        for f in as_completed(futs):
            if f.result() != 200:
                errs += 1

    # Wait for ES indexing then fetch latencies
    elapsed = time.perf_counter() - started
    print(f"  sent {sent}, errors {errs}, achieved {sent/elapsed:.0f}/s")
    print("  waiting 30s for ES indexing ...")
    deadline = time.time() + 60
    latencies = []
    while time.time() < deadline:
        time.sleep(2)
        body = {"size": 10000, "query": {"term": {"run_id": run_id}},
                "_source": ["client_ts_ms", "ingest_ts_ms", "infer_ts_ms"]}
        try:
            r = requests.post(f"{ES_URL}/siemai-events-v2-*/_search",
                              json=body, timeout=10)
            if r.status_code == 200:
                hits = r.json().get("hits", {}).get("hits", [])
                latencies = []
                for h in hits:
                    s = h["_source"]
                    c = s.get("client_ts_ms"); i = s.get("infer_ts_ms")
                    if c and i:
                        latencies.append(int(i) - int(c))
                if len(hits) >= sent * 0.95:
                    break
                print(f"    indexed {len(hits):,}/{sent}")
        except Exception:
            pass

    return {
        "sent": sent, "errors": errs,
        "indexed": len(latencies),
        **summarize(latencies),
    }


def summarize(latencies):
    if not latencies:
        return {"count": 0}
    arr = np.array(latencies)
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
    ap.add_argument("--ml-only-n", type=int, default=1000)
    ap.add_argument("--pipeline-rate", type=int, default=100)
    ap.add_argument("--pipeline-duration", type=int, default=15)
    args = ap.parse_args()

    out = {
        "A_ml_only_direct": variant_a_ml_only(args.ml_only_n),
        "B_pipeline_no_ml": variant_b_pipeline_no_ml(
            rate=args.pipeline_rate, duration=args.pipeline_duration),
        "C_pipeline_with_ml": variant_c_pipeline_with_ml(
            rate=args.pipeline_rate, duration=args.pipeline_duration),
    }

    # Compute marginal ML cost (p95 difference)
    if out["B_pipeline_no_ml"].get("p95_ms") and out["C_pipeline_with_ml"].get("p95_ms"):
        out["marginal_ml_cost_p95_ms"] = (out["C_pipeline_with_ml"]["p95_ms"]
                                            - out["B_pipeline_no_ml"]["p95_ms"])

    print("\n=== Ablation summary ===")
    print(json.dumps(out, indent=2))

    json.dump(out, open(RESULTS / "ablation.json", "w"), indent=2)
    print(f"\nSaved {RESULTS / 'ablation.json'}")


if __name__ == "__main__":
    main()
