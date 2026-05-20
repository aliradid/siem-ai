"""SOAR latency harness. Runs three playbooks (iptables drop, TheHive case,
Azure account disable) end-to-end and times them. The Azure path is always
simulated; iptables runs for real if you're root; TheHive hits a real instance
if you set THEHIVE_URL + THEHIVE_API_KEY.
"""

import argparse
import json
import os
import statistics
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import requests

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOGSTASH_URL = "http://localhost:8080"
ES_URL = "http://localhost:9200"

THEHIVE_URL = os.environ.get("THEHIVE_URL")
THEHIVE_API_KEY = os.environ.get("THEHIVE_API_KEY")


def inject_malicious_event(run_id: str) -> int:
    """Send a NSL-KDD attack-shaped event (15 selected features) into the pipeline.
    Values are taken from a typical neptune/SYN-flood record so the trained XGBoost
    model classifies it as the attack class. Returns send_ts_ms.
    """
    # service, flag, src_bytes, dst_bytes, logged_in, count, serror_rate,
    # srv_serror_rate, same_srv_rate, diff_srv_rate, dst_host_srv_count,
    # dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_serror_rate,
    # dst_host_srv_serror_rate
    # real KDDTest+ attack record (15 selected features, original order); the
    # trained model scores it malicious (score ~1.0) once the serving scaler is applied
    event = {
        "run_id": run_id,
        "features": [49.0, 1.0, 0.0, 0.0, 0.0, 229.0, 0.0, 0.0, 0.04, 0.06, 10.0, 0.04, 0.06, 0.0, 0.0],
        "_label_map": {"0": "benign", "1": "malicious"},
        "source_ip": "203.0.113.42",
        "synthetic": True,
    }
    ts = int(time.time() * 1000)
    event["client_ts_ms"] = ts
    requests.post(LOGSTASH_URL, json=event, timeout=5,
                  headers={"Content-Type": "application/json"})
    return ts


def wait_for_alert(run_id: str, timeout_s: int = 30) -> Optional[dict]:
    """Poll the ALERTS index for the malicious event. Returns None on timeout —
    callers treat that as a failed cycle. No fallback to the main events index:
    a cycle is 'successful' only if the ML actually classified the event as
    non-benign and Logstash routed it to the alerts index. Silently accepting a
    benign-classified event would mask detection regressions in the per-cycle
    latency."""
    deadline = time.time() + timeout_s
    body = {
        "size": 1,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"run_id": run_id}},
                ]
            }
        },
        "sort": [{"@timestamp": "desc"}],
    }
    while time.time() < deadline:
        r = requests.post(f"{ES_URL}/siemai-events-v2-alerts-*/_search", json=body, timeout=5)
        if r.status_code == 200:
            hits = r.json().get("hits", {}).get("hits", [])
            if hits:
                return hits[0]["_source"]
        time.sleep(0.2)
    return None


def play_iptables_drop(ip: str) -> int:
    """Add an iptables DROP for ip. Returns elapsed ms. No-op if not root."""
    start = time.perf_counter()
    if os.geteuid() == 0:
        try:
            subprocess.run(
                ["iptables", "-I", "INPUT", "-s", ip, "-j", "DROP"],
                check=True, capture_output=True, timeout=5,
            )
            subprocess.run(
                ["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"],
                check=True, capture_output=True, timeout=5,
            )
        except Exception as e:
            print(f"  iptables failed: {e}")
    else:
        # Simulate the syscall cost with a sleep
        time.sleep(0.005)
    return int((time.perf_counter() - start) * 1000)


def play_thehive_case(alert: dict) -> int:
    start = time.perf_counter()
    if THEHIVE_URL and THEHIVE_API_KEY:
        try:
            requests.post(
                f"{THEHIVE_URL.rstrip('/')}/api/case",
                headers={"Authorization": f"Bearer {THEHIVE_API_KEY}"},
                json={
                    "title": f"SIEM-AI alert {alert.get('run_id')}",
                    "description": json.dumps(alert)[:1000],
                    "severity": 3,
                    "tags": ["siem-ai", "automated"],
                },
                timeout=5,
            )
        except Exception as e:
            print(f"  TheHive call failed: {e}")
    else:
        time.sleep(0.020)  # simulate network round trip
    return int((time.perf_counter() - start) * 1000)


def play_account_disable(account: str) -> int:
    """Simulated Azure AD Graph API account disable."""
    start = time.perf_counter()
    # In real deployment: POST to /users/{account}/microsoft.graph.disable
    time.sleep(0.050)
    return int((time.perf_counter() - start) * 1000)


def run_cycle(cycle_idx: int) -> dict:
    run_id = f"soar-{uuid.uuid4().hex[:8]}"
    t0 = time.perf_counter()

    send_ts = inject_malicious_event(run_id)
    t_send = (time.perf_counter() - t0) * 1000

    alert = wait_for_alert(run_id, timeout_s=30)
    t_detect = (time.perf_counter() - t0) * 1000
    if alert is None:
        return {
            "cycle": cycle_idx,
            "run_id": run_id,
            "ok": False,
            "reason": "alert never indexed within timeout",
        }

    iptables_ms = play_iptables_drop(alert.get("source_ip", "203.0.113.42"))
    thehive_ms = play_thehive_case(alert)
    azure_ms = play_account_disable(alert.get("user", "alice"))
    t_total = (time.perf_counter() - t0) * 1000

    return {
        "cycle": cycle_idx,
        "run_id": run_id,
        "ok": True,
        "send_ms": t_send,
        "ingestion_to_alert_ms": t_detect - t_send,
        "playbooks": {
            "iptables_ms": iptables_ms,
            "thehive_ms": thehive_ms,
            "azure_disable_ms": azure_ms,
        },
        "total_alert_to_containment_ms": t_total - t_detect,
        "total_ingest_to_containment_ms": t_total,
    }


def summarize(cycles: list[dict]) -> dict:
    ok = [c for c in cycles if c.get("ok")]
    if not ok:
        return {"successful_cycles": 0, "failures": len(cycles)}

    def p(field, q):
        vals = [c[field] for c in ok if field in c]
        return float(np.percentile(vals, q)) if vals else None

    return {
        "successful_cycles": len(ok),
        "failures": len(cycles) - len(ok),
        "ingestion_to_alert_ms": {
            "p50": p("ingestion_to_alert_ms", 50),
            "p95": p("ingestion_to_alert_ms", 95),
            "p99": p("ingestion_to_alert_ms", 99),
        },
        "ingest_to_containment_ms": {
            "p50": p("total_ingest_to_containment_ms", 50),
            "p95": p("total_ingest_to_containment_ms", 95),
            "p99": p("total_ingest_to_containment_ms", 99),
        },
        "playbook_means_ms": {
            "iptables": statistics.mean(c["playbooks"]["iptables_ms"] for c in ok),
            "thehive":  statistics.mean(c["playbooks"]["thehive_ms"] for c in ok),
            "azure":    statistics.mean(c["playbooks"]["azure_disable_ms"] for c in ok),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=20)
    args = ap.parse_args()

    print(f"Running {args.cycles} SOAR validation cycles...")
    cycles = []
    for i in range(args.cycles):
        c = run_cycle(i)
        ok = "✓" if c.get("ok") else "✗"
        total = c.get("total_ingest_to_containment_ms")
        total_str = f"{total:.1f} ms" if total is not None else "n/a"
        print(f"  [{i+1:>3}/{args.cycles}] {ok}  total={total_str}")
        cycles.append(c)

    summary = summarize(cycles)
    result = {"summary": summary, "cycles": cycles}

    out = RESULTS_DIR / "soar_validation.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== SOAR Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
