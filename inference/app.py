"""Tiny Flask service that loads the trained XGBoost model and exposes
POST /predict. Logstash calls it for every event in the pipeline.
"""

import os
import time
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/models/nsl_kdd_xgb.joblib"))
SCALER_PATH = Path(os.environ.get("SCALER_PATH", "/app/models/nsl_kdd_scaler.joblib"))

app = Flask(__name__)
model = None
scaler = None


def load_scaler():
    """Load the StandardScaler fit during training. The model is trained on
    scaled features, so serving must apply the same transform or the
    predictions are meaningless. A missing scaler is a hard fault (symmetric
    with load_model), not a silent skip."""
    global scaler
    if scaler is None:
        if not SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Scaler not found at {SCALER_PATH}. Train it first with "
                "scripts/01_train_nsl_kdd.py and mount results/ into the container."
            )
        scaler = joblib.load(SCALER_PATH)
        app.logger.info(f"Loaded scaler from {SCALER_PATH}")
    return scaler


def load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train it first with "
                "scripts/01_train_nsl_kdd.py and mount results/ into the container."
            )
        model = joblib.load(MODEL_PATH)
        app.logger.info(f"Loaded model from {MODEL_PATH}")
    return model


@app.route("/healthz", methods=["GET"])
def healthz():
    try:
        load_model()
        load_scaler()
        return jsonify({"status": "ok", "model": str(MODEL_PATH), "scaler": str(SCALER_PATH)})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    start = time.perf_counter()
    payload = request.get_json(force=True, silent=True) or {}

    # accept a few shapes here because Logstash and the bench send
    # things slightly differently:
    #   {"features": [v1, v2, ...]}            list / JSON array
    #   {"features": "v1,v2,...,vN"}           CSV string (Logstash sprintf)
    #   {"features": {"k1": v1, "k2": v2}}     dict of named features
    #   {k1: v1, k2: v2, ...}                  flat top-level numeric dict
    feats = payload.get("features")
    if isinstance(feats, str):
        try:
            feats = [float(x) for x in feats.strip("[]").split(",") if x.strip() not in ("", "null", "nil")]
        except ValueError as e:
            return jsonify({"error": f"could not parse CSV features: {e}"}), 400
    if isinstance(feats, dict):
        feats = [v for _, v in sorted(feats.items()) if isinstance(v, (int, float))]
    if feats is None:
        feats = [v for k, v in sorted(payload.items()) if isinstance(v, (int, float))]

    if not feats:
        return jsonify({"error": "no numeric features supplied", "payload_keys": list(payload.keys())}), 400

    try:
        X = np.asarray(feats, dtype=np.float32).reshape(1, -1)
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"could not coerce features to numeric vector: {e}"}), 400
    try:
        m = load_model()
        sc = load_scaler()
        X = sc.transform(X)
        proba = m.predict_proba(X)[0]
        pred = int(np.argmax(proba))
        score = float(proba[pred])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    label_map = payload.get("_label_map") or {0: "benign", 1: "malicious"}
    # JSON-supplied maps have string keys; accept either int or str so callers
    # like scripts/05_soar_validation.py {"0": "benign", "1": "malicious"} work
    label = label_map.get(pred) or label_map.get(str(pred)) or str(pred)

    latency_ms = (time.perf_counter() - start) * 1000.0
    return jsonify({"label": label, "score": score, "latency_ms": latency_ms})


if __name__ == "__main__":
    # local: python app.py
    app.run(host="0.0.0.0", port=5000, debug=False)
