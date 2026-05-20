"""Bi-LSTM on HDFS LogHub anomaly detection.

Parses raw logs with a Drain-lite template extractor, groups events
per BlockId, slices into 100-event windows, then trains a 2x128 Bi-LSTM
with early stopping. Long-running (~100 min on CPU).
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Bidirectional, Dense, Dropout, Embedding, GlobalAveragePooling1D, LSTM,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "hdfs"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 100
WINDOW_STRIDE = 50
EMBED_DIM = 64
LSTM_UNITS = 128
DROPOUT = 0.3
BATCH_SIZE = 256
EPOCHS = 30
PATIENCE = 5
RANDOM_STATE = 42

# Regex for Drain-lite template extraction
BLOCK_RE = re.compile(r"blk_-?\d+")
NUM_RE = re.compile(r"\b\d+\b")
IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
PATH_RE = re.compile(r"/[^\s,]+")


def download_dataset():
    """Get HDFS.log + anomaly_label.csv into DATA_DIR (Kaggle if missing)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_log = DATA_DIR / "HDFS.log"
    label_csv = DATA_DIR / "anomaly_label.csv"

    if raw_log.exists() and label_csv.exists():
        print(f"Using cached dataset at {DATA_DIR}")
        return raw_log, label_csv

    import kagglehub
    print("Downloading HDFS LogHub from Kaggle...")
    # try a couple of slugs, they get renamed sometimes
    candidates = [
        "omduggineni/loghub-hdfs",
        "logpai/loghub",
    ]
    src_root = None
    for slug in candidates:
        try:
            src_root = Path(kagglehub.dataset_download(slug))
            print(f"Downloaded from {slug}: {src_root}")
            break
        except Exception as e:
            print(f"  {slug} failed: {e}")
            continue
    if src_root is None:
        raise RuntimeError(
            "Could not download HDFS dataset. Manually download HDFS_v1 from "
            "https://github.com/logpai/loghub and place HDFS.log + anomaly_label.csv "
            f"in {DATA_DIR}"
        )

    import shutil
    for fname in ("HDFS.log", "anomaly_label.csv"):
        matches = list(src_root.rglob(fname))
        if matches:
            shutil.copy(matches[0], DATA_DIR / fname)
            print(f"  copied {fname}")

    if not (raw_log.exists() and label_csv.exists()):
        raise FileNotFoundError(
            f"Required files missing. Place HDFS.log and anomaly_label.csv in {DATA_DIR}"
        )
    return raw_log, label_csv


def normalize_message(msg: str) -> str:
    """Strip out the variable bits so we end up with a stable template id."""
    msg = BLOCK_RE.sub("BLK", msg)
    msg = IP_RE.sub("IP", msg)
    msg = HEX_RE.sub("HEX", msg)
    msg = PATH_RE.sub("PATH", msg)
    msg = NUM_RE.sub("NUM", msg)
    msg = re.sub(r"\s+", " ", msg).strip()
    return msg


def parse_logs_to_sessions(raw_log_path: Path):
    """Extract templates from the raw log and bucket events by BlockId."""
    print(f"Parsing {raw_log_path}...")
    template_to_id: dict[str, int] = {}
    sessions: dict[str, list[int]] = defaultdict(list)

    line_re = re.compile(
        r"^\d{6} \d{6} \d+ (?:INFO|WARN|ERROR|DEBUG) (?P<component>[^:]+): (?P<content>.+)$"
    )

    n_lines = 0
    n_with_block = 0
    with open(raw_log_path, "r", errors="ignore") as f:
        for line in tqdm(f, desc="Parsing"):
            n_lines += 1
            line = line.rstrip("\n")
            m = line_re.match(line)
            if not m:
                # fallback: grab everything after the level token
                parts = line.split(" ", 5)
                if len(parts) < 6:
                    continue
                content = parts[5]
            else:
                content = m.group("content")

            block_ids = BLOCK_RE.findall(content)
            if not block_ids:
                continue
            n_with_block += 1

            template = normalize_message(content)
            if template not in template_to_id:
                template_to_id[template] = len(template_to_id) + 1  # keep 0 for pad
            eid = template_to_id[template]

            for blk in block_ids:
                sessions[blk].append(eid)

    print(f"  lines: {n_lines:,}  |  with block: {n_with_block:,}")
    print(f"  unique blocks: {len(sessions):,}")
    print(f"  unique templates: {len(template_to_id):,}")
    return sessions, template_to_id


def window_blocks(block_ids, sessions, labels):
    """Chop each block's event sequence into 100-event windows.
    Returns X (list of int-lists), y (np int array), block_per_window (parallel list).
    """
    X, y, blocks = [], [], []
    for blk in block_ids:
        events = sessions.get(blk)
        if events is None:
            continue
        label = labels.get(blk)
        if label is None:
            continue
        if len(events) <= WINDOW_SIZE:
            X.append(events); y.append(label); blocks.append(blk)
        else:
            for start in range(0, len(events) - WINDOW_SIZE + 1, WINDOW_STRIDE):
                X.append(events[start:start + WINDOW_SIZE])
                y.append(label)
                blocks.append(blk)
    return X, np.array(y, dtype=np.int32), blocks


def build_model(vocab_size: int):
    model = Sequential([
        Embedding(input_dim=vocab_size + 1, output_dim=EMBED_DIM, mask_zero=True),
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, dropout=DROPOUT)),
        Bidirectional(LSTM(LSTM_UNITS, dropout=DROPOUT)),
        Dense(64, activation="relu"),
        Dropout(DROPOUT),
        Dense(2, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_from_preprocessed(traces_csv):
    """Fast path: load BlockId -> [event_ids] from the preprocessed Event_traces.csv
    (LogHub HDFS_v1 distribution). Skips the 1.5 GB raw HDFS.log download AND the
    parse phase. The Features column already contains the per-block template ID
    sequence as '[E5,E22,...]'. Produces the same per-block event-id sequences as
    parse_logs_to_sessions, just from a denser, faster source."""
    print(f"Using preprocessed traces from {traces_csv}")
    df = pd.read_csv(traces_csv, usecols=["BlockId", "Features"])
    template_to_id: dict[str, int] = {}
    sessions: dict[str, list[int]] = {}
    for blk, feats in zip(df["BlockId"].astype(str), df["Features"]):
        if not isinstance(feats, str):
            continue
        events = []
        for tok in feats.strip("[]").split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok not in template_to_id:
                template_to_id[tok] = len(template_to_id) + 1  # keep 0 for pad
            events.append(template_to_id[tok])
        sessions[blk] = events
    print(f"  blocks: {len(sessions):,}  |  templates: {len(template_to_id):,}")
    return sessions, template_to_id


def main():
    preproc = DATA_DIR / "preprocessed" / "Event_traces.csv"
    label_csv = DATA_DIR / "anomaly_label.csv"
    if preproc.exists() and label_csv.exists():
        sessions, template_to_id = load_from_preprocessed(preproc)
    else:
        raw_log, label_csv = download_dataset()
        sessions, template_to_id = parse_logs_to_sessions(raw_log)

    labels_df = pd.read_csv(label_csv)
    # expected cols: BlockId, Label (Label in {'Normal','Anomaly'})
    label_col = "Label" if "Label" in labels_df.columns else labels_df.columns[1]
    block_col = "BlockId" if "BlockId" in labels_df.columns else labels_df.columns[0]
    labels_df["y"] = (labels_df[label_col].str.lower() == "anomaly").astype(int)
    labels = dict(zip(labels_df[block_col], labels_df["y"]))
    print(f"Labels: {labels_df['y'].value_counts().to_dict()}")

    # === Split at the BlockId / session level so windows from the same block
    # never appear in more than one partition. The paper claims this; the
    # previous implementation split on the windows array, leaking. ===
    labeled_blocks = [b for b in sessions if b in labels]
    block_y = np.array([labels[b] for b in labeled_blocks])
    print(f"Sessions: {len(labeled_blocks):,}  |  anomalous: {int(block_y.sum()):,}  ({100*block_y.mean():.2f}%)")

    train_val_blocks, test_blocks = train_test_split(
        labeled_blocks, test_size=0.20, stratify=block_y, random_state=RANDOM_STATE
    )
    train_val_y = np.array([labels[b] for b in train_val_blocks])
    train_blocks, val_blocks = train_test_split(
        train_val_blocks, test_size=0.125, stratify=train_val_y, random_state=RANDOM_STATE
    )
    print(f"Block-level split: train={len(train_blocks):,} val={len(val_blocks):,} test={len(test_blocks):,}")

    # Window each partition INDEPENDENTLY. Within-block overlap (stride 50)
    # is augmentation in the partition that already owns the block; it is
    # leakage only when the same block straddles partitions, which is now
    # impossible by construction.
    X_train_list, y_train, _              = window_blocks(train_blocks, sessions, labels)
    X_val_list,   y_val,   _              = window_blocks(val_blocks, sessions, labels)
    X_test_list,  y_test,  test_blk_per_w = window_blocks(test_blocks, sessions, labels)
    print(f"Windows after split: train={len(X_train_list):,} val={len(X_val_list):,} test={len(X_test_list):,}")

    vocab_size = len(template_to_id)
    X_train = pad_sequences(X_train_list, maxlen=WINDOW_SIZE, padding="post", truncating="post")
    X_val   = pad_sequences(X_val_list,   maxlen=WINDOW_SIZE, padding="post", truncating="post")
    X_test  = pad_sequences(X_test_list,  maxlen=WINDOW_SIZE, padding="post", truncating="post")
    print(f"\nTrain: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    # weight the minority class (anomaly is ~5%)
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    total = n_neg + n_pos
    class_weight = {0: total / (2 * n_neg), 1: total / (2 * n_pos)}
    print(f"Class weights: {class_weight}")

    model = build_model(vocab_size)
    model.summary()

    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[es],
        verbose=2,
    )

    y_pred = model.predict(X_test, batch_size=BATCH_SIZE).argmax(axis=1)

    # === Window-level metrics (for completeness) ===
    win_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    win_cm = confusion_matrix(y_test, y_pred).tolist()

    # === Session-level aggregation (the paper's "anomalous sessions" claim) ===
    # A block is predicted anomaly if ANY of its windows is. Ground truth is
    # the block's label (all windows of a block share it by construction).
    block_pred = {}
    block_truth = {}
    for blk, pred in zip(test_blk_per_w, y_pred):
        block_pred[blk] = max(block_pred.get(blk, 0), int(pred))
        block_truth[blk] = labels[blk]
    ordered_blocks = sorted(block_pred)
    y_block_true = np.array([block_truth[b] for b in ordered_blocks])
    y_block_pred = np.array([block_pred[b] for b in ordered_blocks])
    sess_report = classification_report(y_block_true, y_block_pred, output_dict=True, zero_division=0)
    sess_cm = confusion_matrix(y_block_true, y_block_pred).tolist()

    n_test_sessions = int(len(ordered_blocks))
    n_test_anom = int(y_block_true.sum())
    n_test_anom_caught = int(((y_block_true == 1) & (y_block_pred == 1)).sum())

    metrics = {
        "_split_protocol": "block-level 70/10/20 stratified split (seed 42); windows of size 100, stride 50, generated INDEPENDENTLY within each partition so no BlockId straddles train/val/test",
        "vocab_size": int(vocab_size),
        # session-level counts (the paper's headline unit)
        "n_train_sessions": int(len(train_blocks)),
        "n_val_sessions": int(len(val_blocks)),
        "n_test_sessions": n_test_sessions,
        "n_test_anomalous_sessions": n_test_anom,
        "n_test_anomalous_sessions_recovered": n_test_anom_caught,
        # window-level counts (for the model)
        "n_train_windows": int(len(X_train)),
        "n_val_windows": int(len(X_val)),
        "n_test_windows": int(len(X_test)),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        # === HEADLINE: session-level metrics ===
        "test_accuracy": float(sess_report["accuracy"]),
        "test_precision_macro": float(sess_report["macro avg"]["precision"]),
        "test_recall_macro": float(sess_report["macro avg"]["recall"]),
        "test_f1_macro": float(sess_report["macro avg"]["f1-score"]),
        "per_class": {
            "0_normal": sess_report.get("0", {}),
            "1_anomaly": sess_report.get("1", {}),
        },
        "confusion_matrix": sess_cm,
        # window-level metrics (kept for completeness)
        "window_test_accuracy": float(win_report["accuracy"]),
        "window_test_f1_macro": float(win_report["macro avg"]["f1-score"]),
        "window_confusion_matrix": win_cm,
        "n_test": n_test_sessions,  # back-compat: 'n_test' now means SESSIONS
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }

    print("\n=== HDFS Bi-LSTM Results (session-level, block-grouped split) ===")
    print(f"  Test sessions          : {n_test_sessions:,}  ({n_test_anom:,} anomalous)")
    print(f"  Anomalies recovered    : {n_test_anom_caught}/{n_test_anom}")
    print(f"  Accuracy               : {metrics['test_accuracy']:.4f}")
    print(f"  Precision (macro)      : {metrics['test_precision_macro']:.4f}")
    print(f"  Recall    (macro)      : {metrics['test_recall_macro']:.4f}")
    print(f"  F1        (macro)      : {metrics['test_f1_macro']:.4f}")
    print(f"  [window-level F1 macro : {metrics['window_test_f1_macro']:.4f}]")

    with open(RESULTS_DIR / "hdfs_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {RESULTS_DIR / 'hdfs_metrics.json'}")

    model.save(RESULTS_DIR / "hdfs_lstm.keras")
    print(f"Model saved to: {RESULTS_DIR / 'hdfs_lstm.keras'}")


if __name__ == "__main__":
    main()
