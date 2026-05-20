"""Three simple baselines on HDFS event-occurrence vectors, so we have
something to compare the Bi-LSTM in script 03 against: logistic regression,
random forest, and an isolation forest trained on normals only.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_DIR / "results"
CSV = PROJECT_DIR / "data" / "hdfs" / "preprocessed" / "Event_occurrence_matrix.csv"


def main():
    print(f"Loading {CSV}...")
    df = pd.read_csv(CSV)
    print(f"  rows: {len(df):,}, cols: {df.shape[1]}")

    # 1 = anomaly, 0 = normal
    y = (df["Label"].str.lower() == "fail").astype(int).values
    print(f"  class distribution: {pd.Series(y).value_counts().to_dict()}")

    feat_cols = [c for c in df.columns if c.startswith("E") and c[1:].isdigit()]
    print(f"  using {len(feat_cols)} template-count features")
    X = df[feat_cols].fillna(0).values.astype(np.float32)

    # same 70/10/20 split as the Bi-LSTM run, so numbers are comparable
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42
    )
    print(f"  train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # 1. Logistic Regression
    print("\n[1/3] Logistic Regression ...")
    lr = LogisticRegression(max_iter=300, n_jobs=-1, class_weight="balanced",
                            random_state=42)
    lr.fit(X_train_s, y_train)
    y_pred = lr.predict(X_test_s)
    y_proba = lr.predict_proba(X_test_s)[:, 1]
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="macro",
                                                  zero_division=0)
    results["LogisticRegression"] = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision_macro": float(p),
        "test_recall_macro": float(r),
        "test_f1_macro": float(f),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    for k, v in results["LogisticRegression"].items():
        print(f"    {k}: {v:.4f}")

    # 2. Random Forest
    print("\n[2/3] Random Forest (200 trees) ...")
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42,
                                 class_weight="balanced")
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="macro",
                                                  zero_division=0)
    results["RandomForest"] = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision_macro": float(p),
        "test_recall_macro": float(r),
        "test_f1_macro": float(f),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    for k, v in results["RandomForest"].items():
        print(f"    {k}: {v:.4f}")

    # 3. Isolation Forest, trained on normals only (unsupervised baseline)
    print("\n[3/3] Isolation Forest (unsupervised, normals only) ...")
    if_clf = IsolationForest(n_estimators=100, contamination=0.03,
                              random_state=42, n_jobs=-1)
    if_clf.fit(X_train_s[y_train == 0])
    score = -if_clf.score_samples(X_test_s)  # higher = more anomalous
    threshold = np.quantile(score, 0.97)  # match contamination
    y_pred = (score >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="macro",
                                                  zero_division=0)
    results["IsolationForest_unsupervised"] = {
        "note": "Trained on normals only; threshold at 97th percentile of test scores",
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision_macro": float(p),
        "test_recall_macro": float(r),
        "test_f1_macro": float(f),
        "test_roc_auc": float(roc_auc_score(y_test, score)),
    }
    for k, v in results["IsolationForest_unsupervised"].items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    out = RESULTS / "hdfs_baselines.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
