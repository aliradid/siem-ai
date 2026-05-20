"""GUIDE 3-class triage (Benign / Suspicious / Malicious) with XGBoost.

Heads up: the Kaggle download is ~5-8 GB and training takes ~50 min
on Apple Silicon.
"""

import gc
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "guide"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction
TARGET_COL = "IncidentGrade"

# IDs, timestamps, anything leaky or high-cardinality
DROP_COLS = [
    "Id", "OrgId", "IncidentId", "AlertId", "DetectorId",
    "Timestamp",
    "Sha256", "IpAddress", "Url", "AccountSid", "AccountUpn",
    "AccountObjectId", "AccountName", "NetworkMessageId", "EmailClusterId",
    "RegistryKey", "RegistryValueName", "RegistryValueData",
    "ApplicationId", "ApplicationName", "OAuthApplicationId",
    "FileName", "FolderPath", "ResourceIdName", "DeviceId", "DeviceName",
    "ThreatFamily",  # huge cardinality
    "City", "State",  # leaky on some classes
    # Post-hoc SOC analyst decisions — using them as features is label leakage
    # because they encode actions/verdicts only available AFTER triage.
    "ActionGrouped", "ActionGranular", "LastVerdict", "SuspicionLevel",
    # Kaggle split marker (Public/Private), not a detection feature.
    "Usage",
]

CATEGORICAL_COLS = [
    "Category", "MitreTechniques",
    "EntityType", "EvidenceRole", "Roles",
    "ResourceType", "OSFamily", "OSVersion", "AntispamDirection",
    "CountryCode",
]


def download_dataset() -> tuple[Path, Path]:
    """Locate (or download) Microsoft's canonical GUIDE Train and Test CSVs.
    Returns (train_csv, test_csv). The Microsoft GUIDE distribution ships a
    PRE-DEFINED partition (GUIDE_Train.csv / GUIDE_Test.csv) and the published
    benchmark numbers are evaluated on Test — random splits of Train are not
    comparable to the literature."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def find_pair(root: Path):
        cs = list(root.rglob("*.csv"))
        tr = next((p for p in cs if p.name.lower().endswith("guide_train.csv")), None)
        te = next((p for p in cs if p.name.lower().endswith("guide_test.csv")), None)
        return tr, te

    tr, te = find_pair(DATA_DIR)
    if tr and te:
        print(f"Using cached dataset:\n  train={tr}\n  test ={te}")
        return tr, te

    import kagglehub
    print("Downloading Microsoft GUIDE from Kaggle (this is large, ~5-8 GB)...")
    path = kagglehub.dataset_download("Microsoft/microsoft-security-incident-prediction")
    src_root = Path(path)
    tr, te = find_pair(src_root)
    if not (tr and te):
        raise FileNotFoundError(
            f"Could not locate GUIDE_Train.csv + GUIDE_Test.csv in {src_root}"
        )
    print(f"Using:\n  train={tr}\n  test ={te}")
    return tr, te


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  rows: {len(df):,}, cols: {df.shape[1]}")

    df = df.dropna(subset=[TARGET_COL])
    print(f"  rows with label: {len(df):,}")

    keep_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=keep_drop)

    # Keep Microsoft's CANONICAL label set: BenignPositive / FalsePositive /
    # TruePositive. These describe the SOC outcome (not severity); calling
    # FalsePositive "Suspicious" or TruePositive "Malicious" is incorrect
    # because it changes the operational meaning.
    label_map = {
        "BenignPositive": 0,
        "FalsePositive":  1,
        "TruePositive":   2,
    }
    df[TARGET_COL] = df[TARGET_COL].map(label_map)
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    print(f"  class distribution: {df[TARGET_COL].value_counts().to_dict()}")
    return df


def fit_categorical_encoders(df: pd.DataFrame) -> dict:
    """Fit LabelEncoder on TRAIN only. Returns {col: (encoder, unk_code)}.
    Unseen categories in test/val get the reserved UNK code (= len(classes_))."""
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        le = LabelEncoder().fit(df[col].fillna("UNK").astype(str))
        encoders[col] = (le, len(le.classes_))
    # any other object-typed column → category codes fitted on train only
    object_cols = list(df.select_dtypes(include=["object"]).columns)
    return encoders, object_cols


def apply_categorical_encoders(df: pd.DataFrame, encoders: dict, object_cols: list) -> pd.DataFrame:
    """Apply train-fitted encoders to a (train, val, or test) frame. Unseen
    categories map to UNK rather than crashing or being treated as a new code
    fit on the test set."""
    for col, (le, unk_code) in encoders.items():
        if col not in df.columns:
            continue
        s = df[col].fillna("UNK").astype(str)
        # vectorized: map known classes to their codes, unseen -> UNK code
        mapping = {c: i for i, c in enumerate(le.classes_)}
        df[col] = s.map(mapping).fillna(unk_code).astype("int32")
    # Defensive: coerce ANY remaining object column in this frame to codes,
    # so a stray object-typed column present in test but not train (e.g. a
    # marker column) can never reach XGBoost as dtype=object.
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("UNK").astype("category").cat.codes
    return df


def main():
    train_csv, test_csv = download_dataset()
    df_train_full = load_and_clean(train_csv)
    df_test       = load_and_clean(test_csv)

    # Fit encoders on TRAIN ONLY, apply to both.
    encoders, object_cols = fit_categorical_encoders(df_train_full)
    df_train_full = apply_categorical_encoders(df_train_full, encoders, object_cols)
    df_test       = apply_categorical_encoders(df_test,       encoders, object_cols)

    y_train_full = df_train_full[TARGET_COL].values
    X_train_full = df_train_full.drop(columns=[TARGET_COL]).fillna(0)
    y_test       = df_test[TARGET_COL].values
    X_test       = df_test.drop(columns=[TARGET_COL]).fillna(0)

    # Carve a small validation slice out of TRAIN (12.5% -> ~1.19M val rows
    # from a ~9.5M-row Train; the held-out Microsoft Test remains untouched).
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.125, stratify=y_train_full, random_state=42
    )
    print(f"\nTrain: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}  "
          f"(Microsoft canonical GUIDE_Train -> GUIDE_Test split)")

    del df_train_full, df_test, X_train_full, y_train_full
    gc.collect()

    print("\nTraining XGBoost (early stopping after 250 rounds w/o improvement)...")
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        n_estimators=3000,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=250,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    print(f"\nBest iteration: {model.best_iteration}")

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "best_iteration": int(model.best_iteration),
        "test_accuracy": float(report["accuracy"]),
        "test_precision_macro": float(report["macro avg"]["precision"]),
        "test_recall_macro": float(report["macro avg"]["recall"]),
        "test_f1_macro": float(report["macro avg"]["f1-score"]),
        "test_precision_weighted": float(report["weighted avg"]["precision"]),
        "test_recall_weighted": float(report["weighted avg"]["recall"]),
        "test_f1_weighted": float(report["weighted avg"]["f1-score"]),
        "confusion_matrix": cm,
        "per_class": {
            "0_BenignPositive": report.get("0", {}),
            "1_FalsePositive":  report.get("1", {}),
            "2_TruePositive":   report.get("2", {}),
        },
    }

    print("\n=== GUIDE Results ===")
    print(f"  Accuracy : {metrics['test_accuracy']:.4f}")
    print(f"  Precision (macro) : {metrics['test_precision_macro']:.4f}")
    print(f"  Recall    (macro) : {metrics['test_recall_macro']:.4f}")
    print(f"  F1        (macro) : {metrics['test_f1_macro']:.4f}")
    for label, scores in metrics["per_class"].items():
        if scores:
            print(f"    {label}: P={scores['precision']:.3f}  R={scores['recall']:.3f}  F1={scores['f1-score']:.3f}  support={int(scores['support'])}")

    with open(RESULTS_DIR / "guide_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {RESULTS_DIR / 'guide_metrics.json'}")

    joblib.dump(model, RESULTS_DIR / "guide_xgb.joblib")
    print(f"Model saved to: {RESULTS_DIR / 'guide_xgb.joblib'}")


if __name__ == "__main__":
    main()
