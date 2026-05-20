"""XGBoost on NSL-KDD, binary normal vs attack.

Uses the canonical benchmark protocol: train (with CV grid search) on the
full KDDTrain+ set, then evaluate once on the held-out KDDTest+ set.
KDDTest+ contains 17 attack types that never appear in KDDTrain+, so the
test score (~0.77 F1) is far below the in-distribution CV score (~0.99) --
that gap is the whole point and the reason a random split of KDDTrain+
alone would massively overstate accuracy.
"""

import json
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "nsl_kdd"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]
CAT_COLS = ["protocol_type", "service", "flag"]


def download_dataset():
    """Make sure KDDTrain+.txt and KDDTest+.txt are in DATA_DIR (Kaggle if missing)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_file = DATA_DIR / "KDDTrain+.txt"
    test_file = DATA_DIR / "KDDTest+.txt"
    if train_file.exists() and test_file.exists():
        print(f"Using cached dataset at {DATA_DIR}")
        return train_file, test_file

    import kagglehub
    import shutil
    print("Downloading NSL-KDD from Kaggle...")
    src_root = Path(kagglehub.dataset_download("hassan06/nslkdd"))
    for name in ("KDDTrain+.txt", "KDDTest+.txt"):
        hits = list(src_root.rglob(name))
        if hits:
            shutil.copy(hits[0], DATA_DIR / name)
            print(f"  copied {name}")
    if not (train_file.exists() and test_file.exists()):
        raise FileNotFoundError(f"KDDTrain+.txt / KDDTest+.txt missing in {DATA_DIR}")
    return train_file, test_file


def load(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["y"] = (df["label"] != "normal").astype(int)
    return df


def main():
    train_file, test_file = download_dataset()
    train_df = load(train_file)
    test_df = load(test_file)
    print(f"KDDTrain+: {len(train_df):,} rows  |  KDDTest+: {len(test_df):,} rows")

    # encode the three categoricals; fit on train+test vocab so unseen
    # test services (KDDTest+ has a few) don't blow up transform
    for col in CAT_COLS:
        le = LabelEncoder().fit(pd.concat([train_df[col], test_df[col]]).astype(str))
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))

    drop = ["label", "difficulty", "y"]
    X_train_full = train_df.drop(columns=drop)
    X_test_full = test_df.drop(columns=drop)
    y_train = train_df["y"].values
    y_test = test_df["y"].values

    # top 15 features by mutual information, fit on train only
    print("\nSelecting top 15 features (mutual information, train only)...")
    selector = SelectKBest(partial(mutual_info_classif, random_state=42), k=15).fit(X_train_full, y_train)
    selected = X_train_full.columns[selector.get_support()].tolist()
    print(f"Selected: {selected}")

    scaler = StandardScaler().fit(selector.transform(X_train_full))
    X_train = scaler.transform(selector.transform(X_train_full))
    X_test = scaler.transform(selector.transform(X_test_full))

    print("\nGrid search (3-fold CV on KDDTrain+)...")
    param_grid = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.5, 0.8, 0.9],
        "colsample_bytree": [0.8],
        "n_estimators": [128, 800],
    }
    base = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=42, n_jobs=-1,
    )
    grid = GridSearchCV(base, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV F1 on train: {grid.best_score_:.4f}")

    # evaluate the chosen model ONCE on the held-out KDDTest+
    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = float(roc_auc_score(y_test, y_proba))

    metrics = {
        "protocol": "KDDTrain+ -> KDDTest+ (published NSL-KDD benchmark)",
        "best_params": grid.best_params_,
        "best_cv_f1_on_train": float(grid.best_score_),
        "test_accuracy": float(report["accuracy"]),
        "test_precision_macro": float(report["macro avg"]["precision"]),
        "test_recall_macro": float(report["macro avg"]["recall"]),
        "test_f1_macro": float(report["macro avg"]["f1-score"]),
        "test_roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "per_class": {"0_normal": report["0"], "1_attack": report["1"]},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "selected_features": selected,
    }

    print("\n=== KDDTest+ results (canonical protocol) ===")
    print(f"  Accuracy : {metrics['test_accuracy']:.4f}")
    print(f"  Precision: {metrics['test_precision_macro']:.4f}")
    print(f"  Recall   : {metrics['test_recall_macro']:.4f}")
    print(f"  F1 macro : {metrics['test_f1_macro']:.4f}")
    print(f"  ROC-AUC  : {metrics['test_roc_auc']:.4f}")

    out = RESULTS_DIR / "nsl_kdd_metrics_proper.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics -> {out}")

    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title(f"NSL-KDD (KDDTest+) ROC, AUC = {roc_auc:.4f}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "nsl_kdd_roc.png", dpi=150)
    joblib.dump(model, RESULTS_DIR / "nsl_kdd_xgb.joblib")
    joblib.dump(scaler, RESULTS_DIR / "nsl_kdd_scaler.joblib")
    print("Saved ROC plot + model + scaler.")


if __name__ == "__main__":
    main()
