"""Re-run the same XGBoost recipe as script 01, but on UNSW-NB15.
The point is to show the recipe isn't overfit to NSL-KDD.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, precision_recall_fscore_support,
                              roc_auc_score)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA = PROJECT_DIR / "data" / "unsw_nb15"
RESULTS = PROJECT_DIR / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def download() -> tuple[Path, Path]:
    DATA.mkdir(parents=True, exist_ok=True)
    train = DATA / "UNSW_NB15_training-set.csv"
    test = DATA / "UNSW_NB15_testing-set.csv"
    if train.exists() and test.exists():
        print(f"Using cached dataset at {DATA}")
        return train, test
    import kagglehub
    print("Downloading UNSW-NB15 from Kaggle...")
    path = Path(kagglehub.dataset_download("mrwellsdavid/unsw-nb15"))
    # copy just the two split CSVs we actually need
    import shutil
    for name in ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"):
        src = path / name
        if src.exists():
            shutil.copy(src, DATA / name)
    return DATA / "UNSW_NB15_training-set.csv", DATA / "UNSW_NB15_testing-set.csv"


def main():
    csv_a, csv_b = download()
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)
    # The Kaggle "mrwellsdavid" mirror has the canonical filenames SWAPPED
    # relative to Moustafa & Slay (2015), which defines train=175 341 and
    # test=82 332. Assign by row count so we always match the canonical
    # protocol regardless of which file the mirror calls "training-set".
    if len(df_a) > len(df_b):
        train, test = df_a, df_b
    else:
        train, test = df_b, df_a
    assert len(train) > len(test), "canonical UNSW-NB15: train > test"
    print(f"Train: {train.shape}, Test: {test.shape}  (canonical Moustafa split: 175 341 / 82 332)")
    print(f"Train columns: {list(train.columns)}")
    print(f"Label distribution (train): {train['label'].value_counts().to_dict()}")

    # drop id and the multiclass label; keep the binary `label`
    drop_cols = [c for c in ("id", "attack_cat") if c in train.columns]
    train = train.drop(columns=drop_cols)
    test = test.drop(columns=drop_cols)

    # proto, service, state are categorical. Fit the encoder on TRAIN ONLY
    # (vocabulary fitted only on training data); map any unseen test category
    # to a reserved UNK index. This matches the paper's [081] statement and
    # avoids the vocabulary-leakage of fitting on the train+test union.
    cat_cols = [c for c in train.select_dtypes(include="object").columns]
    print(f"Categorical columns: {cat_cols}")
    for col in cat_cols:
        le = LabelEncoder().fit(train[col].astype(str))
        known = set(le.classes_)
        train[col] = le.transform(train[col].astype(str))
        # Unseen test categories -> code len(classes_) (a fresh UNK token);
        # XGBoost handles unseen integer codes as just another value.
        test_vals = test[col].astype(str)
        unk_code = len(le.classes_)
        test[col] = [le.transform([v])[0] if v in known else unk_code for v in test_vals]

    y_train = train.pop("label").values
    y_test = test.pop("label").values
    X_train = train.values
    X_test = test.values

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # same XGBoost pipeline as scripts/01; depth and subsample re-set for this
    # corpus (max_depth 8 / subsample 0.8 vs 6 / 0.9 on NSL-KDD)
    model = XGBClassifier(
        objective="binary:logistic",
        max_depth=8,
        learning_rate=0.1,
        n_estimators=800,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    print("\nTraining XGBoost ...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "best_params": {
            "max_depth": 8, "learning_rate": 0.1, "n_estimators": 800,
            "subsample": 0.8, "colsample_bytree": 0.8,
        },
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_accuracy": float(acc),
        "test_precision_macro": float(p),
        "test_recall_macro": float(r),
        "test_f1_macro": float(f),
        "test_roc_auc": float(auc),
        "confusion_matrix": cm,
        "per_class": {
            "0_normal": report.get("0", {}),
            "1_attack": report.get("1", {}),
        },
    }

    print("\n=== UNSW-NB15 Results ===")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision (mac) : {p:.4f}")
    print(f"  Recall    (mac) : {r:.4f}")
    print(f"  F1        (mac) : {f:.4f}")
    print(f"  ROC AUC         : {auc:.4f}")
    print(f"  Confusion matrix:\n{np.array(cm)}")

    out = RESULTS / "unsw_nb15_metrics.json"
    json.dump(metrics, open(out, "w"), indent=2)
    print(f"\nSaved {out}")
    joblib.dump(model, RESULTS / "unsw_nb15_xgb.joblib")
    print(f"Saved {RESULTS / 'unsw_nb15_xgb.joblib'}")


if __name__ == "__main__":
    main()
