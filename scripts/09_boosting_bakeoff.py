"""Boosting bake-off + simple baselines on NSL-KDD (Table 6).

Trains XGBoost / LightGBM / CatBoost / RandomForest plus two cheap
baselines (LogisticRegression, RandomForest) under the same canonical
KDDTrain+ -> KDDTest+ protocol and top-15 features as script 01, then
runs McNemar's test (XGBoost vs each other model) on the held-out
predictions. Writes results/boosting_bakeoff.json + nsl_kdd_baselines.json.
"""

import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "nsl_kdd"
RESULTS_DIR = PROJECT_DIR / "results"

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
CAT = ["protocol_type", "service", "flag"]
BEST = dict(max_depth=6, learning_rate=0.1, n_estimators=800, subsample=0.9, colsample_bytree=0.8)


def load(p):
    df = pd.read_csv(p, header=None, names=COLUMNS)
    df["y"] = (df["label"] != "normal").astype(int)
    return df


def prep():
    tr, te = load(DATA_DIR / "KDDTrain+.txt"), load(DATA_DIR / "KDDTest+.txt")
    for c in CAT:
        le = LabelEncoder().fit(pd.concat([tr[c], te[c]]).astype(str))
        tr[c] = le.transform(tr[c].astype(str)); te[c] = le.transform(te[c].astype(str))
    drop = ["label", "difficulty", "y"]
    Xtr_all, Xte_all = tr.drop(columns=drop), te.drop(columns=drop)
    ytr, yte = tr["y"].values, te["y"].values
    sel = SelectKBest(partial(mutual_info_classif, random_state=42), k=15).fit(Xtr_all, ytr)
    sc = StandardScaler().fit(sel.transform(Xtr_all))
    Xtr = sc.transform(sel.transform(Xtr_all))
    Xte = sc.transform(sel.transform(Xte_all))
    return Xtr, ytr, Xte, yte


def score(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    p, r, f, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    return {
        "test_accuracy": float(accuracy_score(yte, pred)),
        "test_precision_macro": float(p),
        "test_recall_macro": float(r),
        "test_f1_macro": float(f),
        "test_roc_auc": float(roc_auc_score(yte, proba)),
    }, pred


def mcnemar(truth, pred_a, pred_b):
    """McNemar with continuity correction. n01 = a right & b wrong; n10 = a wrong & b right."""
    a_ok, b_ok = (pred_a == truth), (pred_b == truth)
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) else 0.0
    return {"chi2": float(chi2), "p_value": float(chi2_dist.sf(chi2, 1)), "n01": n01, "n10": n10}


def main():
    Xtr, ytr, Xte, yte = prep()
    print(f"train {len(ytr):,}  test {len(yte):,}")

    models = {
        "XGBoost": XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                 tree_method="hist", random_state=42, n_jobs=-1, **BEST),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }
    # LightGBM / CatBoost left at library defaults (paper reports them as
    # "default hyperparameters"); only XGBoost is tuned.
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(random_state=42, verbose=-1)
    except ImportError:
        print("  (lightgbm not installed, skipping)")
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(verbose=0, random_seed=42)
    except ImportError:
        print("  (catboost not installed, skipping)")

    bakeoff, preds = {}, {}
    for name, m in models.items():
        bakeoff[name], preds[name] = score(m, Xtr, ytr, Xte, yte)
        print(f"  {name:14s} F1={bakeoff[name]['test_f1_macro']:.4f}  acc={bakeoff[name]['test_accuracy']:.4f}")

    for other in [n for n in models if n != "XGBoost"]:
        bakeoff[f"mcnemar_xgb_vs_{other.lower()}"] = mcnemar(yte, preds["XGBoost"], preds[other])

    with open(RESULTS_DIR / "boosting_bakeoff.json", "w") as f:
        json.dump({"NSL-KDD": bakeoff}, f, indent=2)

    # cheap baselines (Table 6 lower rows)
    base = {}
    lr = LogisticRegression(max_iter=300, class_weight="balanced", n_jobs=-1, random_state=42)
    base["LogisticRegression"], _ = score(lr, Xtr, ytr, Xte, yte)
    base["RandomForest"] = bakeoff["RandomForest"]
    with open(RESULTS_DIR / "nsl_kdd_baselines.json", "w") as f:
        json.dump(base, f, indent=2)
    print(f"\nWrote boosting_bakeoff.json + nsl_kdd_baselines.json")


if __name__ == "__main__":
    main()
