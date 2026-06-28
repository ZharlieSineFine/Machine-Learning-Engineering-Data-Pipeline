"""
scripts/model_train.py - Assignment 2 model training.

Trains several model types on the gold feature store, selects the best by
validation AUC, audits it out-of-time (OOT), and saves a self-contained
artefact (model + fitted preprocessing + feature list + metrics) to the model
bank for the inference step to load.

Run from the repo root (inside the container, /opt/airflow):
    python3 scripts/model_train.py --modelname credit_model.pkl

A few things that matter here:
* OOT is the last `--oot_months` months, and it has to match the feature
  store's `oot_months`. Gold fits its winsor/impute stats on
  `snapshot_date < train_cutoff`; a different boundary would let those stats
  see data from the model's future, i.e. train/test contamination.
* The train/validation split is temporal: the most recent non-OOT months are
  the validation set, so validation never contains the future.
* The artefact carries the fitted imputer, scaler, and exact feature order, so
  inference replays the same transform without re-fitting anything.
"""

import argparse
import os
import pickle
from datetime import datetime

import pandas as pd
import pyspark

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
import xgboost as xgb

GOLD_FEATURE_STORE = "datamart/gold/feature_store"
GOLD_LABEL_STORE = "datamart/gold/label_store"
MODEL_BANK_DIR = "model_bank"

# Keys / target / raw date columns that are never model features.
DROP_COLS = ["Customer_ID", "snapshot_date", "loan_start_date", "label", "label_def"]

SEED = 88


def _spark():
    spark = (
        pyspark.sql.SparkSession.builder.appName("model_train")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_training_frame(spark):
    """Join the gold feature + label stores into one pandas frame (+ month key)."""
    fs = spark.read.option("recursiveFileLookup", "true").parquet(GOLD_FEATURE_STORE)
    lb = (
        spark.read.option("recursiveFileLookup", "true")
        .parquet(GOLD_LABEL_STORE)
        .select("Customer_ID", "snapshot_date", "label")
    )
    pdf = fs.join(lb, on=["Customer_ID", "snapshot_date"], how="inner").toPandas()
    pdf["_month"] = pd.to_datetime(pdf["snapshot_date"]).dt.strftime("%Y-%m-%d")
    return pdf


def temporal_split(pdf, train_start, oot_months, val_fraction):
    """Split months into train / validation / OOT with no future leakage."""
    months = sorted(pdf["_month"].unique())
    if train_start:
        months = [m for m in months if m >= train_start]

    oot_months = min(oot_months, max(1, len(months) - 1))
    oot_cut = months[-oot_months]  # first OOT month == gold train_cutoff
    in_window = [m for m in months if m < oot_cut]

    n_val = max(1, int(round(len(in_window) * val_fraction)))
    val_months = set(in_window[-n_val:])
    train_months = set(in_window[:-n_val]) or set(in_window)  # never empty

    train = pdf[pdf["_month"].isin(train_months)]
    val = pdf[pdf["_month"].isin(val_months)]
    oot = pdf[pdf["_month"] >= oot_cut]
    meta = {
        "train_months": sorted(train_months),
        "val_months": sorted(val_months),
        "oot_cutoff": oot_cut,
        "oot_months_list": sorted(m for m in months if m >= oot_cut),
    }
    return train, val, oot, meta


def _xy(frame, feature_cols):
    X = frame[feature_cols].to_numpy(dtype="float64")
    y = frame["label"].to_numpy(dtype="int")
    return X, y


def _metrics(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    auc = float(roc_auc_score(y_true, proba))
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def train_candidates(x_train, y_train, scale_pos_weight):
    """Fit each candidate on scaled train data. Returns (models, xgb_best_params)."""
    models = {}

    models["logistic_regression"] = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=SEED
    ).fit(x_train, y_train)

    models["random_forest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    ).fit(x_train, y_train)

    xgb_base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        n_jobs=1,
    )
    param_dist = {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 5, 10],
        "reg_lambda": [1.0, 5.0, 10.0],
    }
    search = RandomizedSearchCV(
        xgb_base,
        param_distributions=param_dist,
        n_iter=30,
        scoring="roc_auc",
        cv=3,
        random_state=SEED,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    models["xgboost"] = search.best_estimator_
    return models, search.best_params_


def main(train_start, oot_months, val_fraction, modelname):
    print("\n--- model_train: start ---")
    spark = _spark()
    try:
        pdf = load_training_frame(spark)
    finally:
        spark.stop()
    print(
        f"loaded rows={len(pdf)} customers={pdf['Customer_ID'].nunique()} "
        f"months={pdf['_month'].nunique()}"
    )

    train, val, oot, meta = temporal_split(pdf, train_start, oot_months, val_fraction)
    print(f"train: rows={len(train)} months={len(meta['train_months'])}")
    print(f"val:   rows={len(val)} months={meta['val_months']}")
    print(f"oot:   rows={len(oot)} cutoff={meta['oot_cutoff']} months={meta['oot_months_list']}")

    feature_cols = [c for c in pdf.columns if c not in DROP_COLS and c != "_month"]
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(pdf[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns present: {non_numeric}")
    print(f"n_features={len(feature_cols)}")

    x_tr_raw, y_tr = _xy(train, feature_cols)
    x_va_raw, y_va = _xy(val, feature_cols)
    x_oot_raw, y_oot = _xy(oot, feature_cols)

    # Fit imputer + scaler on TRAIN ONLY (leakage-safe). The imputer covers the
    # occ_* one-hot nulls (gold leaves them null when Occupation is the missing
    # placeholder) plus any other residual nulls.
    imputer = SimpleImputer(strategy="median").fit(x_tr_raw)
    scaler = StandardScaler().fit(imputer.transform(x_tr_raw))

    def prep(x):
        return scaler.transform(imputer.transform(x))

    x_tr, x_va, x_oot = prep(x_tr_raw), prep(x_va_raw), prep(x_oot_raw)

    pos = int(y_tr.sum())
    neg = int((y_tr == 0).sum())
    scale_pos_weight = (neg / pos) if pos else 1.0
    print(f"train default_rate={y_tr.mean():.4f} scale_pos_weight={scale_pos_weight:.3f}")

    models, xgb_best_params = train_candidates(x_tr, y_tr, scale_pos_weight)

    # Per-model validation + OOT metrics (full table for the deck).
    per_model = {}
    print("\nmodel            | val_AUC  val_Gini | oot_AUC  oot_Gini")
    for name, mdl in models.items():
        vm = _metrics(y_va, mdl.predict_proba(x_va)[:, 1])
        om = _metrics(y_oot, mdl.predict_proba(x_oot)[:, 1])
        per_model[name] = {"val": vm, "oot": om}
        print(
            f"{name:16} | {vm['auc']:.4f}  {vm['gini']:.4f}  | "
            f"{om['auc']:.4f}  {om['gini']:.4f}"
        )

    best_name = max(per_model, key=lambda k: per_model[k]["val"]["auc"])
    best_model = models[best_name]
    print(
        f"\nselected: {best_name} "
        f"(val AUC={per_model[best_name]['val']['auc']:.4f}, "
        f"OOT AUC={per_model[best_name]['oot']['auc']:.4f})"
    )

    artefact = {
        "model": best_model,
        "model_type": best_name,
        "model_name": modelname,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "preprocessing": {
            "imputer": imputer,
            "scaler": scaler,
            "feature_cols": feature_cols,
        },
        "train_config": {
            "train_start": meta["train_months"][0] if meta["train_months"] else None,
            "train_end": meta["train_months"][-1] if meta["train_months"] else None,
            "val_months": meta["val_months"],
            "oot_cutoff": meta["oot_cutoff"],
            "oot_months": oot_months,
        },
        "metrics": {
            "per_model": per_model,
            "auc_val": per_model[best_name]["val"]["auc"],
            "auc_oot": per_model[best_name]["oot"]["auc"],
            "gini_oot": per_model[best_name]["oot"]["gini"],
        },
        "hyperparameters": (
            xgb_best_params if best_name == "xgboost" else best_model.get_params()
        ),
    }

    os.makedirs(MODEL_BANK_DIR, exist_ok=True)
    out_path = os.path.join(MODEL_BANK_DIR, modelname)
    with open(out_path, "wb") as f:
        pickle.dump(artefact, f)
    print(f"\nsaved artefact -> {out_path}")
    print("--- model_train: done ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train credit-default models and save the best to the model bank."
    )
    parser.add_argument(
        "--train_start",
        type=str,
        default=None,
        help="earliest snapshot month YYYY-MM-DD (default: earliest available)",
    )
    parser.add_argument(
        "--oot_months",
        type=int,
        default=2,
        help="most-recent months held out as OOT; must match the gold oot_months",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="fraction of the non-OOT months used as temporal validation",
    )
    parser.add_argument(
        "--modelname",
        type=str,
        default="credit_model.pkl",
        help="artefact filename written under model_bank/",
    )
    args = parser.parse_args()
    main(args.train_start, args.oot_months, args.val_fraction, args.modelname)
