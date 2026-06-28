"""
scripts/model_monitor.py - Model monitoring DAG node.

Joins the stored predictions with ground-truth labels, computes per-month
performance + stability metrics, writes a monitoring gold table, and renders
PNG charts for the slide deck. Also flags drift against the artefact's OOT AUC.

Run from the repo root (inside the container, /opt/airflow):
    python3 scripts/model_monitor.py --modelname credit_model.pkl
"""

import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never to a display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pyspark  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

MODEL_BANK_DIR = "model_bank"
GOLD_LABEL_STORE = "datamart/gold/label_store"
PRED_BASE = "datamart/gold/model_predictions"
MONITOR_DIR = "datamart/gold/model_monitoring"
DRIFT_THRESHOLD = 0.05


def _month_metrics(g):
    y = g["label"].to_numpy(dtype="int")
    p = g["model_predictions"].to_numpy(dtype="float64")
    pred = (p >= 0.5).astype(int)
    auc = roc_auc_score(y, p) if len(set(y.tolist())) > 1 else None
    return pd.Series(
        {
            "n": len(g),
            "default_rate": float(y.mean()),
            "mean_pred": float(p.mean()),
            "auc": (float(auc) if auc is not None else None),
            "gini": (float(2 * auc - 1) if auc is not None else None),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
        }
    )


def main(modelname):
    print("\n--- model_monitor: start ---")
    stem = modelname[:-4] if modelname.endswith(".pkl") else modelname

    baseline_auc = None
    artefact_path = os.path.join(MODEL_BANK_DIR, modelname)
    if os.path.exists(artefact_path):
        with open(artefact_path, "rb") as f:
            baseline_auc = pickle.load(f).get("metrics", {}).get("auc_oot")

    spark = (
        pyspark.sql.SparkSession.builder.appName("model_monitor").master("local[*]").getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    preds = spark.read.option("recursiveFileLookup", "true").parquet(
        os.path.join(PRED_BASE, stem)
    )
    labels = (
        spark.read.option("recursiveFileLookup", "true")
        .parquet(GOLD_LABEL_STORE)
        .select("Customer_ID", "snapshot_date", "label")
    )
    joined = preds.join(labels, on=["Customer_ID", "snapshot_date"], how="inner").toPandas()
    spark.stop()

    if joined.empty:
        print("no joined prediction/label rows; nothing to monitor.")
        print("--- model_monitor: done ---\n")
        return

    joined["snapshot_date"] = pd.to_datetime(joined["snapshot_date"])
    joined["month"] = joined["snapshot_date"].dt.strftime("%Y-%m-%d")

    rows = []
    for m, g in joined.groupby("month"):
        s = _month_metrics(g)
        s["month"] = m
        rows.append(s)
    monitor = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    monitor.insert(0, "model_name", modelname)
    ordered = ["model_name", "month"] + [c for c in monitor.columns if c not in ("model_name", "month")]
    monitor = monitor[ordered]
    print(monitor.to_string(index=False))

    os.makedirs(MONITOR_DIR, exist_ok=True)
    charts_dir = os.path.join(MONITOR_DIR, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    table_path = os.path.join(MONITOR_DIR, f"{stem}_monitoring.parquet")
    monitor.to_parquet(table_path, index=False, engine="pyarrow")
    print(f"[monitor] table -> {table_path}")

    x = pd.to_datetime(monitor["month"])

    # Chart 1: performance over time (AUC + Gini).
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, monitor["auc"], marker="o", label="AUC")
    ax.plot(x, monitor["gini"], marker="s", label="Gini")
    if baseline_auc is not None:
        ax.axhline(baseline_auc, ls="--", color="grey", label=f"OOT baseline AUC={baseline_auc:.3f}")
        ax.axhline(baseline_auc - DRIFT_THRESHOLD, ls=":", color="red", label=f"drift floor (-{DRIFT_THRESHOLD})")
    ax.set_title(f"{stem}: performance over time")
    ax.set_xlabel("snapshot_date")
    ax.set_ylabel("score")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, f"{stem}_auc_over_time.png"), dpi=120)
    plt.close(fig)

    # Chart 2: actual default rate vs mean predicted prob (calibration check).
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, monitor["default_rate"], marker="o", label="actual default rate")
    ax.plot(x, monitor["mean_pred"], marker="s", label="mean predicted prob")
    ax.set_title(f"{stem}: default rate vs mean predicted")
    ax.set_xlabel("snapshot_date")
    ax.set_ylabel("rate")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, f"{stem}_calibration_over_time.png"), dpi=120)
    plt.close(fig)

    # Chart 3: prediction distribution, earliest vs latest month.
    months = sorted(joined["month"].unique())
    first_m, last_m = months[0], months[-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = [i / 10 for i in range(11)]
    ax.hist(joined.loc[joined["month"] == first_m, "model_predictions"], bins=bins, alpha=0.5, density=True, label=f"first ({first_m})")
    ax.hist(joined.loc[joined["month"] == last_m, "model_predictions"], bins=bins, alpha=0.5, density=True, label=f"latest ({last_m})")
    ax.set_title(f"{stem}: prediction distribution shift")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, f"{stem}_prediction_distribution.png"), dpi=120)
    plt.close(fig)
    print(f"[monitor] charts -> {charts_dir}")

    # Drift check against the honest OOT baseline (not the inflated train AUC).
    recent = monitor["auc"].dropna()
    if baseline_auc is not None and len(recent):
        latest = float(recent.iloc[-1])
        if latest < baseline_auc - DRIFT_THRESHOLD:
            print(
                f"DRIFT DETECTED: latest AUC={latest:.3f} is >{DRIFT_THRESHOLD} below "
                f"OOT baseline={baseline_auc:.3f}. Recommend retraining."
            )
        else:
            print(f"no drift: latest AUC={latest:.3f} vs OOT baseline={baseline_auc:.3f}")

    print("--- model_monitor: done ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor model performance and stability over time.")
    parser.add_argument(
        "--modelname",
        type=str,
        default="credit_model.pkl",
        help="artefact filename under model_bank/ (also names the predictions subfolder)",
    )
    args = parser.parse_args()
    main(args.modelname)
