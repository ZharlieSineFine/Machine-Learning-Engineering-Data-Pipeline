"""
scripts/model_inference.py - Model inference DAG node.

Loads the trained artefact from the model bank, scores the gold feature store,
and writes per-month prediction partitions to the gold model_predictions store.

By default it scores EVERY month in the feature store (so the monitoring step
can show behaviour across time). Pass --snapshotdate to score a single month.

Run from the repo root (inside the container, /opt/airflow):
    python3 scripts/model_inference.py --modelname credit_model.pkl
"""

import argparse
import os
import pickle

import pandas as pd
import pyspark
from pyspark.sql.functions import col

MODEL_BANK_DIR = "model_bank"
GOLD_FEATURE_STORE = "datamart/gold/feature_store"
PRED_BASE = "datamart/gold/model_predictions"


def main(snapshotdate, modelname):
    print("\n--- model_inference: start ---")
    artefact_path = os.path.join(MODEL_BANK_DIR, modelname)
    with open(artefact_path, "rb") as f:
        artefact = pickle.load(f)
    model = artefact["model"]
    pp = artefact["preprocessing"]
    imputer, scaler, feature_cols = pp["imputer"], pp["scaler"], pp["feature_cols"]
    print(f"loaded {artefact_path} (type={artefact.get('model_type')}, n_features={len(feature_cols)})")

    spark = (
        pyspark.sql.SparkSession.builder.appName("model_inference").master("local[*]").getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    fs = spark.read.option("recursiveFileLookup", "true").parquet(GOLD_FEATURE_STORE)
    if snapshotdate:
        fs = fs.filter(col("snapshot_date") == snapshotdate)
    pdf = fs.toPandas()
    spark.stop()

    if pdf.empty:
        print(f"no feature rows for snapshotdate={snapshotdate}; nothing to score.")
        print("--- model_inference: done ---\n")
        return

    # Identical transform to training: impute (median) -> scale, same column order.
    X = pdf[feature_cols].to_numpy(dtype="float64")
    proba = model.predict_proba(scaler.transform(imputer.transform(X)))[:, 1]

    out = pdf[["Customer_ID", "snapshot_date"]].copy()
    # Store snapshot_date as a date (date32), NOT a pandas datetime64[ns]. pandas/pyarrow
    # would otherwise write TIMESTAMP(NANOS), which Spark (model_monitor) cannot read
    # ("Illegal Parquet type INT64 TIMESTAMP(NANOS)"). date32 also matches the gold label
    # store's DateType, so the monitor join on (Customer_ID, snapshot_date) still works.
    ts = pd.to_datetime(out["snapshot_date"])
    out["snapshot_date"] = ts.dt.date
    out["model_name"] = modelname
    out["model_predictions"] = proba

    stem = modelname[:-4] if modelname.endswith(".pkl") else modelname
    gold_dir = os.path.join(PRED_BASE, stem)
    os.makedirs(gold_dir, exist_ok=True)

    # One parquet partition per scored month (small data -> pandas/pyarrow write,
    # robust on both Windows and Docker; read back via recursiveFileLookup).
    out["_m"] = ts.dt.strftime("%Y-%m-%d")  # use parsed datetime; snapshot_date is now date
    for m in sorted(out["_m"].unique()):
        part = out[out["_m"] == m].drop(columns="_m")
        fpath = os.path.join(gold_dir, f"{stem}_predictions_{m.replace('-', '_')}.parquet")
        part.to_parquet(fpath, index=False, engine="pyarrow")
        print(f"[inference] {m} rows={len(part)} -> {fpath}")

    print("--- model_inference: done ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the feature store with a trained model.")
    parser.add_argument(
        "--snapshotdate",
        type=str,
        default=None,
        help="YYYY-MM-DD to score one month; omit to score all months",
    )
    parser.add_argument(
        "--modelname",
        type=str,
        default="credit_model.pkl",
        help="artefact filename under model_bank/",
    )
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)
