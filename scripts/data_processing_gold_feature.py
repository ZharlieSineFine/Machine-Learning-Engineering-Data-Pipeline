"""
scripts/data_processing_gold_feature.py - Gold feature store DAG node.

Builds the ML-ready feature store. This is a GLOBAL step: winsor caps (p99) and
median imputation are fit on the train window (snapshot_date < train_cutoff,
where train_cutoff = the first of the last `--oot_months` cohort months), so it
must run after the full silver history and the label store exist. Thin CLI
wrapper over utils.data_processing_gold_table.build_feature_store.

Run from the repo root (inside the container, /opt/airflow):
    python3 scripts/data_processing_gold_feature.py --oot_months 2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyspark  # noqa: E402

import utils.data_processing_gold_table as gold  # noqa: E402


def main(oot_months):
    print("\n--- gold_feature: start ---")
    spark = (
        pyspark.sql.SparkSession.builder.appName("gold_feature").master("local[*]").getOrCreate()
    )
    spark.conf.set("spark.sql.ansi.enabled", "false")
    spark.sparkContext.setLogLevel("ERROR")

    gold_feature_store_directory = "datamart/gold/feature_store/"
    os.makedirs(gold_feature_store_directory, exist_ok=True)

    gold.build_feature_store(
        "datamart/silver/",
        "datamart/gold/label_store/",
        gold_feature_store_directory,
        spark,
        oot_months=oot_months,
    )

    spark.stop()
    print("--- gold_feature: done ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the gold feature store (global train-window fit).")
    parser.add_argument(
        "--oot_months",
        type=int,
        default=2,
        help="months reserved for OOT; sets train_cutoff for winsor/impute fit",
    )
    args = parser.parse_args()
    main(args.oot_months)
