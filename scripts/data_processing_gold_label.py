"""
scripts/data_processing_gold_label.py - Gold label store DAG node.

Builds the label store (label = 1 if dpd >= 30 at mob = 6) for every cohort
month in the range. Per-month and independent, so it loops the range and calls
utils.data_processing_gold_table.process_labels_gold_table for each month.

Run from the repo root (inside the container, /opt/airflow):
    python3 scripts/data_processing_gold_label.py --start_date 2023-01-01 --end_date 2025-11-01
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyspark  # noqa: E402

import utils.data_processing_gold_table as gold  # noqa: E402


def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    dates = []
    current = datetime(start_date.year, start_date.month, 1)
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    return dates


def main(start_date, end_date, dpd, mob):
    print("\n--- gold_label: start ---")
    spark = (
        pyspark.sql.SparkSession.builder.appName("gold_label").master("local[*]").getOrCreate()
    )
    spark.conf.set("spark.sql.ansi.enabled", "false")
    spark.sparkContext.setLogLevel("ERROR")

    dates = generate_first_of_month_dates(start_date, end_date)
    gold_label_store_directory = "datamart/gold/label_store/"
    os.makedirs(gold_label_store_directory, exist_ok=True)
    silver_loan_daily_directory = os.path.join("datamart/silver/", "loan_daily")

    for date_str in dates:
        gold.process_labels_gold_table(
            date_str,
            silver_loan_daily_directory,
            gold_label_store_directory,
            spark,
            dpd=dpd,
            mob=mob,
        )

    spark.stop()
    print("--- gold_label: done ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the gold label store for a date range.")
    parser.add_argument("--start_date", type=str, default="2023-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, default="2025-11-01", help="YYYY-MM-DD")
    parser.add_argument("--dpd", type=int, default=30, help="days-past-due threshold")
    parser.add_argument("--mob", type=int, default=6, help="months-on-book observation point")
    args = parser.parse_args()
    main(args.start_date, args.end_date, args.dpd, args.mob)
