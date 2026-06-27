"""
scripts/data_processing_silver.py - Silver layer DAG node.

Cleans each bronze partition (type casts, sentinel nulls, mob/dpd on loan_daily)
into silver parquet for every month in the range. Thin CLI wrapper over
utils.data_processing_silver_table.

Run from the repo root (inside the container, /opt/airflow):
    python3 scripts/data_processing_silver.py --start_date 2023-01-01 --end_date 2025-11-01
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyspark  # noqa: E402

import utils.data_processing_bronze_table as bronze  # noqa: E402  (SOURCE_CONFIG registry)
import utils.data_processing_silver_table as silver  # noqa: E402


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


def main(start_date, end_date):
    print("\n--- silver: start ---")
    spark = (
        pyspark.sql.SparkSession.builder.appName("silver").master("local[*]").getOrCreate()
    )
    spark.conf.set("spark.sql.ansi.enabled", "false")
    spark.sparkContext.setLogLevel("ERROR")

    dates = generate_first_of_month_dates(start_date, end_date)
    bronze_base_directory = "datamart/bronze/"
    silver_base_directory = "datamart/silver/"
    os.makedirs(silver_base_directory, exist_ok=True)

    for source_name in bronze.SOURCE_CONFIG:
        silver.process_silver_source_all_snapshots(
            source_name,
            dates,
            bronze_base_directory,
            silver_base_directory,
            spark,
            bronze.SOURCE_CONFIG,
        )

    spark.stop()
    print("--- silver: done ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the silver layer for a date range.")
    parser.add_argument("--start_date", type=str, default="2023-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, default="2025-11-01", help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.start_date, args.end_date)
