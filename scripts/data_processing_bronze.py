"""
scripts/data_processing_bronze.py - Bronze layer DAG node.

Lands all raw sources (clickstream, attributes, financials, loan_daily) as
faithful bronze partitions for every month in the range. Thin CLI wrapper over
utils.data_processing_bronze_table, so the Airflow DAG can call it directly.

Run from the repo root (inside the container, /opt/airflow):
    python3 scripts/data_processing_bronze.py --start_date 2023-01-01 --end_date 2025-11-01
"""

import argparse
import os
import sys
from datetime import datetime

# Make the repo-root `utils` package importable when run as scripts/<file>.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyspark  # noqa: E402

import utils.data_processing_bronze_table as bronze  # noqa: E402


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
    print("\n--- bronze: start ---")
    spark = (
        pyspark.sql.SparkSession.builder.appName("bronze").master("local[*]").getOrCreate()
    )
    spark.conf.set("spark.sql.ansi.enabled", "false")
    spark.sparkContext.setLogLevel("ERROR")

    dates = generate_first_of_month_dates(start_date, end_date)
    bronze_base_directory = "datamart/bronze/"
    os.makedirs(bronze_base_directory, exist_ok=True)

    for source_name in bronze.SOURCE_CONFIG:
        bronze.process_bronze_source_all_snapshots(
            source_name, dates, bronze_base_directory, spark, bronze.SOURCE_CONFIG
        )

    spark.stop()
    print("--- bronze: done ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the bronze layer for a date range.")
    parser.add_argument("--start_date", type=str, default="2023-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, default="2025-11-01", help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.start_date, args.end_date)
