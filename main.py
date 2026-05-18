import os
from datetime import datetime

import pyspark

import utils.data_processing_bronze_table as bronze

# --- Spark session (same pattern as Lab 2) ---
spark = (
    pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")


def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    first_of_month_dates = []
    current_date = datetime(start_date.year, start_date.month, 1)
    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
    return first_of_month_dates


# Widest span across all sources (loan_daily runs latest, to 2025-11).
# Bronze lands everything available; date windowing/alignment is a GOLD concern.
start_date_str = "2023-01-01"
end_date_str = "2025-11-01"
dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)

bronze_base_directory = "datamart/bronze/"
os.makedirs(bronze_base_directory, exist_ok=True)

# One read.csv + one cache lifetime per source (not per month).
for source_name in bronze.SOURCE_CONFIG:
    bronze.process_bronze_source_all_snapshots(
        source_name, dates_str_lst, bronze_base_directory, spark, bronze.SOURCE_CONFIG
    )
