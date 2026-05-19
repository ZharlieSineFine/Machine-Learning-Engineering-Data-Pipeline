import os
from datetime import datetime

import pyspark

import utils.data_processing_bronze_table as bronze
import utils.data_processing_silver_table as silver
import utils.data_processing_gold_table as gold

# --- Spark session ---
spark = (
    pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
)

spark.conf.set("spark.sql.ansi.enabled", "false")
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



start_date_str = "2023-01-01"
end_date_str = "2025-11-01"
dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)

bronze_base_directory = "datamart/bronze/"
os.makedirs(bronze_base_directory, exist_ok=True)

# Process bronze tables
for source_name in bronze.SOURCE_CONFIG:
    bronze.process_bronze_source_all_snapshots(
        source_name, dates_str_lst, bronze_base_directory, spark, bronze.SOURCE_CONFIG
    )


# Process silver tables
silver_base_directory = "datamart/silver/"
os.makedirs(silver_base_directory, exist_ok=True)

for source_name in bronze.SOURCE_CONFIG:
    silver.process_silver_source_all_snapshots(
        source_name, dates_str_lst, bronze_base_directory,
        silver_base_directory, spark, bronze.SOURCE_CONFIG,
    )

# Process gold label store
gold_label_store_directory = "datamart/gold/label_store/"
os.makedirs(gold_label_store_directory, exist_ok=True)
silver_loan_daily_directory = os.path.join(silver_base_directory, "loan_daily")

for date_str in dates_str_lst:
    gold.process_labels_gold_table(
        date_str,
        silver_loan_daily_directory,
        gold_label_store_directory,
        spark,
        dpd=30,
        mob=6,
    )

# Process gold feature store
gold_feature_store_directory = "datamart/gold/feature_store/"
os.makedirs(gold_feature_store_directory, exist_ok=True)
gold.build_feature_store(
    silver_base_directory,
    gold_label_store_directory,
    gold_feature_store_directory,
    spark,
    oot_months=2,
)
