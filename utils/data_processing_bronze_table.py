import os
from pyspark.sql.functions import col, to_date, date_format, coalesce

# Declare all raw sources here
SOURCE_CONFIG = {
    "clickstream": {"csv_path": "data/feature_clickstream.csv", "bronze_subdir": "clickstream"},
    "attributes": {"csv_path": "data/features_attributes.csv", "bronze_subdir": "attributes"},
    "financials": {"csv_path": "data/features_financials.csv", "bronze_subdir": "financials"},
    "loan_daily": {"csv_path": "data/lms_loan_daily.csv", "bronze_subdir": "loan_daily"},
}


def _normalize_us_date_column(df, column_name):
    """M/d/yyyy (raw LMS) -> ISO yyyy-MM-dd string for downstream DateType casts."""
    return df.withColumn(
        column_name,
        coalesce(
            date_format(to_date(col(column_name), "M/d/yyyy"), "yyyy-MM-dd"),
            col(column_name),
        ),
    )


def process_bronze_source_all_snapshots(
    source_name, snapshot_dates, bronze_base_directory, spark, source_config
):
    """
    Land all requested snapshot partitions for ONE source.

    Reads the raw CSV once, caches in memory, then filters/writes per snapshot_date.
    Bronze contract: faithful raw copy — strings only, no cleaning, skip empty partitions.
    """
    cfg = source_config[source_name]
    csv_file_path = cfg["csv_path"]
    bronze_dir = os.path.join(bronze_base_directory, cfg["bronze_subdir"])
    os.makedirs(bronze_dir, exist_ok=True)

    df = spark.read.csv(csv_file_path, header=True, inferSchema=False)
    # Raw CSVs use M/d/yyyy ("1/1/2023"); pipeline downstream expects ISO yyyy-MM-dd.
    # Fallback to original string if a row doesn't match the M/d/yyyy pattern.
    df = _normalize_us_date_column(df, "snapshot_date")
    if "loan_start_date" in df.columns:
        df = _normalize_us_date_column(df, "loan_start_date")
    df.cache()
    try:
        # Warm cache once so subsequent per-month work reuses scanned partitions.
        print(f"[bronze:{source_name}] cached row count (full file): {df.count()}")
        for snapshot_date_str in snapshot_dates:
            part = df.filter(col("snapshot_date") == snapshot_date_str)
            row_count = part.count()
            print(f"[bronze:{source_name}] {snapshot_date_str} row count: {row_count}")
            if row_count == 0:
                continue
            partition_name = f"bronze_{source_name}_{snapshot_date_str.replace('-', '_')}.csv"
            filepath = os.path.join(bronze_dir, partition_name)
            part.toPandas().to_csv(filepath, index=False)
            print(f"[bronze:{source_name}] saved to: {filepath}")
    finally:
        df.unpersist()
