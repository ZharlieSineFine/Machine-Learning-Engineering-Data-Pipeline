import os


from utils.data_processing_bronze_table import SOURCE_CONFIG

from pyspark.sql.functions import (
    col,
    regexp_replace,
    regexp_extract,
    trim,
    when,
    ceil,
    add_months,
    datediff,
    coalesce,
    lit,
)
from pyspark.sql.types import (
    StringType,
    IntegerType,
    FloatType,
    DoubleType,
    DateType,
)


def _read_bronze_partition(
    source_name, snapshot_date_str, bronze_base_directory, source_config, spark
):
    cfg = source_config[source_name]
    partition_name = f"bronze_{source_name}_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_base_directory, cfg["bronze_subdir"], partition_name)
    if not os.path.exists(filepath):
        return None
    return spark.read.csv(filepath, header=True, inferSchema=False)


def _num(c):
    """Deterministic numeric clean: drop any char not digit/dot/minus -> Double."""
    return regexp_replace(col(c), r"[^0-9.-]", "").cast(DoubleType())


def _int(c):
    """Same strip, via Double so '11.0' parses, then Integer."""
    return regexp_replace(col(c), r"[^0-9.-]", "").cast(DoubleType()).cast(IntegerType())


def clean_clickstream(df):
    for i in range(1, 21):
        df = df.withColumn(f"fe_{i}", _int(f"fe_{i}"))
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    return df


def clean_attributes(df):
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    for pii in ("Name", "SSN"):
        if pii in df.columns:
            df = df.drop(pii)
    df = df.withColumn("Age", _int("Age"))
    df = df.withColumn("Occupation", trim(col("Occupation")))
    df = df.withColumn(
        "Occupation",
        when(col("Occupation").rlike(r"^_+$"), None)
        .otherwise(col("Occupation"))
        .cast(StringType()),
    )
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    return df


def clean_financials(df):
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))

    double_cols = [
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Changed_Credit_Limit",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance",
    ]
    for c in double_cols:
        df = df.withColumn(c, _num(c))

    int_cols = [
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Num_Credit_Inquiries",
    ]
    for c in int_cols:
        df = df.withColumn(c, _int(c))

    df = df.withColumn("Type_of_Loan", trim(col("Type_of_Loan")))

    df = df.withColumn(
        "Credit_Mix",
        when(trim(col("Credit_Mix")) == "_", None)
        .otherwise(trim(col("Credit_Mix")))
        .cast(StringType()),
    )
    df = df.withColumn(
        "Payment_Behaviour",
        when(trim(col("Payment_Behaviour")) == "!@9#%8", None)
        .otherwise(trim(col("Payment_Behaviour")))
        .cast(StringType()),
    )
    df = df.withColumn(
        "Payment_of_Min_Amount",
        trim(col("Payment_of_Min_Amount")).cast(StringType()),
    )

    src = coalesce(trim(col("Credit_History_Age")), lit(""))
    yrs = regexp_extract(src, r"(\d+)\s*Year", 1)
    mos = regexp_extract(src, r"(\d+)\s*Month", 1)
    df = df.withColumn(
        "credit_history_age_months",
        when((yrs == lit("")) & (mos == lit("")), None)
        .otherwise(
            (when(yrs == lit(""), lit(0)).otherwise(yrs.cast(IntegerType())) * lit(12))
            + when(mos == lit(""), lit(0)).otherwise(mos.cast(IntegerType()))
        ).cast(IntegerType()),
    )
    df = df.drop("Credit_History_Age")

    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    return df


def clean_loan_daily(df):
    """Implement type map and calculate mob/dpd."""
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }
    for c, t in column_type_map.items():
        df = df.withColumn(c, col(c).cast(t))

    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    df = df.withColumn(
        "installments_missed",
        ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType()),
    )
    df = df.fillna(0, subset=["installments_missed"])
    df = df.withColumn(
        "first_missed_date",
        when(
            col("installments_missed") > 0,
            add_months(col("snapshot_date"), -1 * col("installments_missed")),
        ).cast(DateType()),
    )
    df = df.withColumn(
        "dpd",
        when(col("overdue_amt") > 0.0, datediff(col("snapshot_date"), col("first_missed_date")))
        .otherwise(0)
        .cast(IntegerType()),
    )
    return df


SILVER_CLEANERS = {
    "clickstream": clean_clickstream,
    "attributes": clean_attributes,
    "financials": clean_financials,
    "loan_daily": clean_loan_daily,
}


def process_silver_source_all_snapshots(
    source_name,
    snapshot_dates,
    bronze_base_directory,
    silver_base_directory,
    spark,
    source_config,
):
    """Read each bronze partition, apply cleaner, write silver parquet."""
    cleaner = SILVER_CLEANERS[source_name]
    silver_dir = os.path.join(
        silver_base_directory, source_config[source_name]["bronze_subdir"]
    )
    os.makedirs(silver_dir, exist_ok=True)

    for snapshot_date_str in snapshot_dates:
        df = _read_bronze_partition(
            source_name, snapshot_date_str, bronze_base_directory, source_config, spark
        )
        if df is None:
            continue
        row_count = df.count()
        if row_count == 0:
            continue
        df = cleaner(df)
        partition_name = f"silver_{source_name}_{snapshot_date_str.replace('-', '_')}.parquet"
        filepath = os.path.join(silver_dir, partition_name)
        df.write.mode("overwrite").parquet(filepath)
        print(f"[silver:{source_name}] {snapshot_date_str} rows={row_count} -> {filepath}")
