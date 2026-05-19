import os

from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, DoubleType, StringType

CLICKSTREAM_FE = [f"fe_{i}" for i in range(1, 21)]
AGE_MIN, AGE_MAX = 18, 100

# out-of-range -> null -> train-median
COUNT_CAPS = {
    "Num_Bank_Accounts": (0, 20),
    "Num_Credit_Card": (0, 20),
    "Interest_Rate": (0, 100),
    "Num_of_Loan": (0, 20),
    "Num_of_Delayed_Payment": (0, 100),
    "Num_Credit_Inquiries": (0, 50),
}
# winsorize the top tail on train only
WINSOR_COLS = ["Annual_Income", "Outstanding_Debt", "Amount_invested_monthly"]
WINSOR_P = 0.99
NUMERIC_IMPUTE = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Changed_Credit_Limit",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "credit_history_age_months",
] + list(COUNT_CAPS)
ZERO_IMPUTE = [
    f"{c}_{s}"
    for c in CLICKSTREAM_FE
    for s in ("mean", "std", "min", "max", "last")
] + ["clickstream_n_obs"]

CREDIT_MIX_MAP = {"Bad": 0, "Standard": 1, "Good": 2}
PAY_MIN_MAP = {"No": 0, "Yes": 1, "NM": -1}
OCCUPATIONS = [
    "Lawyer",
    "Mechanic",
    "Media_Manager",
    "Doctor",
    "Journalist",
    "Accountant",
    "Architect",
    "Engineer",
    "Scientist",
    "Teacher",
    "Developer",
    "Entrepreneur",
    "Manager",
    "Musician",
    "Writer",
]
LOAN_TYPES = [
    "Auto Loan",
    "Credit-Builder Loan",
    "Personal Loan",
    "Home Equity Loan",
    "Mortgage Loan",
    "Student Loan",
    "Payday Loan",
    "Debt Consolidation Loan",
    "Not Specified",
]


def _write_gold_parquet(df, filepath):
    if os.name == "nt" and not os.environ.get("HADOOP_HOME"):
        df.toPandas().to_parquet(filepath, index=False, engine="pyarrow")
    else:
        df.write.mode("overwrite").parquet(filepath)


def _read_silver_source(silver_base_directory, subdir, spark):
    path = os.path.join(silver_base_directory, subdir)
    return spark.read.option("recursiveFileLookup", "true").parquet(path)


def process_labels_gold_table(
    snapshot_date_str,
    silver_loan_daily_directory,
    gold_label_store_directory,
    spark,
    dpd,
    mob,
):
    partition_name = f"silver_loan_daily_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_loan_daily_directory, partition_name)
    if not os.path.exists(filepath):
        return None

    df = spark.read.parquet(filepath).filter(col("mob") == mob)
    if df.limit(1).count() == 0:
        return None

    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    os.makedirs(gold_label_store_directory, exist_ok=True)
    out = os.path.join(
        gold_label_store_directory,
        f"gold_label_store_{snapshot_date_str.replace('-', '_')}.parquet",
    )
    _write_gold_parquet(df, out)
    print(f"[gold:label] {snapshot_date_str} rows={df.count()} -> {out}")
    return df


def _clickstream_features_asof(silver_base_directory, cust_appdate, spark):
    clk = _read_silver_source(silver_base_directory, "clickstream", spark)
    clk = clk.join(cust_appdate, on="Customer_ID", how="inner")
    clk = clk.filter(col("snapshot_date") <= col("loan_start_date"))

    aggs = []
    for c in CLICKSTREAM_FE:
        aggs += [
            F.avg(c).alias(f"{c}_mean"),
            F.stddev(c).alias(f"{c}_std"),
            F.min(c).alias(f"{c}_min"),
            F.max(c).alias(f"{c}_max"),
        ]
    aggs.append(F.count(F.lit(1)).alias("clickstream_n_obs"))
    grouped = clk.groupBy("Customer_ID").agg(*aggs)

    w = Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())
    last = (
        clk.withColumn("_rn", F.row_number().over(w))
        .filter(col("_rn") == 1)
        .select("Customer_ID", *[col(c).alias(f"{c}_last") for c in CLICKSTREAM_FE])
    )
    return grouped.join(last, on="Customer_ID", how="inner")


def _profile_at_application(silver_base_directory, subdir, appdate, spark):
    """One row/customer = the silver row captured AT loan_start_date."""
    src = _read_silver_source(silver_base_directory, subdir, spark)
    return (
        src.join(appdate, on="Customer_ID", how="inner")
        .filter(col("snapshot_date") == col("loan_start_date"))
        .drop("snapshot_date", "loan_start_date")
    )


def _apply_age_validity(df, min_age=AGE_MIN, max_age=AGE_MAX):
    """Null Age outside [min,max]; flag rows that had an out-of-range value."""
    if "Age" not in df.columns:
        return df
    oor = col("Age").isNotNull() & ((col("Age") < min_age) | (col("Age") > max_age))
    return (
        df.withColumn(
            "age_out_of_range",
            F.when(oor, 1).otherwise(0).cast(IntegerType()),
        ).withColumn("Age", F.when(oor, None).otherwise(col("Age")))
    )


def _map_col(name, mapping, default):
    """Deterministic categorical -> int via a FIXED map (no fitted statistic)."""
    e = F.when(col(name).isNull(), F.lit(default))
    for k, v in mapping.items():
        e = e.when(col(name) == k, F.lit(v))
    return e.otherwise(F.lit(default)).cast(IntegerType())


def _engineer_features(df):
    for c, (lo, hi) in COUNT_CAPS.items():
        df = df.withColumn(
            c,
            F.when((col(c) >= lo) & (col(c) <= hi), col(c))
            .otherwise(None)
            .cast(IntegerType()),
        )

    df = df.withColumn(
        "Delay_from_due_date",
        F.when(col("Delay_from_due_date") < 0, 0)
        .otherwise(col("Delay_from_due_date"))
        .cast(IntegerType()),
    )

    df = df.withColumn(
        "Monthly_Balance",
        F.when(F.abs(col("Monthly_Balance")) > F.lit(1e12), None)
        .otherwise(col("Monthly_Balance"))
        .cast(DoubleType()),
    )

    df = df.withColumn(
        "Monthly_Inhand_Salary",
        F.when(
            col("Monthly_Inhand_Salary").isNull() & col("Annual_Income").isNotNull(),
            col("Annual_Income") / F.lit(12.0),
        )
        .otherwise(col("Monthly_Inhand_Salary"))
        .cast(DoubleType()),
    )

    df = df.withColumn(
        "credit_mix_missing", col("Credit_Mix").isNull().cast(IntegerType())
    )
    df = df.withColumn("Credit_Mix", _map_col("Credit_Mix", CREDIT_MIX_MAP, -1))
    df = df.withColumn(
        "Payment_of_Min_Amount", _map_col("Payment_of_Min_Amount", PAY_MIN_MAP, -1)
    )

    df = df.withColumn(
        "payment_behaviour_missing",
        col("Payment_Behaviour").isNull().cast(IntegerType()),
    )
    df = df.withColumn(
        "pay_spend_level",
        F.when(col("Payment_Behaviour").rlike("^High_spent"), 1)
        .when(col("Payment_Behaviour").rlike("^Low_spent"), 0)
        .otherwise(-1)
        .cast(IntegerType()),
    )
    df = df.withColumn(
        "pay_value_level",
        F.when(col("Payment_Behaviour").rlike("Large_value"), 2)
        .when(col("Payment_Behaviour").rlike("Medium_value"), 1)
        .when(col("Payment_Behaviour").rlike("Small_value"), 0)
        .otherwise(-1)
        .cast(IntegerType()),
    )
    df = df.drop("Payment_Behaviour")

    df = df.withColumn(
        "occupation_missing", col("Occupation").isNull().cast(IntegerType())
    )
    for occ in OCCUPATIONS:
        slug = occ.lower().replace(" ", "_").replace("-", "_")
        df = df.withColumn(f"occ_{slug}", (col("Occupation") == occ).cast(IntegerType()))
    df = df.withColumn(
        "occ_other",
        (
            col("Occupation").isNotNull() & ~col("Occupation").isin(OCCUPATIONS)
        ).cast(IntegerType()),
    )
    df = df.drop("Occupation")

    df = df.withColumn(
        "_tl",
        F.regexp_replace(
            F.coalesce(col("Type_of_Loan"), F.lit("")), r"\s+and\s+", ", "
        ),
    )
    df = df.withColumn(
        "_tla",
        F.expr(
            "array_distinct(filter(transform(split(_tl, ','), x -> trim(x)), y -> y != ''))"
        ),
    )
    df = df.withColumn(
        "type_of_loan_missing", (F.size(col("_tla")) == 0).cast(IntegerType())
    )
    df = df.withColumn("num_loan_types", F.size(col("_tla")).cast(IntegerType()))
    for t in LOAN_TYPES:
        slug = t.lower().replace(" ", "_").replace("-", "_")
        df = df.withColumn(
            f"loan_{slug}", F.array_contains(col("_tla"), t).cast(IntegerType())
        )
    df = df.drop("Type_of_Loan", "_tl", "_tla")
    return df


def _impute(df, train_cutoff):
    """Train-window winsor p99 + median impute; applied to all rows."""
    fill_cols = [c for c in (ZERO_IMPUTE + NUMERIC_IMPUTE) if c in df.columns]
    for c in fill_cols:
        df = df.withColumn(c, col(c).cast(DoubleType()))

    train_df = df.filter(col("snapshot_date") < F.to_date(F.lit(train_cutoff)))

    wcols = [c for c in WINSOR_COLS if c in df.columns]
    if wcols:
        caps = train_df.approxQuantile(wcols, [WINSOR_P], 0.001)
        for c, q in zip(wcols, caps):
            if q:
                df = df.withColumn(
                    c, F.when(col(c) > F.lit(q[0]), F.lit(q[0])).otherwise(col(c))
                )

    qs = train_df.approxQuantile(fill_cols, [0.5], 0.001)
    for c, q in zip(fill_cols, qs):
        m = q[0] if q else 0.0
        df = df.withColumn(c, F.coalesce(col(c), F.lit(m)))
    return df


def build_feature_store(
    silver_base_directory,
    gold_label_store_directory,
    gold_feature_store_directory,
    spark,
    oot_months=2,
):
    labels = spark.read.option("recursiveFileLookup", "true").parquet(
        gold_label_store_directory
    )
    spine = labels.select("Customer_ID", "snapshot_date").dropDuplicates()
    n_customers = spine.select("Customer_ID").distinct().count()

    months = sorted(
        r["m"]
        for r in spine.select(
            F.date_format(col("snapshot_date"), "yyyy-MM-dd").alias("m")
        )
        .distinct()
        .collect()
    )
    oot_months = min(oot_months, max(1, len(months) - 1))
    train_cutoff = months[-oot_months]
    print(
        f"[gold:feature] train_cutoff={train_cutoff} "
        f"(winsor p99 + impute median fit on snapshot_date < {train_cutoff} only)"
    )

    loans = _read_silver_source(silver_base_directory, "loan_daily", spark)
    appdate = loans.groupBy("Customer_ID").agg(
        F.min("loan_start_date").alias("loan_start_date")
    )

    attr = _profile_at_application(silver_base_directory, "attributes", appdate, spark)
    fin = _profile_at_application(silver_base_directory, "financials", appdate, spark)
    clk = _clickstream_features_asof(silver_base_directory, appdate, spark)

    n_attr, n_fin = attr.count(), fin.count()
    print(f"[gold:feature] reconcile customers={n_customers} attr={n_attr} fin={n_fin}")

    fs = (
        spine.join(appdate, on="Customer_ID", how="left")
        .join(attr, on="Customer_ID", how="left")
        .join(fin, on="Customer_ID", how="left")
        .join(clk, on="Customer_ID", how="left")
    )

    fs = fs.withColumn(
        "clickstream_missing",
        F.when(col("clickstream_n_obs").isNull(), 1)
        .otherwise(0)
        .cast(IntegerType()),
    )
    fs = fs.withColumn(
        "attributes_missing",
        F.when(col("Age").isNull() & col("Occupation").isNull(), 1)
        .otherwise(0)
        .cast(IntegerType()),
    )
    fs = fs.withColumn(
        "financials_missing",
        F.when(col("Annual_Income").isNull() & col("Credit_Mix").isNull(), 1)
        .otherwise(0)
        .cast(IntegerType()),
    )

    fs = _apply_age_validity(fs)
    fs = _engineer_features(fs)
    fs = _impute(fs, train_cutoff)

    os.makedirs(gold_feature_store_directory, exist_ok=True)
    fs = fs.withColumn("_snap", F.date_format(col("snapshot_date"), "yyyy-MM-dd"))
    fs.cache()
    fs.count()
    try:
        for ds in [r["_snap"] for r in fs.select("_snap").distinct().collect()]:
            part = fs.filter(col("_snap") == ds).drop("_snap")
            if part.limit(1).count() == 0:
                continue
            out = os.path.join(
                gold_feature_store_directory,
                f"gold_feature_store_{ds.replace('-', '_')}.parquet",
            )
            _write_gold_parquet(part, out)
            print(f"[gold:feature] {ds} rows={part.count()} -> {out}")
    finally:
        fs.unpersist()
