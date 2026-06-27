"""
dags/ml_pipeline_dag.py - end-to-end medallion + ML lifecycle DAG.

Single-run pipeline (trigger once; catchup off). Every layer is a real
BashOperator node calling a standalone script in scripts/:

    dep_check -> bronze -> silver -> gold_label -> gold_feature
              -> model_train -> model_inference -> model_monitor -> complete

The gold feature store (winsor/impute fit on the train window across ALL months)
and model training are global steps, so the data layers run as full-range batch
nodes before them. Inference and monitoring then span every cohort month, which
is how "behaviour across time" is shown.

Every BashOperator runs from /opt/airflow so the scripts' relative paths
(`data/`, `datamart/`, `model_bank/`) resolve to the mounted volumes.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator

START_DATE = "2023-01-01"
END_DATE = "2025-11-01"
MODEL_NAME = "credit_model.pkl"
OOT_MONTHS = 2

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _bash(script_and_args: str) -> str:
    # Run from the project root so relative datamart/ and model_bank/ paths resolve.
    return f"cd /opt/airflow && python3 scripts/{script_and_args}"


with DAG(
    dag_id="ml_pipeline",
    default_args=default_args,
    description="Medallion data pipeline + ML train/inference/monitor (single run).",
    schedule_interval=None,  # manual trigger; this is a batch end-to-end run
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["cs611", "assignment2"],
) as dag:

    dep_check_source_data = DummyOperator(task_id="dep_check_source_data")

    bronze = BashOperator(
        task_id="bronze",
        bash_command=_bash(
            f"data_processing_bronze.py --start_date {START_DATE} --end_date {END_DATE}"
        ),
    )

    silver = BashOperator(
        task_id="silver",
        bash_command=_bash(
            f"data_processing_silver.py --start_date {START_DATE} --end_date {END_DATE}"
        ),
    )

    gold_label = BashOperator(
        task_id="gold_label_store",
        bash_command=_bash(
            f"data_processing_gold_label.py --start_date {START_DATE} --end_date {END_DATE}"
        ),
    )

    gold_feature = BashOperator(
        task_id="gold_feature_store",
        bash_command=_bash(f"data_processing_gold_feature.py --oot_months {OOT_MONTHS}"),
    )

    model_train = BashOperator(
        task_id="model_train",
        bash_command=_bash(
            f"model_train.py --modelname {MODEL_NAME} --oot_months {OOT_MONTHS}"
        ),
    )

    model_inference = BashOperator(
        task_id="model_inference",
        bash_command=_bash(f"model_inference.py --modelname {MODEL_NAME}"),
    )

    model_monitor = BashOperator(
        task_id="model_monitor",
        bash_command=_bash(f"model_monitor.py --modelname {MODEL_NAME}"),
    )

    pipeline_complete = DummyOperator(task_id="pipeline_complete")

    (
        dep_check_source_data
        >> bronze
        >> silver
        >> gold_label
        >> gold_feature
        >> model_train
        >> model_inference
        >> model_monitor
        >> pipeline_complete
    )
