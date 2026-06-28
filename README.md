# Machine Learning Engineering - Meddalion Pipeline + Airflow DAG

End-to-end PySpark + Airflow pipeline for **credit default prediction** at a digital lender. Raw LMS and feature CSVs are ingested into a **medallion datamart** (bronze, silver, gold), then a model is trained, used to score every monthly cohort, and monitored over time. The whole lifecycle runs as a single Apache Airflow DAG (`ml_pipeline`).

## Pipeline at a glance

The DAG runs the medallion layers and the ML lifecycle as one sequential chain:

```
dep_check -> bronze -> silver -> gold_label -> gold_feature
          -> model_train -> model_inference -> model_monitor -> pipeline_complete
```

| Stage | Script | What it produces |
|-------|--------|------------------|
| Bronze | `scripts/data_processing_bronze.py` | Raw CSV snapshots by source and month under `datamart/bronze/` |
| Silver | `scripts/data_processing_silver.py` | Cleaned, type-cast partitions (`mob`/`dpd` on loan daily) under `datamart/silver/` |
| Gold label | `scripts/data_processing_gold_label.py` | `label = 1` if `dpd >= 30` at `mob = 6`, under `datamart/gold/label_store/` |
| Gold feature | `scripts/data_processing_gold_feature.py` | Leakage-safe feature store under `datamart/gold/feature_store/` (~12,500 rows) |
| Model train | `scripts/model_train.py` | Best of logistic regression / random forest / XGBoost, saved to `model_bank/credit_model.pkl` |
| Model inference | `scripts/model_inference.py` | Per-month predictions under `datamart/gold/model_predictions/` |
| Model monitor | `scripts/model_monitor.py` | Per-month metrics table + PNG charts under `datamart/gold/model_monitoring/` |

The feature store holds one row per `(Customer_ID, snapshot_date)`: application-time attributes and financials, clickstream aggregates up to `loan_start_date`, categorical encodings, missingness flags, age validity, five engineered ratios, and winsor caps (p99) plus median imputation fit only on `snapshot_date < train_cutoff`. The last two cohort months are held out for out-of-time (OOT) evaluation.

## Repository layout

```
.
├── data/                   # Input CSVs (4 sources)
├── dags/
│   └── ml_pipeline_dag.py  # The Airflow DAG that orchestrates everything
├── scripts/                # One CLI script per pipeline stage
├── utils/                  # Bronze / silver / gold Spark transforms
├── model_bank/             # Trained model artefact (created by the run)
├── datamart/               # Medallion output (created by the run)
├── Dockerfile              # Airflow 2.6.1 + Python 3.10 + Java 17 + deps
├── docker-compose.yaml     # airflow-init / webserver / scheduler
└── requirements.txt
```

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/), installed and running.
- The four source CSVs in `data/`: `feature_clickstream.csv`, `features_attributes.csv`, `features_financials.csv`, `lms_loan_daily.csv`.

## Reproduce our results

Run every command from the repository root (the folder with `docker-compose.yaml`).

### 1. Build the image

```bash
docker compose build
```

The first build takes a few minutes because it installs Java 17 and the Python dependencies.

### 2. Start Airflow

```bash
docker compose up -d
```

This starts the stack: a one-shot `fix-permissions` and `airflow-init` step (they create the SQLite metadata DB and an `admin` user, then exit), followed by the long-running `airflow-webserver` and `airflow-scheduler`. Wait about 30 seconds, then open:

**http://localhost:8080** and log in with **admin / admin**.

> Open `localhost:8080`, not `0.0.0.0:8080`. The `0.0.0.0` in the logs is the bind address inside the container, not a browsable URL.

### 3. Trigger the pipeline

`ml_pipeline` is unpaused on startup, so you can run it right away:

- **From the UI:** find `ml_pipeline` and click the play (Trigger DAG) button.
- **From the CLI:**
  ```bash
  docker compose exec airflow-scheduler airflow dags trigger ml_pipeline
  ```

It runs as one sequential chain, so just let it finish. A full run takes roughly **30 to 50 minutes**: Spark starts fresh for each layer over monthly snapshots from 2023-01 through 2025-11, then trains and evaluates three model families.

Watch progress in the Grid view, or from the CLI:

```bash
# list runs to get the run id
docker compose exec airflow-scheduler airflow dags list-runs -d ml_pipeline
# then check per-task status for that run
docker compose exec airflow-scheduler airflow tasks states-for-dag-run ml_pipeline <run_id>
```

### 4. Collect the outputs

When `pipeline_complete` is green, the artefacts are written to your host (the project folders are bind-mounted into the container):

| Output | Path |
|--------|------|
| Trained model | `model_bank/credit_model.pkl` |
| Label store | `datamart/gold/label_store/` (25 monthly partitions) |
| Feature store | `datamart/gold/feature_store/` (same cohort months) |
| Predictions | `datamart/gold/model_predictions/credit_model/` (one parquet per month) |
| Monitoring table | `datamart/gold/model_monitoring/credit_model_monitoring.parquet` |
| Monitoring charts | `datamart/gold/model_monitoring/charts/*.png` |

The training log prints a per-model comparison (validation and OOT AUC/Gini) and which model was selected. The monitoring log prints the per-month metric table and a drift verdict against the OOT baseline AUC.

### 5. Stop everything

```bash
docker compose down
```

Add `-v` to also drop the Airflow metadata volume if you want a completely clean slate next time.

## Run a single stage (debugging)

Each stage is a standalone script, so you can run one in isolation inside the container without the DAG:

```bash
docker compose exec airflow-scheduler python3 scripts/model_train.py --modelname credit_model.pkl --oot_months 2
docker compose exec airflow-scheduler python3 scripts/model_monitor.py --modelname credit_model.pkl
```

## Notes

- **Window and OOT settings** come from `dags/ml_pipeline_dag.py` (`START_DATE`, `END_DATE`, `OOT_MONTHS`). `OOT_MONTHS` must match between `gold_feature` and `model_train` so the train-window statistics and the model's training window share the same cutoff.
- **Generated data** (`datamart/` and the model artefact) is produced by the run and is gitignored. Clone the repo, run the DAG, and the outputs appear locally.
