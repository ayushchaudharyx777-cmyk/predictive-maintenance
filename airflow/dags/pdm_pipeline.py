"""Airflow DAGs for the predictive-maintenance project (works with Airflow 2.x and 3.x).

  pdm_training    weekly : validate data -> train -> quality gate
  pdm_monitoring  daily  : drift check on the API's logged traffic

Setup: copy/symlink this file into $AIRFLOW_HOME/dags and set  PDM_PROJECT_DIR=/path/to/predictive-maintenance
(Airflow does not run natively on Windows: use WSL2 or the official Docker image.)
A failing task = a real problem: bad data, the quality gate failed, or drift is at ALERT level.
Exit code 99 means "nothing to check yet" and is shown as Skipped, not Failed.
"""
import os
from datetime import datetime, timedelta

try:                                                   # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator
    from airflow.sdk import DAG
except ImportError:                                    # Airflow 2
    from airflow import DAG
    from airflow.operators.bash import BashOperator

PROJECT = os.environ.get("PDM_PROJECT_DIR", "/opt/predictive-maintenance")
DEFAULTS = {"owner": "ml-team", "retries": 1, "retry_delay": timedelta(minutes=5)}


def sh(cmd: str) -> str:
    return f"cd {PROJECT} && {cmd}"


with DAG(
    dag_id="pdm_training",
    description="Weekly retrain with data validation and a quality gate",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    default_args=DEFAULTS,
    tags=["predictive-maintenance", "training"],
) as training_dag:
    validate = BashOperator(task_id="validate_data", bash_command=sh("python validate.py"), retries=0)
    train = BashOperator(task_id="train", bash_command=sh("python train.py"))
    gate = BashOperator(task_id="quality_gate", bash_command=sh("python scripts/quality_gate.py"), retries=0)
    validate >> train >> gate

with DAG(
    dag_id="pdm_monitoring",
    description="Daily drift monitoring of the served model",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args=DEFAULTS,
    tags=["predictive-maintenance", "monitoring"],
) as monitoring_dag:
    drift = BashOperator(
        task_id="drift_check",
        bash_command=sh("[ -f logs/predictions.jsonl ] || exit 99; python monitor.py --current logs/predictions.jsonl"),
        skip_on_exit_code=99, retries=0)
