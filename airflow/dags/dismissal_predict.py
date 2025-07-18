from datetime import datetime, timedelta
import subprocess

from airflow.operators.python import PythonOperator  # type: ignore

from airflow import DAG


# Универсальная обертка под subprocess
def make_subprocess_callable(script_path):
    def _run():
        subprocess.run(["python", script_path], check=True)

    return _run


# Конфигурация пайплайнов
pipelines = {
    "dissmissal_predict_get_data_and_pred": {
        "schedule": timedelta(days=3),
        "tasks": [
            (
                "run_dataset",
                "/home/root6/python/dismissal_predict_v2/dismissal_predict/dataset.py",
            ),
            (
                "run_prepare_dataset",
                "/home/root6/python/dismissal_predict_v2/dismissal_predict/prepare_dataset.py",
            ),
            (
                "run_final_prepare_df",
                "/home/root6/python/dismissal_predict_v2/dismissal_predict/final_prepare_df.py",
            ),
            (
                "run_predict",
                "/home/root6/python/dismissal_predict_v2/dismissal_predict/modeling/predict.py",
            ),
        ],
    },
    "dissmissal_predict_train": {
        "schedule": timedelta(weeks=2),
        "tasks": [
            (
                "run_train",
                "/home/root6/python/dismissal_predict_v2/dismissal_predict/modeling/train.py",
            ),
        ],
    },
}


# Общие аргументы DAG
default_args = {
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    "start_date": datetime(2024, 1, 1),
}


# Создание DAG-ов
for dag_id, pipeline in pipelines.items():
    dag = DAG(
        dag_id=dag_id,
        schedule=pipeline["schedule"],
        default_args=default_args,
        catchup=False,
        max_active_runs=1,  # Один одновременный запуск DAG
        max_active_tasks=1,  # Одна одновременная задача внутри DAG
        tags=["dismissal"],
    )

    with dag:
        previous_task = None
        for task_id, script_path in pipeline["tasks"]:
            task = PythonOperator(
                task_id=task_id, python_callable=make_subprocess_callable(script_path)
            )
            if previous_task:
                previous_task >> task  # type: ignore
            previous_task = task

    globals()[dag_id] = dag
