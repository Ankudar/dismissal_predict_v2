from datetime import datetime
import subprocess

from airflow.operators.python import PythonOperator

from airflow import DAG


def run_dataset():
    subprocess.run(
        [
            "python",
            "~/dismissal_predict_v2/dismissal_predict/dataset.py",
        ],
        check=True,
    )


def run_prepare_dataset():
    subprocess.run(
        [
            "python",
            "~/dismissal_predict_v2/dismissal_predict/prepare_dataset.py",
        ],
        check=True,
    )


def run_final_prepare_df():
    subprocess.run(
        [
            "python",
            "~/dismissal_predict_v2/dismissal_predict/final_prepare_df.py",
        ],
        check=True,
    )


def run_train():
    subprocess.run(
        [
            "python",
            "~/dismissal_predict_v2/dismissal_predict/modeling/train.py",
        ],
        check=True,
    )


def run_predict():
    subprocess.run(
        [
            "python",
            "~/dismissal_predict_v2/dismissal_predict/modeling/predict.py",
        ],
        check=True,
    )


with DAG(
    dag_id="dismissal_predict_dag",
    start_date=datetime(1990, 1, 1),
    schedule="0 4 * * 1",
    catchup=False,
) as dag:
    t1 = PythonOperator(
        task_id="run_dataset",
        python_callable=run_dataset,
    )
    t2 = PythonOperator(
        task_id="run_prepare_dataset",
        python_callable=run_prepare_dataset,
    )
    t3 = PythonOperator(
        task_id="run_final_prepare_df",
        python_callable=run_final_prepare_df,
    )
    t4 = PythonOperator(
        task_id="run_train",
        python_callable=run_train,
    )
    t5 = PythonOperator(
        task_id="run_predict",
        python_callable=run_predict,
    )

    t1 >> t2 >> t3 >> t4 >> t5  # type: ignore
