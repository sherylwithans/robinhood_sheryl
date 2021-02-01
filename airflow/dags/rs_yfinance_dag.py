import sys
import os
from datetime import datetime, timedelta
from airflow import DAG
# from airflow.models import Variable
from airflow.operators.python_operator import PythonOperator

AIRFLOW_HOME = os.environ.get("AF_HOME")
sys.path.append(AIRFLOW_HOME)

from robinhood_sheryl.rs_db import insert_yf_data, local_tz

# import logging

# logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s: %(lineno)d')
# date_time = ' %(asctime)s - %(levelname)s - %(message)s'
# logging.debug('Start of program')


default_args = {
    'owner': 'sheryl',
    'depends_on_past': False,
    'start_date': datetime(2019, 8, 1, tzinfo=local_tz),
    'email': [os.environ.get("AF_EMAIL")],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'dag_insert_yf_data', default_args=default_args, schedule_interval="*/5 8-19 * * 1-5", catchup=False,
    )

process_dag = PythonOperator(
    task_id='dag_insert_yf_data',
    python_callable=insert_yf_data,
    dag=dag)