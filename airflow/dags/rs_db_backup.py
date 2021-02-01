import sys
import os
from datetime import datetime, timedelta
from airflow import DAG
# from airflow.models import Variable
from airflow.operators.bash_operator import BashOperator

AIRFLOW_HOME = os.environ.get("AF_HOME")
sys.path.append(AIRFLOW_HOME)

from robinhood_sheryl.rs_db import local_tz

SYS_DIR = AIRFLOW_HOME+'../'

db_host = os.environ["DB_HOST"]
# Extract host and port from db_host
host_args = db_host.split(":")
db_hostname, db_port = host_args[0], int(host_args[1])

default_args = {
    'owner': 'sheryl',
    'depends_on_past': False,
    'start_date': datetime(2019, 8, 1,tzinfo=local_tz),
    'email': [os.environ.get("AF_EMAIL")],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'dag_db_backup', default_args=default_args, schedule_interval="5 20 * * 1-5", catchup=False,
    )

process_dag = BashOperator(
    task_id='dag_db_backup',
    bash_command=f'pg_dump -h {db_hostname} -U postgres prod > {SYS_DIR}prod_dump.sql && gsutil cp {SYS_DIR}prod_dump.sql gs://quantwannadb/',
    dag=dag,
    execution_timeout=timedelta(seconds=300))
