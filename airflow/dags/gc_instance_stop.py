import sys
import os
from datetime import datetime, timedelta
from airflow import DAG
# from airflow.models import Variable
from airflow.operators.python_operator import PythonOperator

AIRFLOW_HOME = os.environ.get("AF_HOME")
SYS_DIR = AIRFLOW_HOME + '../'
sys.path.append(SYS_DIR)

import pendulum

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
from cloud_vm.gc_instance_update import stop_instance

local_tz = pendulum.timezone("US/Eastern")

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
    'dag_stop_instance', default_args=default_args, schedule_interval="30 20 * * 1-5", catchup=False,
    )

process_dag = PythonOperator(
    task_id='dag_stop_instance',
    python_callable=stop_instance,
    dag=dag)