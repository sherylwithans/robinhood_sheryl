import sys
import os
from datetime import datetime, timedelta
from airflow import DAG
# from airflow.models import Variable
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator

AIRFLOW_HOME = os.environ.get("AF_HOME")
SYS_DIR = AIRFLOW_HOME + '../'
sys.path.append(SYS_DIR)

import pendulum

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
    'email_test', default_args=default_args, schedule_interval=None, catchup=False,
)

process_dag = EmailOperator(
    task_id='test_email',
    to=os.environ.get("AF_EMAIL"),
    subject='robinhood_sheryl test email',
    html_content="""<h3>TEST</h3>""",
    dag=dag)
