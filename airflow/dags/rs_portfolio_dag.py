import sys
import os
from datetime import datetime, timedelta
from airflow import DAG
# from airflow.models import Variable
from airflow.operators.python_operator import PythonOperator

AIRFLOW_HOME = os.environ.get("AF_HOME")
sys.path.append(AIRFLOW_HOME)

# from robinhood_sheryl.rs_db import insert_portfolio_data, local_tz
from robinhood_sheryl.rs_db import *

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

login()  # custom robinhood login function

dag = DAG(
    'dag_insert_portfolio_data',
    default_args=default_args,
    schedule_interval="*/5 8-19 * * 1-5",
    catchup=False,
)


def get_python_operator(task_id, python_callable):
    return PythonOperator(
        task_id=task_id,
        python_callable=python_callable,
        dag=dag,
        execution_timeout=timedelta(seconds=60))


insert_portfolio_data = get_python_operator(task_id='insert_portfolio_data',
                                            python_callable=insert_portfolio_data)

insert_crypto_data = get_python_operator(task_id='insert_crypto_data',
                                         python_callable=insert_crypto_data)

insert_options_data = get_python_operator(task_id='insert_options_data',
                                          python_callable=insert_options_data)

insert_portfolio_summary_data = get_python_operator(task_id='insert_portfolio_summary_data',
                                                    python_callable=insert_portfolio_summary_data)


insert_portfolio_data >> insert_crypto_data >> insert_options_data >> insert_portfolio_summary_data