import sys
import os
from datetime import datetime, timedelta
from airflow import DAG
# from airflow.models import Variable
from airflow.operators.bash_operator import BashOperator

AIRFLOW_HOME = os.environ.get("AF_HOME")
sys.path.append(AIRFLOW_HOME)

from robinhood_sheryl.rs_db import local_tz

SYS_DIR = AIRFLOW_HOME + '../airflow'


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
    'dag_delete_logs', default_args=default_args, schedule_interval="0 20 * * 1-5", catchup=False,
    )

process_dag = BashOperator(
    task_id='dag_delete_logs',
    bash_command='echo 1',
    dag=dag)


paths = ['scheduler',
         'dag_insert_yf_data',
         'dag_insert_portfolio_data',
         'dag_delete_logs',
         # 'dag_start_instance',
         # 'dag_stop_instance',
         ]

# for i in range(3):
#     task = BashOperator(
#         task_id='runme_'+str(i),
#         bash_command='echo "{{ task_instance_key_str }}" && sleep 1',
#         dag=dag,
#         )
#     task >> process_dag

delete_interval = 5

for p in paths:
    command = fr"find {SYS_DIR}/logs/{p}/{p}/ -mtime +16 -exec rm -rf {{}} ;"
    task = BashOperator(
        task_id=f'delete_{p}',
        bash_command=f"""
                      for d in {SYS_DIR}/logs/{p}/* ; do     
                          find $d -maxdepth 1 -type d -mtime +{delete_interval} -exec rm -rf {{}} \\;
                      done
                    """,
        dag=dag,
        )
    task >> process_dag


    #  find ./logs -type f -mtime +16 -exec rm {} \;
    #