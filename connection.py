from airflow.models import Variable
from sqlalchemy import create_engine

host = Variable.get("host")
db_name = Variable.get("db_name")
username = Variable.get("username")
password = Variable.get("password")
port = Variable.get("port")
conn = create_engine(f'postgres://{username}:{password}@{host}:{port}/{db_name}')
                           # .format(username=username, password=password,
                           #         url=host, db_name=db_name), echo=False)
# conn = connection.connect()
print(conn)