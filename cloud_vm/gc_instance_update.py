# requirements.txt
#
# google-api-python-client
# google-auth-httplib2
# google-auth
# oauth2client
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import os

credentials = GoogleCredentials.get_application_default()
GCE_SERVICE = discovery.build('compute', 'v1', credentials=credentials, cache_discovery=False)
PROJECT = 'quantwannabe'  # TODO: Update placeholder value.
ZONE = 'us-central1-c'


def list_instances(instance='all', print_all=False, get_ext_ip=False, get_int_ip=False):
    result = GCE_SERVICE.instances().list(project=PROJECT, zone=ZONE).execute()

    keys = ['id', 'name', 'status', 'lastStartTimestamp', 'lastStopTimestamp']
    if 'items' in result:
        results = result['items']
        if instance == 'all':
            return results
        else:
            for i in results:
                if i['name'] == instance:
                    if print_all:
                        print(i)
                    elif get_ext_ip:
                        ext_ip = i.get('networkInterfaces')[0].get('accessConfigs')[0].get('natIP')
                        return ext_ip
                    elif get_int_ip:
                        int_ip = i.get('networkInterfaces')[0].get('networkIP')
                        return int_ip
                    else:
                        print({k: i.get(k) for k in keys})
                        return {k: i.get(k) for k in keys}
            print('INVALID: Instance not found in list')
    else:
        print('EMPTY: No items in result')


def start_instance(instance='postgres-db'):
    if list_instances(instance).get('status') == 'TERMINATED':
        request = GCE_SERVICE.instances().start(project=PROJECT, zone=ZONE, instance=instance)
        response = request.execute()
        print(f'STARTING: instance {instance} is starting up')
        print(response)
        return True
    else:
        print(f'SKIPPED: instance {instance} is already running')
        return False


def stop_instance(instance='postgres-db'):
    if list_instances(instance).get('status') == 'RUNNING':
        request = GCE_SERVICE.instances().stop(project=PROJECT, zone=ZONE, instance=instance)
        response = request.execute()
        print(f'TERMINATED: instance {instance} is shutting down')
        print(response)
        return True
    else:
        print(f'SKIPPED: instance {instance} is already terminated')
        return False


if __name__ == "__main__":
    print('here')
    # start_instance('robinhood-sheryl-app')
    # list_instances('postgres-db-prod', get_int_ip=True)
    list_instances('postgres-db-prod', get_int_ip=True)
    print(os.environ.get("DB_HOST"))