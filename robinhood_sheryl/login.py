import os
import pickle
import random

import robin_stocks.helper as helper
import robin_stocks.urls as urls

# custom imports for login
import pyotp
from datetime import datetime

# from robinhood_sheryl.secrets import *
# from google.cloud import storage
#
# storage_client = storage.Client()
# bucket = storage_client.get_bucket('quantwannadb')
# blob_pickle_path='.tokens/robinhood.pickle'
# blob = bucket.blob(blob_pickle_path)
# blob_pickle = blob.download_as_string()

pickle_path='.tokens/robinhood.pickle'


def generate_device_token():
    """This function will generate a token used when loggin on.
    :returns: A string representing the token.
    """
    rands = []
    for i in range(0, 16):
        r = random.random()
        rand = 4294967296.0 * r
        rands.append((int(rand) >> ((3 & i) << 3)) & 255)

    hexa = []
    for i in range(0, 256):
        hexa.append(str(hex(i+256)).lstrip("0x").rstrip("L")[1:])

    id = ""
    for i in range(0, 16):
        id += hexa[rands[i]]

        if (i == 3) or (i == 5) or (i == 7) or (i == 9):
            id += "-"

    return(id)


def respond_to_challenge(challenge_id, sms_code):
    """This function will post to the challenge url.
    :param challenge_id: The challenge id.
    :type challenge_id: str
    :param sms_code: The sms code.
    :type sms_code: str
    :returns:  The response from requests.
    """
    url = urls.challenge_url(challenge_id)
    payload = {
        'response': sms_code
    }
    return(helper.request_post(url, payload))


# def login(username=None, password=None, expiresIn=86400, scope='internal', by_sms=True, store_session=True, mfa_code=None,path=None):
# def login(event, context):
def login():
    """This function will effectively log the user into robinhood by getting an
    authentication token and saving it to the session header. By default, it
    will store the authentication token in a pickle file and load that value
    on subsequent logins.
    :param username: The username for your robinhood account, usually your email.
        Not required if credentials are already cached and valid.
    :type username: Optional[str]
    :param password: The password for your robinhood account. Not required if
        credentials are already cached and valid.
    :type password: Optional[str]
    :param expiresIn: The time until your login session expires. This is in seconds.
    :type expiresIn: Optional[int]
    :param scope: Specifies the scope of the authentication.
    :type scope: Optional[str]
    :param by_sms: Specifies whether to send an email(False) or an sms(True)
    :type by_sms: Optional[boolean]
    :param store_session: Specifies whether to save the log in authorization
        for future log ins.
    :type store_session: Optional[boolean]
    :param mfa_code: MFA token if enabled.
    :type mfa_code: Optional[str]
    :returns:  A dictionary with log in information. The 'access_token' keyword contains the access token, and the 'detail' keyword \
    contains information on whether the access token was generated or loaded from pickle file.
    """
    print('logging in')
    username = None
    password = None
    expiresIn = 86400
    scope = 'internal'
    by_sms = True
    store_session = True
    mfa_code = None

    device_token = generate_device_token()
    # home_dir = path if path else os.path.expanduser("~")
    home_dir = os.path.expanduser("~")
    data_dir = os.path.join(home_dir, ".tokens")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    creds_file = "robinhood.pickle"
    pickle_path = os.path.join(data_dir, creds_file)
    print(pickle_path)
    # Challenge type is used if not logging in with two-factor authentication.
    if by_sms:
        challenge_type = "sms"
    else:
        challenge_type = "email"

    url = urls.login_url()
    payload = {
        'client_id': 'c82SH0WZOsabOXGP2sxqcj34FxkvfnWRZBKlBjFS',
        'expires_in': expiresIn,
        'grant_type': 'password',
        'password': password,
        'scope': scope,
        'username': username,
        'challenge_type': challenge_type,
        'device_token': device_token
    }

    if mfa_code:
        payload['mfa_code'] = mfa_code

    # If authentication has been stored in pickle file then load it. Stops login server from being pinged so much.
    if os.path.isfile(pickle_path):
    # if blob_pickle_path:
        # If store_session has been set to false then delete the pickle file, otherwise try to load it.
        # Loading pickle file will fail if the acess_token has expired.
        if store_session:
            try:
                with open(pickle_path, 'rb') as f:
                # if blob_pickle_path:
                    pickle_data = pickle.load(f)
                    print('loaded pickle')
                    # pickle_data = pickle.loads(blob_pickle)
                    # print('loaded blob_pickle')
                    access_token = pickle_data['access_token']
                    token_type = pickle_data['token_type']
                    refresh_token = pickle_data['refresh_token']
                    # Set device_token to be the original device token when first logged in.
                    pickle_device_token = pickle_data['device_token']
                    payload['device_token'] = pickle_device_token
                    # Set login status to True in order to try and get account info.
                    helper.set_login_state(True)
                    helper.update_session(
                        'Authorization', '{0} {1}'.format(token_type, access_token))
                    # Try to load account profile to check that authorization token is still valid.
                    res = helper.request_get(
                        urls.portfolio_profile(), 'regular', payload, jsonify_data=False)
                    # Raises exception is response code is not 200.
                    res.raise_for_status()
                    return({'access_token': access_token, 'token_type': token_type,
                            'expires_in': expiresIn, 'scope': scope, 'detail': 'logged in using authentication in {0}'.format(creds_file),
                            'backup_code': None, 'refresh_token': refresh_token})
            except:
                print(
                    "ERROR: There was an issue loading pickle file. Authentication may be expired - logging in normally.", file=helper.get_output())

                helper.set_login_state(False)
                helper.update_session('Authorization', None)
        else:
            os.remove(pickle_path)

    # Try to log in normally.
    if not username:
        # username = input("Robinhood username: ")
        # payload['username'] = access_secret_version('quantwannabe', 'RB_USER', 'latest')
        payload['username'] = os.environ.get("RB_USER")

    if not password:
        # password = getpass.getpass("Robinhood password: ")
        # payload['password'] = access_secret_version('quantwannabe', 'RB_PASS', 'latest')
        payload['password'] = os.environ.get("RB_PASS")

    data = helper.request_post(url, payload)
    # Handle case where mfa or challenge is required.
    if data:
        if 'mfa_required' in data:
            # totp = str(pyotp.TOTP(access_secret_version('quantwannabe', 'RB_TOKEN', 'latest')).now())
            totp = str(pyotp.TOTP(os.environ.get("RB_TOKEN")).now())
            print(f"logging in with {totp} at {str(datetime.now())}")
            # mfa_token = input("Please type in the MFA code: ")
            mfa_token=totp
            payload['mfa_code'] = mfa_token
            res = helper.request_post(url, payload, jsonify_data=False)
            while (res.status_code != 200):
                mfa_token = input(
                    "That MFA code was not correct. Please type in another MFA code: ")
                payload['mfa_code'] = mfa_token
                res = helper.request_post(url, payload, jsonify_data=False)
            data = res.json()
        elif 'challenge' in data:
            challenge_id = data['challenge']['id']
            sms_code = input('Enter Robinhood code for validation: ')
            res = respond_to_challenge(challenge_id, sms_code)
            while 'challenge' in res and res['challenge']['remaining_attempts'] > 0:
                sms_code = input('That code was not correct. {0} tries remaining. Please type in another code: '.format(
                    res['challenge']['remaining_attempts']))
                res = respond_to_challenge(challenge_id, sms_code)
            helper.update_session(
                'X-ROBINHOOD-CHALLENGE-RESPONSE-ID', challenge_id)
            data = helper.request_post(url, payload)
        # Update Session data with authorization or raise exception with the information present in data.
        if 'access_token' in data:
            token = '{0} {1}'.format(data['token_type'], data['access_token'])
            helper.update_session('Authorization', token)
            helper.set_login_state(True)
            data['detail'] = "logged in with brand new authentication code."
            if store_session:
                with open(pickle_path, 'wb') as f:
                    pickle.dump({'token_type': data['token_type'],
                                 'access_token': data['access_token'],
                                 'refresh_token': data['refresh_token'],
                                 'device_token': device_token}, f)
                print(f'updated pickle in {pickle_path}')
                # blob = bucket.blob(blob_pickle_path)
                # print(f'upload blob from {pickle_path} to {blob_pickle_path}')
                # with open(pickle_path, 'rb') as f:
                #     new_pickle = f.read()
                # blob.upload_from_string(new_pickle)
        else:
            raise Exception(data['detail'])
    else:
        raise Exception('Error: Trouble connecting to robinhood API. Check internet connection.')
    return data

if __name__ == "__main__":
    # storage_client = storage.Client()
    # bucket = storage_client.get_bucket('quantwannadb')
    # blob_pickle_path = '.tokens/robinhood.pickle'
    # blob = bucket.blob(blob_pickle_path)
    # blob_pickle = blob.download_as_string()
    # print(blob_pickle)
    # blob.upload_from_filename("C:/Users/shery/.tokens/robinhood.pickle")
    login()