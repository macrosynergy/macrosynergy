
import base64
import os
import requests
import json

BASE_URL = "https://platform.jpmorgan.com/research/dataquery/api/v2"

class CertAuth(object):
    """
    Class used to access DataQuery via certificate and private key. To access the API
    login both username & password are required as well as a certified certificate
    and private key to verify the request.

    For unittesting mock up methods

        1. _create_api (return: mp_api)
        2. _renew_token (mb_api fetch_token)
        3. _fetch (mb_api get and post)

    :param <str> username: username for login to REST API for
        JP Morgan DataQuery.
    :param <str> password: password.
    :param <str> crt: string with location of public certificate.
    :param <str> key: string with private key location.
    """

    def __init__(self, username: str,
                 password: str,
                 crt: str = "api_macrosynergy_com.crt",
                 key: str = "api_macrosynergy_com.key",
                 base_url: str = BASE_URL):

        u_error = f"username must be a <str> and not <{type(username)}>."
        assert isinstance(username, str), u_error

        u_password = f"password must be a <str> and not <{type(password)}>."
        assert isinstance(password, str), u_password
        self.auth = base64.b64encode(bytes(f'{username:s}:{password:s}',
                                           "utf-8")).decode('ascii')
        self.headers = {"Authorization": f"Basic {self.auth:s}"}

        self.base_url = base_url

        # Key and Certificate.
        self.key = self.valid_path(key, "key")

        self.crt = self.valid_path(crt, "crt")
        self.last_response = None

    @staticmethod
    def valid_path(directory: str, file_type: str):
        """
        Validates the key & certificate exist in the referenced directory.
        """
        error_type = file_type + " file must be a <str>."
        assert isinstance(directory, str), error_type

        try:
            os.path.exists(directory)
        except OSError as err:
            print("OS error: {0}".format(err))
        else:
            return directory

    def get_dq_api_result(self, url, params: dict = None):
        """
        Method used exclusively to request data from the API.
        """
        r = requests.get(url=url, cert=(self.crt, self.key),
                         headers=self.headers, params=params)

        return r

    def _fetch(self, endpoint: str = "/groups", select: str = "groups",
               params: dict = None):
        """
        Used to test if DataQuery is responding.

        :param <str> endpoint: default '/groups', end-point of DataQuery to be explored.
        :param <str> select: default 'groups' string with select for within the endpoint.
        :param <str> params: dictionary of parameters to be passed to request

        :return: list of response from DataQuery
        :rtype: <list>

        """

        url = self.base_url + endpoint
        self.last_url = url

        results = []

        auth_check = lambda string: string.split('-')[1].strip().split('<')[0]

        r = self.get_dq_api_result(url=url, params=params)
        self.status_code = r.status_code
        self.last_response = r.text

        if self.last_response[0] != "{":
            condition = auth_check(self.last_response)

            error = condition + " - unable to access DataQuery. Password expired."
            if condition == 'Authentication Failure':
                raise RuntimeError(error)

        self.last_url = r.url

        response = json.loads(self.last_response)

        assert select in response.keys()
        results.extend(response[select])

        if isinstance(response["info"], dict):
            results = response["info"]
            print(results['description'])

        return results