"""Authentication classes for DataQuery of OAuth and CertAuth"""
import base64
import os
import requests
import json
from typing import Union, Optional
from datetime import datetime

OAUTH_BASE_URL = "https://api-developer.jpmorgan.com/research/dataquery-authe/api/v2/"
OAUTH_TOKEN_URL = "https://authe.jpmchase.com/as/token.oauth2"
OAUTH_DQ_RESOURCE_ID = "JPMC:URI:RS-06785-DataQueryExternalApi-PROD"
CERT_BASE_URL = "https://platform.jpmorgan.com/research/dataquery/api/v2"


class CertAuth(object):
    """Certificate Authentication

    Class used to access DataQuery via certificate and private key. To access the API
    login both username & password are required as well as a certified certificate
    and private key to verify the request.

    :param <str> username: username for login to REST API for JP Morgan DataQuery.
    :param <str> password: password.
    :param <str> crt: string with location of public certificate.
    :param <str> key: string with private key location.
    """

    def __init__(
            self,
            username: str,
            password: str,
            crt: str = "api_macrosynergy_com.crt",
            key: str = "api_macrosynergy_com.key",
            base_url: str = CERT_BASE_URL
    ):

        assert isinstance(username, str), (
            f"username must be a <str> and not <{type(username)}>."
        )

        assert isinstance(password, str), (
            f"password must be a <str> and not <{type(password)}>."
        )

        self.auth = base64.b64encode(
            bytes(f'{username:s}:{password:s}', "utf-8")
        ).decode('ascii')

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

    def get_dq_api_result(self, url: str, params: dict = None):
        """
        Method used exclusively to request data from the API.
        """
        r = requests.get(
            url=url,
            cert=(self.crt, self.key),
            headers=self.headers,
            params=params
        )

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


class OAuth(object):
    """Accessing DataQuery via OAuth.

    :param <str> client_id: string with client id, username.
    :param <str> client_secret: string with client secret, password.
    :param <str> url:
    :param <str> token_url:
    :param <str> dq_resource_id:
    """

    def __init__(
            self,
            client_id: str,
            client_secret: str,
            url: str = OAUTH_BASE_URL,
            token_url: str = OAUTH_TOKEN_URL,
            dq_resource_id: str = OAUTH_DQ_RESOURCE_ID
    ):

        self.base_url: str = url
        self.__token_url: str = token_url
        self.__dq_api_resource_id: str = dq_resource_id

        assert isinstance(client_id, str), (
            f"client_id argument must be a string and not <{type(client_id)}>."
        )
        self.client_id: str = client_id

        assert isinstance(client_secret, str), (
            f"client_secret must be a str and not <{type(client_secret)}>."
        )

        self.client_secret: str = client_secret
        self.last_response: Optional[str] = None
        self._stored_token: Optional[dict] = None

    def _valid_token(self) -> bool:
        if self._stored_token is None:
            return False

        created: datetime = self._stored_token['created_at']
        expires: datetime = self._stored_token['expires_in']
        if (datetime.now() - created).total_seconds() / 60 >= expires - 1:
            return False

        return True

    def _get_token(self) -> str:
        if self._valid_token():
            return self._stored_token['access_token']

        with requests.post(
                url=self.__token_url,
                proxies={},
                data={
                    'grant_type': 'client_credentials',
                    'client_id': self.client_id,
                    'client_secret': self.client_secret,
                    'aud': self.__dq_api_resource_id
                }
        ) as r:
            assert r.ok
            js: dict = r.json()

        self._stored_token: dict = {
            'created_at': datetime.now(),
            'access_token': js['access_token'],
            'expires_in': js['expires_in']
        }

        return self._stored_token['access_token']

    def get_dq_api_result(self, url, params: dict = None):
        """Method used exclusively to request data from the API."""
        with requests.get(
                url=url,
                params=params,
                headers={'Authorization': 'Bearer ' + self._get_token()},
                proxies={}
        ) as r:
            self.last_response = r.text

        return r

    def _fetch(self, endpoint: str = "/groups", select: str = "groups",
               payload: Union[list, dict] = None, params: dict = None):
        """Used to test if DataQuery is responding.

        :param <str> endpoint: default '/groups', end-point of DataQuery to be explored.
        :param <str> select:
        :param <list/dict> payload: default None, defines payload for a 'POST' method.
        :param <dict> params:

        :return: list with response from DataQuery Web API.
        :rtype: <list>
        """

        results = []
        url = self.base_url + endpoint
        r = self.get_dq_api_result(url=url, params=params)

        self.last_response = r.text
        response = json.loads(self.last_response)

        assert select in response.keys()
        results.extend(response[select])

        if isinstance(response["info"], dict):
            results = response["info"]
            print(results['description'])

        return results
