"""Authentication classes for DataQuery of OAuth and CertAuth"""
import base64
import os
import requests
from typing import Optional, Dict
from datetime import datetime

OAUTH_BASE_URL: str = "https://api-developer.jpmorgan.com/research/dataquery-authe/api/v2/"
OAUTH_TOKEN_URL: str = "https://authe.jpmchase.com/as/token.oauth2"
OAUTH_DQ_RESOURCE_ID: str = "JPMC:URI:RS-06785-DataQueryExternalApi-PROD"
CERT_BASE_URL: str = "https://platform.jpmorgan.com/research/dataquery/api/v2"


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

        self.auth: str = base64.b64encode(
            bytes(f'{username:s}:{password:s}', "utf-8")
        ).decode('ascii')

        self.headers: Dict[str, str] = {"Authorization": f"Basic {self.auth:s}"}
        self.base_url: str = base_url

        # Key and Certificate.
        self.key: str = self.valid_path(key, "key")
        self.crt: str = self.valid_path(crt, "crt")

        # For debugging purposes save last request response
        self.status_code: Optional[int] = None
        self.last_response: Optional[str] = None
        self.last_url: Optional[str] = None

    @staticmethod
    def valid_path(directory: str, file_type: str) -> Optional[str]:
        """Validates the key & certificate exist in the referenced directory."""
        assert isinstance(directory, str), (
            f"{file_type:s} file must be a <str> not {directory}{type(directory)})."
        )

        if not os.path.exists(directory) and os.path.isfile(directory):
            # TODO should this not raise an error (exception)?
            print(f"Not a valid file path on the system: {directory} - return None")
            return None

        return directory

    def get_dq_api_result(self, url: str, params: dict = None) -> requests.Response:
        """Method used exclusively to request data from the API."""
        with requests.get(
            url=url,
            cert=(self.crt, self.key),
            headers=self.headers,
            params=params
        ) as r:
            self.status_code: int = r.status_code
            self.last_response: str = r.text
            self.last_url: str = r.url

        return r


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
        self._stored_token: Optional[dict] = None

        # For debugging purposes save last request response
        self.status_code: Optional[int] = None
        self.last_response: Optional[str] = None
        self.last_url: Optional[str] = None

    def _valid_token(self) -> bool:
        if self._stored_token is None:
            return False

        created: datetime = self._stored_token['created_at']
        expires: int = self._stored_token['expires_in']  # Minutes
        if (datetime.now() - created).total_seconds() / 60 >= (expires - 1):
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

    def get_dq_api_result(self, url: str, params: dict = None):
        """Method used exclusively to request data from the API."""
        with requests.get(
                url=url,
                params=params,
                headers={'Authorization': 'Bearer ' + self._get_token()},
                proxies={}
        ) as r:
            self.last_response: str = r.text
            self.status_code: int = r.status_code
            self.last_url: str = r.url

        return r
