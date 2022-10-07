
"""Authentication classes for DataQuery of OAuth and CertAuth."""
import base64
import os
import requests
from typing import Optional, Dict, Tuple
from datetime import datetime

CERT_BASE_URL: str = "https://platform.jpmorgan.com/research/dataquery/api/v2"
OAUTH_BASE_URL: str = (
    "https://api-developer.jpmorgan.com/research/dataquery-authe/api/v2"
)
OAUTH_TOKEN_URL: str = "https://authe.jpmchase.com/as/token.oauth2"
OAUTH_DQ_RESOURCE_ID: str = "JPMC:URI:RS-06785-DataQueryExternalApi-PROD"


def valid_response(r: requests.Response) -> dict:
    """
    Prior to requesting any data, the function will confirm if a connection to the
    DataQuery API is able to be established given the credentials passed. If the status
    code is 200, able to access DataQuery's API.
    """
    if r.status_code == 401:
        raise RuntimeError(
            f"Authentication error - unable to access DataQuery:\n{r.text}"
        )

    elif r.text[0] != "{":

        # Authentication check.
        condition: str = r.text.split("-")[1].strip().split("<")[0]
        if condition == "Authentication Failure":
            raise RuntimeError(
                condition + " - unable to access DataQuery. Password expired."
            )

    assert r.ok, (
        f"Access issue status code {r.status_code},"
        f" headers: {r.headers}, text: {r.text} for url {r.url}."
    )

    return r.json()


def dq_request(
    url: str,
    headers: dict = None,
    params: dict = None,
    method: str = "get",
    cert: Optional[Tuple[str, str]] = None,
    **kwargs
) -> Tuple[dict, str, str]:
    """Will return the request from DataQuery.

    """
    request_error = f"Unknown request method {method} not in ('get', 'post')."
    assert method in ("get", "post"), request_error

    with requests.request(
        method=method,
        url=url,
        cert=cert,
        headers=headers,
        params=params,
        **kwargs,
    ) as r:
        text: str = r.text
        last_url: str = r.url
        js: dict = valid_response(r=r)

    return js, text, last_url


class CertAuth(object):
    """Certificate Authentication.

    Class used to access DataQuery via certificate and private key. To access the API
    login both username & password are required as well as a certified certificate and
    private key to verify the request.

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

        error_user = f"username must be a <str> and not <{type(username)}>."
        assert isinstance(username, str), error_user

        error_password = f"password must be a <str> and not <{type(password)}>."
        assert isinstance(password, str), error_password

        self.auth: str = base64.b64encode(
            bytes(f"{username:s}:{password:s}", "utf-8")
        ).decode("ascii")

        self.headers: Dict[str, str] = {"Authorization": f"Basic {self.auth:s}"}
        self.base_url: str = base_url

        # Key and Certificate.
        self.key: str = self.valid_path(key, "key")
        self.crt: str = self.valid_path(crt, "crt")

        # For debugging purposes save last request response.
        self.status_code: Optional[int] = None
        self.last_response: Optional[str] = None
        self.last_url: Optional[str] = None

    @staticmethod
    def valid_path(directory: str, file_type: str) -> Optional[str]:
        """Validates the key & certificate exist in the referenced directory.

        :param <str> directory: directory hosting the respective files.
        :param <str> file_type: parameter used to distinguish between the certificate or
            key being received.

        """
        dir_error = f"{file_type:s} file must be a <str> not <{type(directory)}>."
        assert isinstance(directory, str), dir_error

        condition = (os.path.exists(directory) and os.path.isfile(directory))
        if not condition:
            raise OSError(
                f"The directory received, {directory}, does not contain the "
                f"respective file, {file_type}."
            )

        return directory

    def get_dq_api_result(self, url: str, params: dict = None) -> dict:
        """Method used exclusively to request data from the API.

        :param <str> url: url to access DQ API.
        :param <dict> params: dictionary containing the required parameters for the
            ticker series.
        """
        js, self.last_response, self.last_url = dq_request(
            url=url, cert=(self.crt, self.key), headers=self.headers, params=params
        )

        return js


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

        id_error = f"client_id argument must be a <str> and not <{type(client_id)}>."
        assert isinstance(client_id, str), id_error
        self.client_id: str = client_id

        secret_error = f"client_secret must be a str and not <{type(client_secret)}>."
        assert isinstance(client_secret, str), secret_error

        self.client_secret: str = client_secret
        self._stored_token: Optional[dict] = None
        self.token_data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "aud": self.__dq_api_resource_id
        }

        # For debugging purposes save last request response.
        self.status_code: Optional[int] = None
        self.last_response: Optional[str] = None
        self.last_url: Optional[str] = None

    def _active_token(self) -> bool:
        """Confirms if the token being used has not expired.
        """
        created: datetime = self._stored_token["created_at"]
        expires: int = self._stored_token["expires_in"]

        return (datetime.now() - created).total_seconds() / 60 >= (expires - 1)

    def _valid_token(self) -> bool:
        """Confirms if the credentials passed correspond to a valid token.
        """
        return not (self._stored_token is None or self._active_token())

    def _get_token(self) -> str:
        """Retrieves the token which is used to access DataQuery via OAuth method.
        """

        if not self._valid_token():

            js, self.last_response, self.last_url = dq_request(
                url=self.__token_url,
                data=self.token_data,
                proxies={},
                method="post")
            self._stored_token: dict = {
                "created_at": datetime.now(),
                "access_token": js["access_token"],
                "expires_in": js["expires_in"]
            }

        return self._stored_token["access_token"]

    def get_dq_api_result(self, url: str, params: dict = None) -> dict:
        """Method used exclusively to request data from the API.

        :param <str> url: url to access DQ API.
        :param <dict> params: dictionary containing the required parameters for the
            ticker series.
        """

        js, self.last_response, self.last_url = dq_request(
            url=url,
            params=params,
            headers={"Authorization": "Bearer " + self._get_token()},
            proxies={},
        )

        return js
