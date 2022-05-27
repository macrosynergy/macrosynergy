
"""
DataQuery Interfaces and API wrappers.
"""
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import json
import logging
from typing import Union
# The functools module is for higher-order functions: functions that act on or return
# other functions.
from functools import partial

logger = logging.getLogger(__name__)

BASE_URL = "https://api-developer.jpmorgan.com/research/dataquery-authe/api/v2/"

class DataQueryWebAPI(object):
    """
    DataQuery REST web API class

    For unittesting mock up methods

        1. _create_api (return: mp_api)
        2. _renew_token (mb_api fetch_token)
        3. _fetch (mb_api get and post)

    :param <str> client_id: string with client id, username.
    :param <str> client_secret: string with client secret, password.
    """

    def __init__(self, client_id: str, client_secret: str, token: str):
        logger.info("Establish connection to DataQuery Web API")
        self.__is_connected: bool = False

        self.__token_url = "https://authe.jpmchase.com/as/token.oauth2"
        self.__dq_api_resource_id = 'JPMC:URI:RS-06785-DataQueryExternalApi-PROD'

        id_error = f"client_id argument must be a string and not <{type(client_id)}>."
        assert isinstance(client_id, str), id_error
        self.client_id: str = client_id

        secret_error = f"client_secret must be a str and not <{type(client_secret)}>."
        assert isinstance(client_secret, str), secret_error

        self.client_secret: str = client_secret

        self.mb_api = self._create_api()
        if not self.mb_api.authorized:
            self._renew_token()

    def _set_is_connected(self, connected: bool):
        self.__is_connected = connected

    def is_connected(self):
        return self.__is_connected

    def _create_api(self):
        client = BackendApplicationClient(client_id=self.client_id)
        mb_api = OAuth2Session(client=client)
        return mb_api

    def _renew_token(self):
        self.mb_api.fetch_token(
            token_url=self.__token_url,
            client_id=self.client_id,
            client_secret=self.client_secret,
            aud=self.__dq_api_resource_id
        )
        self._set_is_connected(connected=self.mb_api.authorized)

    def _fetch(self, request: str, method: str = "POST",
               payload: Union[list, dict] = None, params: dict = None):
        """

        :param <str> request: string with request elements and parameters.
        :param <str> method: default 'GET', either 'GET' or 'POST' http methods.
        :param <list/dict> payload: default None, defines payload for a 'POST' method.

        :return: list with response from DataQuery Web API.
        :rtype: <list[dict]>
        """

        assert isinstance(request, str), f"request of unknown type {type(request)}: {request}"

        assert isinstance(
            method, str
        ), f"method of unknown type {type(method)}: {method}"

        # Helper method that will request new access token if required
        url = BASE_URL + request

        # This is typically the first time when we do not yet have a token
        if not self.mb_api.authorized:
            self._renew_token()

        if method == "GET":
            call_url = self.mb_api.get

        elif method == "POST":
            headers = {"Content-Type": "application/json"}
            call_url = partial(self.mb_api.post, headers=headers, params=params,
                               data=json.dumps(payload))
        else:
            raise NotImplementedError(f"Unknown method; {method:s}")

        try:
            r = call_url(url)
        except:
            self._renew_token()
            r = call_url(url)

        if r.status_code == 401:
            # If authorization failed, it is likely that the token has expired.
            # Get a new one and try again.
            self._renew_token()
            r = call_url(url)

        error_status = f"Status code {r.status_code} for request {request:s}"
        assert (r.status_code == 200), error_status

        json_data = json.loads(r.content)

        return json_data