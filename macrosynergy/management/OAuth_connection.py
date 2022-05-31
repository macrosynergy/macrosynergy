
"""
DataQuery Interfaces and API wrappers.
"""
import requests
import json
from typing import Union
from datetime import datetime

BASE_URL = "https://api-developer.jpmorgan.com/research/dataquery-authe/api/v2/"
TOKEN_URL = "https://authe.jpmchase.com/as/token.oauth2"
DQ_RESOURCE_ID = "JPMC:URI:RS-06785-DataQueryExternalApi-PROD"
_stored_token = None

class DataQueryOAuth(object):
    """
    DataQuery REST web API class

    For unittesting mock up methods

        1. _create_api (return: mp_api)
        2. _renew_token (mb_api fetch_token)
        3. _fetch (mb_api get and post)

    :param <str> client_id: string with client id, username.
    :param <str> client_secret: string with client secret, password.
    """

    def __init__(self, client_id: str, client_secret: str, url: str = BASE_URL,
                 token_url: str = TOKEN_URL, dq_resource_id: str = DQ_RESOURCE_ID):

        self.base_url = url

        self.__token_url = token_url
        self.__dq_api_resource_id = dq_resource_id

        id_error = f"client_id argument must be a string and not <{type(client_id)}>."
        assert isinstance(client_id, str), id_error
        self.client_id: str = client_id

        secret_error = f"client_secret must be a str and not <{type(client_secret)}>."
        assert isinstance(client_secret, str), secret_error

        self.client_secret: str = client_secret

        self.last_response = None

    def _get_token(self):

        global _stored_token

        if _stored_token is None or ((datetime.now() - _stored_token[
            'created_at']).total_seconds() / 60) >= (_stored_token['expires_in'] - 1):
            with requests.post(
                url=self.__token_url,
                proxies={},
                data={'grant_type': 'client_credentials', 'client_id': self.client_id,
                      'client_secret': self.client_secret,
                      'aud': self.__dq_api_resource_id}
            ) as r:
                json = r.json()

                _stored_token = {'created_at': datetime.datetime.now(),
                                 'access_token': json['access_token'],
                                 'expires_in': json['expires_in']}

        return _stored_token['access_token']

    def get_dq_api_result(self, url, params: dict = None):
        """
        Method used exclusively to request data from the API.
        """
        with requests.get(url=url, params=params,
                         headers={'Authorization': 'Bearer ' + self._get_token()},
                         proxies={}) as r:
            self.last_response = r.text

        return r

    def _fetch(self, endpoint: str = "/groups", select: str = "groups",
               payload: Union[list, dict] = None, params: dict = None):
        """
        Used to test if DataQuery is responding.

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