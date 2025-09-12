import json
import datetime
from typing import Optional

import requests

from macrosynergy import __version__ as ms_version_info
from macrosynergy.download.exceptions import AuthenticationError


class JPMorganOAuth(object):
    """
    A lightweight helper for OAuth 2.0 (Client Credentials) authentication used by the
    JPMorgan DataQuery API, DataQuery File API, and JPMorgan Fusion API.

    This class retrieves and manages access tokens for API requests and supports
    loading credentials from a JSON file or a dictionary.

    Parameters
    ----------
    client_id : str
        The OAuth client ID issued by JPMorgan.
    client_secret : str
        The OAuth client secret issued by JPMorgan.
    resource : str
        The token audience (often sent as the `aud` parameter) identifying the
        target JPMorgan API.
    application_name : str
        The name of the application using the API (used for identification/logging only).
    root_url : str
        The base URL for the target JPMorgan API (not required for token
        retrieval but kept for completeness and potential future use).
    auth_url : str
        The full URL of the OAuth token endpoint for JPMorgan.
    proxies : Optional[Dict[str, str]]
        Optional proxies to use for the HTTP requests. Default is None.
    """

    @classmethod
    def from_credentials_json(cls, credentials_json: str):
        """
        Load OAuth credentials from a JSON file and return an instance of JPMorganOAuth.

        Parameters
        ----------
        credentials_json : str
            Path to the JSON file containing the OAuth credentials. This file must
            contain the keys 'client_id' and 'client_secret' and should include
            'resource', 'application_name', 'root_url', and 'auth_url'.

        Returns
        -------
        JPMorganOAuth
            An instance of the JPMorganOAuth class initialized with the credentials from
            the JSON file.
        """
        with open(credentials_json, "r") as f:
            credentials = json.load(f)
        return cls.from_credentials(credentials)

    @classmethod
    def from_credentials(cls, credentials: dict):
        """
        Create an instance of JPMorganOAuth from a dictionary of credentials.

        Parameters
        ----------
        credentials : dict
            A dictionary containing the OAuth credentials. It must include the keys
            'client_id' and 'client_secret', and should include 'resource',
            'application_name', 'root_url', and 'auth_url'.

        Returns
        -------
        JPMorganOAuth
            An instance of the JPMorganOAuth class initialized with the provided
            credentials.
        """
        return cls(**credentials)

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        resource: str,
        application_name: str,
        root_url: str,
        auth_url: str,
        proxies: Optional[dict] = None,
        verify: bool = True,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.resource = resource
        self.application_name = application_name
        self.root_url = root_url
        self.auth_url = auth_url
        self._verify = verify

        # none of the above can be None
        for attr_name, attr_val in [
            ("client_id", self.client_id),
            ("client_secret", self.client_secret),
            ("resource", self.resource),
            ("application_name", self.application_name),
            ("root_url", self.root_url),
            ("auth_url", self.auth_url),
        ]:
            if attr_val is None:
                raise ValueError(f"{attr_name} must be provided and cannot be None.")
            elif not isinstance(attr_val, str):
                raise TypeError(f"{attr_name} must be a string.")

        if proxies is not None and not isinstance(proxies, dict):
            raise TypeError("`proxies` must be a dictionary.")

        self.proxies = proxies or None

        # OAuth 2.0 Client Credentials grant payload
        self.token_data = {
            "grant_type": "client_credentials",
            "aud": resource,
            "client_id": client_id,
            "client_secret": client_secret,
        }

        self._stored_token = None

    def retrieve_token(self):
        """
        Retrieve an access token from the OAuth server and store it in the instance.

        Equivalent cURL request:

        .. code-block:: bash

            curl -X POST "$AUTH_URL" \\
                -d "grant_type=client_credentials" \\
                -d "aud=$RESOURCE" \\
                -d "client_id=$CLIENT_ID" \\
                -d "client_secret=$CLIENT_SECRET"
        """
        err_str = "Error retrieving token: "
        try:
            response = requests.post(
                self.auth_url,
                data=self.token_data,
                proxies=self.proxies,
                verify=self._verify,
            )
            response.raise_for_status()
            token_data = response.json()
            self._stored_token = {
                "created_at": datetime.datetime.now(datetime.timezone.utc),
                "expires_in": token_data["expires_in"],
                "access_token": token_data["access_token"],
            }
        except requests.exceptions.RequestException as exc:
            raise AuthenticationError(f"{err_str}{exc}") from exc
        except requests.exceptions.HTTPError as exc:
            if exc.response.status_code == 401:
                raise AuthenticationError(f"{err_str}{exc}") from exc

    def _is_valid_token(self):
        if self._stored_token is None:
            return False
        return self._stored_token["created_at"] + datetime.timedelta(
            seconds=self._stored_token["expires_in"]
        ) > datetime.datetime.now(datetime.timezone.utc)

    def _get_token(self):
        if not self._is_valid_token():
            self.retrieve_token()
        return self._stored_token["access_token"]

    def _get_user_id(self) -> str:
        return "OAuth_ClientID - " + self.client_id

    def get_auth(self) -> dict:
        """
        Get the authorization headers for API requests.

        Returns
        -------
        dict
            A dictionary containing the authorization headers with the access token.
        """
        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "User-Agent": f"MacrosynergyPackage/{self.application_name}/{ms_version_info}",
        }
        return headers
