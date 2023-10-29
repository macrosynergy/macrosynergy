from typing import List, Optional, Dict, Union, Tuple
import logging
from datetime import datetime, timedelta
from .constants import (
    OAUTH_TOKEN_URL,
    OAUTH_DQ_RESOURCE_ID,
    OAUTH_TRACKING_ID,
)

from .utils import request_wrapper
import base64
import os

logger: logging.Logger = logging.getLogger(__name__)


class OAuth(object):
    """
    Class for handling OAuth authentication for the DataQuery API.

    :param <str> client_id: client ID for the OAuth application.
    :param <str> client_secret: client secret for the OAuth application.
    :param <dict> proxy: proxy to use for requests. Defaults to None.
    :param <str> token_url: URL for getting OAuth tokens.
    :param <str> dq_resource_id: resource ID for the JPMaQS Application.

    :return <OAuth>: OAuth object.

    :raises <ValueError>: if any of the parameters are semantically incorrect.
    :raises <TypeError>: if any of the parameters are of the wrong type.
    :raises <Exception>: other exceptions may be raised by underlying functions.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        proxy: Optional[dict] = None,
        token_url: str = OAUTH_TOKEN_URL,
        dq_resource_id: str = OAUTH_DQ_RESOURCE_ID,
    ):
        logger.debug("Instantiate OAuth pathway to DataQuery")
        vars_types_zip: zip = zip(
            [client_id, client_secret, token_url, dq_resource_id],
            [
                "client_id",
                "client_secret",
                "token_url",
                "dq_resource_id",
            ],
        )

        for varx, namex in vars_types_zip:
            if not isinstance(varx, str):
                raise TypeError(f"{namex} must be a <str> and not {type(varx)}.")

        if not isinstance(proxy, dict) and proxy is not None:
            raise TypeError(f"proxy must be a <dict> and not {type(proxy)}.")

        self.token_url: str = token_url
        self.proxy: Optional[dict] = proxy

        self._stored_token: Optional[dict] = None
        self.token_data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "aud": dq_resource_id,
        }

    def _valid_token(self) -> bool:
        """
        Method to check if the stored token is valid.

        :return <bool>: True if the token is valid, False otherwise.
        """
        if self._stored_token is None:
            logger.debug("No token stored")
            return False

        created: datetime = self._stored_token["created_at"]  # utc time of creation
        expires: datetime = created + timedelta(
            seconds=self._stored_token["expires_in"]
        )

        utcnow = datetime.utcnow()
        is_active: bool = expires > utcnow

        logger.debug(
            "Active token: %s, created: %s, expires: %s, now: %s",
            is_active,
            created,
            expires,
            utcnow,
        )

        return is_active

    def _get_token(self) -> str:
        """Method to get a new OAuth token.

        :return <str>: OAuth token.
        """
        if not self._valid_token():
            logger.debug("Request new OAuth token")
            js = request_wrapper(
                url=self.token_url,
                data=self.token_data,
                method="post",
                proxy=self.proxy,
                tracking_id=OAUTH_TRACKING_ID,
                user_id=self._get_user_id(),
            )
            # on failure, exception will be raised by request_wrapper

            # NOTE : use UTC time for token expiry
            self._stored_token: dict = {
                "created_at": datetime.utcnow(),
                "access_token": js["access_token"],
                "expires_in": js["expires_in"],
            }

        return self._stored_token["access_token"]

    def _get_user_id(self) -> str:
        return "OAuth_ClientID - " + self.token_data["client_id"]

    def get_auth(self) -> Dict[str, Union[str, Optional[Tuple[str, str]]]]:
        """
        Returns a dictionary with the authentication information, in the same
        format as the `macrosynergy.download.dataquery.CertAuth.get_auth()` method.
        """
        headers: Dict = {"Authorization": "Bearer " + self._get_token()}
        return {
            "headers": headers,
            "cert": None,
            "user_id": self._get_user_id(),
        }


class CertAuth(object):
    """
    Class for handling certificate based authentication for the DataQuery API.

    :param <str> username: username for the DataQuery API.
    :param <str> password: password for the DataQuery API.
    :param <str> crt: path to the certificate file.
    :param <str> key: path to the key file.

    :return <CertAuth>: CertAuth object.

    :raises <AssertionError>: if any of the parameters are of the wrong type.
    :raises <FileNotFoundError>: if certificate or key file is missing from filesystem.
    :raises <Exception>: other exceptions may be raised by underlying functions.
    """

    def __init__(
        self,
        username: str,
        password: str,
        crt: str,
        key: str,
        proxy: Optional[dict] = None,
    ):
        for varx, namex in zip([username, password], ["username", "password"]):
            if not isinstance(varx, str):
                raise TypeError(f"{namex} must be a <str> and not {type(varx)}.")

        self.auth: str = base64.b64encode(
            bytes(f"{username:s}:{password:s}", "utf-8")
        ).decode("ascii")

        # Key and Certificate check
        for varx, namex in zip([crt, key], ["crt", "key"]):
            if not isinstance(varx, str):
                raise TypeError(f"{namex} must be a <str> and not {type(varx)}.")
            if not os.path.isfile(varx):
                raise FileNotFoundError(f"The file '{varx}' does not exist.")
        self.key: str = key
        self.crt: str = crt
        self.username: str = username
        self.password: str = password
        self.proxy: Optional[dict] = proxy

    def get_auth(self) -> Dict[str, Union[str, Optional[Tuple[str, str]]]]:
        """
        Returns a dictionary with the authentication information, in the same
        format as the `macrosynergy.download.dataquery.OAuth.get_auth()` method.
        """
        headers = {"Authorization": f"Basic {self.auth:s}"}
        user_id = "CertAuth_Username - " + self.username
        return {
            "headers": headers,
            "cert": (self.crt, self.key),
            "user_id": user_id,
        }
