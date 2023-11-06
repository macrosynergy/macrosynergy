from typing import Any, Dict, List, Optional
import requests
import os
from time import sleep

GH_OAUTH_TOKEN: Optional[str] = os.getenv("GH_TOKEN", None)

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"token {GH_OAUTH_TOKEN}",
    "X-GitHub-Api-Version": "2022-11-28",
}


def api_request(
    url: str,
    headers: Optional[dict] = HEADERS,
    params: Dict[str, Any] = {},
    method: str = "GET",
    max_retries: int = 5,
) -> Any:
    """
    Make a request to the GitHub API.

    :param <str> url: The URL to make the request to.
    :param <Dict[str, str]> headers: The headers to send with the request.
    :param <Dict[str, Any]> params: The parameters to send with the request.
    :return <Any>: The response from the API.
    """
    assert method in ["GET", "DELETE", "POST", "PATCH", "PUT"]

    retries = 0
    while retries < max_retries:
        try:
            response: requests.Response = requests.request(
                url=url, headers=headers, params=params, method=method
            )

            if response.status_code == 404:
                raise Exception(f"404: {url} not found.")
            response.raise_for_status()  # Raise exception for failed requests

            return response.json()
        except Exception as exc:
            fail_codes = [403, 404, 422]
            try:
                if response.status_code in fail_codes:
                    raise Exception(f"Request failed: {exc}")
            except Exception:
                raise Exception(f"Request failed: {exc}")

            print(f"Request failed: {exc}")
            retries += 1
            sleep(1)

    raise Exception(f"Request failed")  # If the request fails, raise an exception
