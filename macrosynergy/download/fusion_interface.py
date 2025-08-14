import requests
import json
import datetime
import time
import logging
import os
from pathlib import Path
import io
import warnings
import functools
import operator
import concurrent.futures as cf
from typing import Dict, Optional, TypeVar, Any, List, Union, Callable

import pandas as pd

import pyarrow as pa  # noqa: F401
import pyarrow.dataset as pa_ds
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.csv as pa_csv

from macrosynergy import __version__ as ms_version_info

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.download.exceptions import NoContentError

FUSION_AUTH_URL: str = "https://authe.jpmorgan.com/as/token.oauth2"
FUSION_ROOT_URL: str = "https://fusion.jpmorgan.com/api/v1"
FUSION_RESOURCE_ID: str = "JPMC:URI:RS-93742-Fusion-PROD"
FUSION_API_DELAY = 1.0  # seconds
CACHE_TTL = 60  # seconds
LAST_API_CALL: Optional[datetime.datetime] = None

logger = logging.getLogger(__name__)


class FusionOAuth(object):
    """
    A class to handle OAuth authentication for the JPMorgan Fusion API.
    This class retrieves and manages access tokens for API requests.
    It supports loading credentials from a JSON file or a dictionary.

    Parameters
    ----------
    client_id : str
        The client ID for the OAuth application.
    client_secret : str
        The client secret for the OAuth application.
    resource : str
        The resource ID for the Fusion API. Default is the global constant
        FUSION_RESOURCE_ID.
    application_name : str
        The name of the application using the Fusion API. Default is "fusion".
    root_url : str
        The root URL for the Fusion API. Default is the global constant
        FUSION_ROOT_URL.
    auth_url : str
        The URL for the OAuth authentication endpoint. Default is the global constant
        FUSION_AUTH_URL.
    proxies : Optional[Dict[str, str]]
        Optional proxies to use for the HTTP requests. Default is None.
    """

    @staticmethod
    def from_credentials_json(credentials_json: str):
        """
        Load OAuth credentials from a JSON file and return an instance of FusionOAuth.

        Parameters
        ----------
        credentials_json : str
            Path to the JSON file containing the OAuth credentials. This file must
            contain the keys 'client_id' and 'client_secret'.

        Returns
        -------
        FusionOAuth
            An instance of the FusionOAuth class initialized with the credentials from the
            JSON file.
        """
        with open(credentials_json, "r") as f:
            credentials = json.load(f)
        return FusionOAuth.from_credentials(credentials)

    @staticmethod
    def from_credentials(credentials: dict):
        """
        Create an instance of FusionOAuth from a dictionary of credentials.

        Parameters
        ----------
        credentials : dict
            A dictionary containing the OAuth credentials. It must include the keys
            'client_id' and 'client_secret'.

        Returns
        -------
        FusionOAuth
            An instance of the FusionOAuth class initialized with the provided credentials.
        """
        return FusionOAuth(**credentials)

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        resource: str = FUSION_RESOURCE_ID,
        application_name: str = "fusion",
        root_url: str = FUSION_ROOT_URL,
        auth_url: str = FUSION_AUTH_URL,
        proxies: Optional[dict] = None,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.resource = resource
        self.application_name = application_name
        self.root_url = root_url
        self.auth_url = auth_url

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

        self.proxies = proxies or None

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

            curl -X POST "https://authe.jpmorgan.com/as/token.oauth2" \\
                -d "grant_type=<FUSION_RESOURCE_ID>&client_id=<CLIENT_ID>&client_secret=<CLIENT_SECRET>"
        """
        try:
            response = requests.post(
                self.auth_url,
                data=self.token_data,
                proxies=self.proxies,
            )
            response.raise_for_status()
            token_data = response.json()
            self._stored_token = {
                "created_at": datetime.datetime.now(),
                "expires_in": token_data["expires_in"],
                "access_token": token_data["access_token"],
            }
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error retrieving token: {e}") from e

    def _is_valid_token(self):
        if self._stored_token is None:
            return False
        return (
            self._stored_token["created_at"]
            + datetime.timedelta(seconds=self._stored_token["expires_in"])
            > datetime.datetime.now()
        )

    def _get_token(self):
        if not self._is_valid_token():
            self.retrieve_token()
        return self._stored_token["access_token"]

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
            "User-Agent": f"MacrosynergyPackage/{ms_version_info}",
        }
        return headers


CachedType = TypeVar("CachedType", bound=Callable[..., Any])


def cache_decorator(
    ttl: int = 60, *, maxsize: Optional[int] = None
) -> Callable[[CachedType], CachedType]:
    """
    Decorator to cache the result of a function for up to `ttl` seconds total.
    Once any call happens at least `ttl` seconds after the last clear, the ENTIRE
    cache is flushed before proceeding.

    Parameters
    ----------
    ttl : int
        Time-to-live for the cache in seconds. After this time, the cache will be cleared.
        Default is 60 seconds.
    maxsize : Optional[int]
        Maximum size of the cache. If None, the default size is used.
    """

    def decorator(func: CachedType) -> CachedType:
        # wrap the function itself in an LRU cache
        cached_func = functools.lru_cache(maxsize=maxsize)(func)
        last_clear = time.time()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_clear
            now = time.time()
            # if TTL has expired, clear everything
            if now - last_clear >= ttl:
                cached_func.cache_clear()
                last_clear = now
            # call the cached version
            return cached_func(*args, **kwargs)

        # expose cache_clear for manual use if desired
        wrapper.cache_clear = cached_func.cache_clear  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def _wait_for_api_call() -> bool:
    """
    Wait for the appropriate time before making an API call to avoid hitting the rate
    limit. This function checks the time since the last API call and sleeps if necessary
    to ensure that the next call is made after the defined delay (FUSION_API_DELAY).
    Uses a global variable `LAST_API_CALL` to track the last call time.
    """
    global LAST_API_CALL
    if LAST_API_CALL is None:
        LAST_API_CALL = datetime.datetime.now()
        return True
    diff = datetime.datetime.now() - LAST_API_CALL
    sleep_for = FUSION_API_DELAY - diff.total_seconds()
    if sleep_for > 0:
        logger.info(f"Sleeping for {sleep_for:.2f} seconds to avoid API rate limit.")
        time.sleep(sleep_for)
    LAST_API_CALL = datetime.datetime.now()
    return True


def request_wrapper(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Any] = None,
    json_payload: Optional[Dict[str, Any]] = None,
    proxies: Optional[Dict[str, str]] = None,
    as_json: Optional[bool] = None,
    as_bytes: Optional[bool] = None,
    as_text: Optional[bool] = None,
) -> Union[Dict[str, Any], str, bytes]:
    """
    A wrapper function for making API requests to the JPMorgan Fusion API.
    """
    if not isinstance(method, str):
        raise TypeError("Method must be a string.")
    if method not in ["GET", "POST", "PUT", "DELETE"]:
        raise ValueError(
            f"Invalid method: {method}. Must be one of 'GET', 'POST', 'PUT', 'DELETE'."
        )

    as_flags = [as_bytes, as_text, as_json]
    check_flags = sum(map(bool, as_flags))
    if check_flags > 1:
        raise ValueError("Only one of `as_json`, `as_bytes`, or `as_text` can be True.")
    if not check_flags:
        as_json = True
    raw_response: Optional[requests.Response] = None
    try:
        _wait_for_api_call()
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            params=params,
            data=data,
            json=json_payload,
            proxies=proxies,
        )
        raw_response = response
        response.raise_for_status()

        if response.status_code == 204 or not response.content:
            raise NoContentError(
                f"No content returned for {method} {url}. Response status code: {response.status_code}"
            )

        if as_bytes:
            return response.content
        if as_text:
            return response.text

        return response.json()

    except requests.exceptions.HTTPError as e_http:
        actual_method: str = (
            e_http.request.method
            if hasattr(e_http, "request") and e_http.request
            else method
        )
        actual_url: str = (
            e_http.response.url
            if hasattr(e_http, "response") and e_http.response
            else url
        )

        error_details: str = (
            f"API HTTP error for {actual_method} {actual_url}: {e_http}"
        )
        if hasattr(e_http, "response") and e_http.response is not None:
            error_details += f"\nStatus Code: {e_http.response.status_code}\nResponse: {e_http.response.text[:500]}"
        raise Exception(error_details) from e_http

    except requests.exceptions.RequestException as e_req:
        error_details = f"API request failed for {method} {url}: {e_req}"
        if hasattr(e_req, "response") and e_req.response is not None:
            error_details += f"\nStatus Code: {e_req.response.status_code}\nResponse: {e_req.response.text[:500]}"
        raise Exception(error_details) from e_req

    except json.JSONDecodeError as e_json:
        error_details = f"Failed to decode JSON response from {method} {url}: {e_json}"
        if raw_response:
            error_details += f"\nResponse text: {raw_response.text[:500]}"
        raise Exception(error_details) from e_json


def request_wrapper_stream_bytes_to_disk(
    filename: str,
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Any] = None,
    json_payload: Optional[Dict[str, Any]] = None,
    proxies: Optional[Dict[str, str]] = None,
    chunk_size: int = 8192,
) -> None:
    """
    Stream a request's response bytes directly to disk, chunk by chunk.

    Parameters
    ----------
    filename : str
        The file path to write the streamed bytes to.
    url : str
        The URL to request.
    method : str
        HTTP method. Only GET is allowed for streaming to disk.
    headers : dict, optional
        HTTP headers.
    params : dict, optional
        Query parameters.
    data : any, optional
        Data to send in the body.
    json_payload : dict, optional
        JSON data to send in the body.
    proxies : dict, optional
        Proxies to use for the request.
    chunk_size : int
        Size of each chunk to write (default 8192).
    """
    if not isinstance(method, str):
        raise TypeError("Method must be a string.")
    if method.upper() != "GET":
        raise ValueError(
            f"Invalid method: {method}. Must be 'GET' for streaming to disk."
        )
    _wait_for_api_call()
    with requests.request(
        method=method.upper(),
        url=url,
        headers=headers,
        params=params,
        data=data,
        json=json_payload,
        proxies=proxies,
        stream=True,
    ) as response:
        response.raise_for_status()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


class SimpleFusionAPIClient:
    def __init__(
        self,
        oauth_handler: FusionOAuth,
        base_url: str = FUSION_ROOT_URL,
        proxies: Optional[Dict[str, str]] = None,
    ):
        if not isinstance(oauth_handler, FusionOAuth):
            raise TypeError("oauth_handler must be an instance of FusionOAuth.")
        self.oauth_handler: FusionOAuth = oauth_handler
        self.base_url: str = base_url.rstrip("/")
        if proxies is not None:
            if proxies != self.oauth_handler.proxies:
                proxy_warning = "Proxies defined for OAuth handler are different from the ones defined for the downloader."
                warnings.warn(proxy_warning)
            self.proxies: Optional[Dict[str, str]] = proxies
        else:
            self.proxies: Optional[Dict[str, str]] = None

    def _request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        json_payload: Optional[Dict[str, Any]] = None,
        as_json: Optional[bool] = None,
        as_bytes: Optional[bool] = None,
        as_text: Optional[bool] = None,
        timestamp: Optional[Any] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        url: str = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers: Dict[str, str] = self.oauth_handler.get_auth()
        if timestamp:
            # timestamp is solely for cache busting purposes
            pass  # pragma: no cover

        return request_wrapper(
            method=method,
            url=url,
            headers=headers,
            params=params,
            data=data,
            json_payload=json_payload,
            proxies=self.proxies,
            as_json=as_json,
            as_bytes=as_bytes,
            as_text=as_text,
            **kwargs,
        )

    @cache_decorator(CACHE_TTL)
    def get_common_catalog(self, **kwargs) -> Dict[str, Any]:
        """
        Get the common catalog from the JPMorgan Fusion API.

        Equivalent cURL request:

        .. code-block:: bash

            curl -X GET "https://fusion.jpmorgan.com/api/v1/catalogs/common" \\
                -H "Authorization: Bearer <ACCESS_TOKEN>"
                
        Returns
        -------
        Dict[str, Any]
            API response containing the common catalog.
        """
        # /v1/catalogs/common
        endpoint: str = "catalogs/common"
        return self._request(method="GET", endpoint=endpoint, **kwargs)

    @cache_decorator(CACHE_TTL)
    def get_products(self, **kwargs) -> Dict[str, Any]:
        """
        Get the list of products available in the JPMorgan Fusion API.
        
        Equivalent cURL request:

        .. code-block:: bash

            curl -X GET "https://fusion.jpmorgan.com/api/v1/catalogs/common/products" \\
                -H "Authorization: Bearer <ACCESS_TOKEN>"
                
        Returns
        -------
        Dict[str, Any]
            API response containing the list of products.
        """
        # /v1/catalogs/common/products
        endpoint: str = "catalogs/common/products"
        return self._request(method="GET", endpoint=endpoint, **kwargs)

    @cache_decorator(CACHE_TTL)
    def get_product_details(
        self, product_id: str = "JPMAQS", **kwargs
    ) -> Dict[str, Any]:
        """
        Get the details of a specific product by its ID.
        
        Equivalent cURL request:
        
        .. code-block:: bash
        
            curl -X GET "https://fusion.jpmorgan.com/api/v1/catalogs/common/products/{product_id}" \\
                -H "Authorization: Bearer <ACCESS_TOKEN>"
                
        Parameters
        ----------
        product_id : str
            The ID of the product to retrieve details for. Default is "JPMAQS".
            
        **kwargs : dict
            Additional keyword arguments to pass to the API request.

        Returns
        -------
        Dict[str, Any]
            API response containing the product details.
        """
        # /v1/catalogs/common/products/{product_id}
        endpoint: str = f"catalogs/common/products/{product_id}"
        return self._request(method="GET", endpoint=endpoint, **kwargs)

    @cache_decorator(CACHE_TTL)
    def get_dataset(self, catalog: str, dataset: str, **kwargs) -> Dict[str, Any]:
        """
        Get the details of a specific dataset from a specified catalog.
        
        Equivalent cURL request:
        
        .. code-block:: bash
        
            curl -X GET "https://fusion.jpmorgan.com/api/v1/catalogs/{catalog}/datasets/{dataset}" \\
                -H "Authorization: Bearer <ACCESS_TOKEN>"
                
        Parameters
        ----------
        catalog : str
            The catalog from which to retrieve the dataset.
        dataset : str
            The identifier of the dataset to retrieve.
        **kwargs : dict
            Additional keyword arguments to pass to the API request.
            
        Returns
        -------
        Dict[str, Any]
            API response containing the dataset details.       
        """
        # /v1/catalogs/{catalog}/datasets/{dataset}
        endpoint: str = f"catalogs/{catalog}/datasets/{dataset}"
        return self._request(method="GET", endpoint=endpoint, **kwargs)

    @cache_decorator(CACHE_TTL)
    def get_dataset_series(
        self, catalog: str, dataset: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Get the series available for a specific dataset in a specified catalog.
        
        Equivalent cURL request:
        
        .. code-block:: bash

            curl -X GET "https://fusion.jpmorgan.com/api/v1/catalogs/{catalog}/datasets/{dataset}/datasetseries" \\
                -H "Authorization: Bearer <ACCESS_TOKEN>"
                
        Parameters
        ----------
        catalog : str
            The catalog from which to retrieve the dataset series.
        dataset : str
            The identifier of the dataset for which to retrieve series.
        **kwargs : dict
            Additional keyword arguments to pass to the API request.
        
        Returns
        -------
        Dict[str, Any]
            API response containing the dataset series details.
        
        """
        # /v1/catalogs/{catalog}/datasets/{dataset}/datasetseries
        endpoint: str = f"catalogs/{catalog}/datasets/{dataset}/datasetseries"
        return self._request(method="GET", endpoint=endpoint, **kwargs)

    @cache_decorator(CACHE_TTL)
    def get_dataset_seriesmember(
        self, catalog: str, dataset: str, seriesmember: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Get the details of a specific series member in a dataset from a specified catalog.
        
        Equivalent cURL request:
        
        .. code-block:: bash

            curl -X GET "https://fusion.jpmorgan.com/api/v1/catalogs/{catalog}/datasets/{dataset}/datasetseries/{seriesmember}" \\
                -H "Authorization: Bearer <ACCESS_TOKEN>"
                
        Parameters
        ----------
        catalog : str
            The catalog from which to retrieve the series member.
        dataset : str
            The identifier of the dataset containing the series member.
        seriesmember : str
            The identifier of the series member to retrieve.
        **kwargs : dict
            Additional keyword arguments to pass to the API request.
            
        Returns
        -------
        Dict[str, Any]
            API response containing the details of the specified series member.
        """
        # /v1/catalogs/{catalog}/datasets/{dataset}/datasetseries/{seriesmember}
        endpoint: str = (
            f"catalogs/{catalog}/datasets/{dataset}/datasetseries/{seriesmember}"
        )
        return self._request(method="GET", endpoint=endpoint, **kwargs)

    @cache_decorator(CACHE_TTL)
    def get_seriesmember_distributions(
        self, catalog: str, dataset: str, seriesmember: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Get the distributions available for a specific series member in a dataset from a
        specified catalog.
        
        Equivalent cURL request:
        
        .. code-block:: bash
        
            curl -X GET "https://fusion.jpmorgan.com/api/v1/catalogs/{catalog}/datasets/{dataset}/datasetseries/{seriesmember}/distributions" \\
                -H "Authorization: Bearer <ACCESS_TOKEN>"
        
        Parameters
        ----------
        catalog : str
            The catalog from which to retrieve the series member distributions.
        dataset : str
            The identifier of the dataset containing the series member.
        seriesmember : str
            The identifier of the series member for which to retrieve distributions.
        **kwargs : dict
            Additional keyword arguments to pass to the API request.
            
        Returns
        -------
        Dict[str, Any]
            API response containing the available distributions for the specified series
            member.
        """
        # /v1/catalogs/{catalog}/datasets/{dataset}/datasetseries/{seriesmember}/distributions
        endpoint: str = f"catalogs/{catalog}/datasets/{dataset}/datasetseries/{seriesmember}/distributions"
        return self._request(method="GET", endpoint=endpoint, **kwargs)

    def get_seriesmember_distribution_details(
        self, catalog: str, dataset: str, seriesmember: str, distribution: str, **kwargs
    ) -> Union[Dict[str, Any], bytes, str]:
        """
        Get the details of a specific distribution for a series member in a dataset from
        a specified catalog.
        
        Equivalent cURL request:
        
        .. code-block:: bash
        
            curl -X GET "https://fusion.jpmorgan.com/api/v1/catalogs/{catalog}/datasets/{dataset}/datasetseries/{seriesmember}/distributions/{distribution}" \\
                -H "Authorization: Bearer <ACCESS_TOKEN>"
        
        Parameters
        ----------
        catalog : str
            The catalog from which to retrieve the series member distribution.
        dataset : str
            The identifier of the dataset containing the series member.
        seriesmember : str
            The identifier of the series member for which to retrieve the distribution.
        distribution : str
            The identifier of the distribution to retrieve (e.g., "parquet").
        **kwargs : dict
            Additional keyword arguments to pass to the API request.
        
        Returns
        -------
        Union[Dict[str, Any], bytes, str]
            API response containing the distribution details (the actual data) for the
            specified series member.
        """
        # /v1/catalogs/{catalog}/datasets/{dataset}/datasetseries/{seriesmember}/distributions/{distribution}
        endpoint: str = f"catalogs/{catalog}/datasets/{dataset}/datasetseries/{seriesmember}/distributions/{distribution}"
        return self._request(method="GET", endpoint=endpoint, **kwargs)

    def get_seriesmember_distribution_details_to_disk(
        self,
        filename: str,
        catalog: str,
        dataset: str,
        seriesmember: str,
        distribution: str = "parquet",
        **kwargs,
    ) -> None:
        """
        Download the distribution for a specific series member in a dataset from a
        specified catalog and save it to disk.

        Parameters
        ----------
        filename : str
            The file path to save the downloaded distribution data.
        catalog : str
            The catalog from which to retrieve the series member distribution.
        dataset : str
            The identifier of the dataset containing the series member.
        seriesmember : str
            The identifier of the series member for which to download the distribution.
        distribution : str
            The identifier of the distribution to download (e.g., "parquet").
        **kwargs : dict
            Additional keyword arguments to pass to the API request.

        Returns
        -------
        None
            The downloaded data is saved directly to the specified file.
        """
        # /v1/catalogs/{catalog}/datasets/{dataset}/datasetseries/{seriesmember}/distributions/{distribution}
        endpoint: str = f"catalogs/{catalog}/datasets/{dataset}/datasetseries/{seriesmember}/distributions/{distribution}"
        headers: Dict[str, str] = self.oauth_handler.get_auth()

        request_wrapper_stream_bytes_to_disk(
            filename=filename,
            headers=headers,
            url=f"{self.base_url}/{endpoint}",
            method="GET",
            **kwargs,
        )


def get_resources_df(
    response_dict: Dict[str, Any],
    resources_key: str = "resources",
    keep_fields: Optional[List[str]] = None,
    custom_sort_columns: bool = True,
) -> pd.DataFrame:
    """
    Extracts the 'resources' field from a response dictionary and returns it as a
    DataFrame.

    Parameters
    ----------
    response_dict : Dict[str, Any]
        The response dictionary containing the 'resources' field.
    resources_key : str
        The key in the response dictionary that contains the resources data.
        Default is 'resources'.
    keep_fields : Optional[List[str]]
        A list of fields to keep in the DataFrame. If None, all fields are kept.
    custom_sort_columns : bool
        If True, the DataFrame will be sorted with specific columns first.
        Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the resources data.
    """
    if resources_key not in response_dict:
        raise ValueError(
            f"Field '{resources_key}' not found in the response dictionary."
        )

    resources_df: pd.DataFrame = pd.DataFrame(response_dict[resources_key])
    if keep_fields is not None:
        resources_df = resources_df[keep_fields]

    if "@id" not in resources_df.columns:
        raise ValueError("Column '@id' not found in the resources DataFrame.")

    if custom_sort_columns:
        _c = ["@id", "identifier", "title"]
        if "title" not in resources_df.columns:
            _c.remove("title")
        msg = f"{_c} must be in the DataFrame columns for custom_sort_columns=True"
        assert all(x in resources_df.columns for x in _c), msg
        new_cols = _c + sorted(filter(lambda x: x not in _c, resources_df.columns))
        resources_df = resources_df[new_cols]
    return resources_df


def convert_ticker_based_pandas_df_to_qdf(
    df: pd.DataFrame, categorical: bool = True
) -> pd.DataFrame:
    """
    Convert Parquet DataFrame with ticker entries to a QDF with cid & xcat columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to convert, which should contain a 'ticker' column.
    categorical : bool
        If True, converts the DataFrame to a QuantamentalDataFrame with categorical data.
    """
    df[["cid", "xcat"]] = df["ticker"].str.split("_", n=1, expand=True)
    df = df.drop(columns=["ticker"])
    df = QuantamentalDataFrame(df, categorical=categorical)
    return df


def convert_ticker_based_parquet_file_to_qdf(
    filename: str,
    compression: str = "zstd",
    as_csv: bool = False,
    qdf: bool = False,
    keep_raw_data: bool = False,
) -> None:
    """
    Convert a Parquet file with ticker entries to a QDF or CSV format.
    This function reads a Parquet file, extracts the 'ticker' column, splits it into
    'cid' and 'xcat', and writes the result to a new Parquet file or CSV file.

    Parameters
    ----------
    filename : str
        The path to the Parquet file to convert. The file must exist.
    compression : str
        The compression algorithm to use for the output Parquet file. Default is 'zstd'.
    as_csv : bool
        If True, the output will be saved as a CSV file instead of a Parquet file.
        Default is False.
    qdf : bool
        If True, the output will be saved as a Quantamental DataFrame, in parquet or
        CSV format depending on the `as_csv` parameter. Default is False.
    keep_raw_data : bool
        If True, the original Parquet file will not be deleted after conversion.
        If False, the original file will be removed after conversion. Default is False.
    """

    # Ensure source exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"No such file: {filename}")

    base, ext = os.path.splitext(filename)
    dirpath = os.path.dirname(filename)

    # quick dump of raw data to csv - if csv and not qdf
    if as_csv and not qdf:
        dataset = pa_ds.dataset(filename, format="parquet")
        scanner = dataset.scanner()
        out_csv = base + ".csv"
        with pa_csv.CSVWriter(out_csv, schema=scanner.dataset_schema) as writer:
            for batch in scanner.to_batches():
                writer.write(batch)
        if not keep_raw_data:
            os.remove(filename)
        return

    # return - nothing todo
    if not qdf:
        return

    # setup pa scanner for lazy loading
    dataset = pa_ds.dataset(filename, format="parquet")
    split = pc.split_pattern(pc.field("ticker"), "_", max_splits=1)
    cols = {
        "real_date": pc.field("real_date"),
        "value": pc.field("value"),
        "grading": pc.field("grading"),
        "eop_lag": pc.field("eop_lag"),
        "mop_lag": pc.field("mop_lag"),
        "last_updated": pc.field("last_updated"),
        "cid": pc.list_element(split, 0),
        "xcat": pc.list_element(split, 1),
    }
    scanner = dataset.scanner(columns=cols)

    # set output extension and path
    out_ext = ".csv" if as_csv else ".parquet"
    if keep_raw_data:
        out_path = os.path.join(dirpath, os.path.basename(base) + "_qdf" + out_ext)
    else:
        # overwrite for parquet, or replace for csv
        out_path = filename if not as_csv else base + ".csv"

    if as_csv:
        schema = scanner.projected_schema
        with pa_csv.CSVWriter(out_path, schema=schema) as writer:
            for batch in scanner.to_batches():
                writer.write(batch)
    else:
        pq.write_table(scanner.to_table(), out_path, compression=compression)

    if qdf and as_csv and not keep_raw_data:
        os.remove(filename)


def convert_ticker_based_pyarrow_table_to_qdf(table: pa.Table) -> pa.Table:
    """
    Convert a PyArrow Table with ticker entries to a Quantamental DataFrame (QDF)
    with 'cid' and 'xcat' columns, splitting on '_' lazily via a Scanner.

    Parameters
    ----------
    table : pa.Table
        The PyArrow Table to convert, which should contain a 'ticker' column.

    Returns
    -------
    pa.Table
        A PyArrow Table with all original columns except 'ticker',
        plus new 'cid' and 'xcat' (string) columns.
        The split only happens when you call to_table().
    """
    if "ticker" not in table.schema.names:
        raise KeyError("Column 'ticker' not found in the table.")

    dataset = pa_ds.dataset(table)

    split = pc.split_pattern(pc.field("ticker"), "_", max_splits=1)
    cols = {
        "real_date": pc.field("real_date"),
        "value": pc.field("value"),
        "grading": pc.field("grading"),
        "eop_lag": pc.field("eop_lag"),
        "mop_lag": pc.field("mop_lag"),
        "last_updated": pc.field("last_updated"),
        "cid": pc.list_element(split, 0),
        "xcat": pc.list_element(split, 1),
    }
    scanner = dataset.scanner(columns=cols)

    return scanner.to_table()


def read_parquet_from_bytes_to_pandas_dataframe(response_bytes: bytes) -> pd.DataFrame:
    """
    Read a Parquet file from bytes and return a DataFrame.
    This function is used to read Parquet files downloaded from the JPMaQS Fusion API.

    Parameters
    ----------
    response_bytes : bytes
        The bytes of the Parquet file to read.

    Returns
    --------
    pd.DataFrame
        A DataFrame containing the data from the Parquet file.
    """

    try:
        return pd.read_parquet(io.BytesIO(response_bytes))
    except KeyboardInterrupt:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read Parquet from bytes: {e}") from e


def read_parquet_from_bytes_to_pyarrow_table(
    response_bytes: bytes, **kwargs
) -> pa.Table:
    """
    Read a Parquet file from bytes and return a PyArrow Table.
    This function is used to read Parquet files downloaded from the JPMaQS Fusion API.

    Parameters
    ----------
    response_bytes : bytes
        The bytes of the Parquet file to read.
    **kwargs : dict
        Additional keyword arguments to pass to `pyarrow.parquet.read_table`.

    Returns
    -------
    pa.Table
        A PyArrow Table containing the data from the Parquet file.
    """
    try:
        return pa.parquet.read_table(io.BytesIO(response_bytes), **kwargs)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read Parquet to PyArrow Table: {e}") from e


def coerce_real_date(table: pa.Table) -> pa.Table:
    idx = table.schema.get_field_index("real_date")
    col = table.column(idx)
    t = col.type

    if pa.types.is_date32(t):
        # Already correct type
        return table

    elif pa.types.is_timestamp(t):
        # Fast direct cast
        dates = pc.cast(col, pa.date32())

    elif pa.types.is_string(t):
        # Trim to YYYY-MM-DD to handle datetime strings
        col = pc.utf8_slice_codeunits(col, 0, 10)
        ts = pc.strptime(col, format="%Y-%m-%d", unit="s")
        dates = pc.cast(ts, pa.date32())

    else:
        raise TypeError(f"Unsupported type for real_date: {t}")

    return table.set_column(idx, "real_date", dates)


def filter_parquet_table_as_qdf(
    table: pa.Table,
    tickers: List[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    qdf: bool = False,
) -> pa.Table:
    """
    Filter a PyArrow Table based on tickers and date range. Optionally converts the
    table from a ticker-based format to a Quantamental DataFrame (QDF).

    Parameters
    ----------
    table : pa.Table
        The PyArrow Table to filter.
    tickers : List[str], optional
        A list of tickers to filter by. If None, no ticker filtering is applied.
    start_date : Optional[str], optional
        The start date for filtering in ISO format (YYYY-MM-DD). If None, no start date
        filtering is applied.
    end_date : Optional[str], optional
        The end date for filtering in ISO format (YYYY-MM-DD). If None, no end date
        filtering is applied.
    qdf : bool, optional
        If True, converts the filtered table to a Quantamental DataFrame (QDF) format.
        Default is False.

    Returns
    -------
    pa.Table
        A filtered PyArrow Table. If `qdf` is True, the table is converted to a QDF.
    """

    if not isinstance(table, pa.Table):
        raise TypeError("Input must be a PyArrow Table.")

    table = coerce_real_date(table)
    if not any([tickers, start_date, end_date]) and not qdf:
        return table

    if bool(start_date) and bool(end_date):
        if pd.Timestamp(start_date) > pd.Timestamp(end_date):
            start_date, end_date = end_date, start_date

    ticker_col = "ticker"
    exprs = []
    if tickers:
        if ticker_col not in table.schema.names:
            raise KeyError(f"No column named '{ticker_col}' in table")
        table.column(ticker_col).type
        tickers_array = pa.array(tickers, type=pa.string())
        exprs.append(pc.is_in(pc.field("ticker"), value_set=tickers_array))

    if start_date:
        dt = datetime.date.fromisoformat(start_date)
        scalar = pa.scalar(dt, type=pa.date32())
        exprs.append(pc.greater_equal(pc.field("real_date"), scalar))

    if end_date:
        dt = datetime.date.fromisoformat(end_date)
        scalar = pa.scalar(dt, type=pa.date32())
        exprs.append(pc.less_equal(pc.field("real_date"), scalar))

    expression = functools.reduce(operator.and_, exprs)
    table = table.filter(expression)
    if qdf:
        table = convert_ticker_based_pyarrow_table_to_qdf(table)
    return table


class JPMaQSFusionClient:
    """
    A client for accessing the JPMaQS product on the JPMorgan Fusion API.
    This client is specific to the JPMaQS product and provides methods to fetch the data
    catalog, list datasets, and download distributions.
    It uses :func:`SimpleFusionAPIClient` to handle the API requests.

    Parameters
    ----------
    oauth_handler : FusionOAuth
        An instance of FusionOAuth to handle OAuth authentication.
    base_url : str
        The base URL for the Fusion API. Default is FUSION_ROOT_URL.
    proxies : Optional[Dict[str, str]]
        Optional proxies to use for the HTTP requests. Default is None.
    """

    def __init__(
        self,
        oauth_handler: FusionOAuth,
        base_url: str = FUSION_ROOT_URL,
        proxies: Optional[Dict[str, str]] = None,
    ):
        self._catalog = "common"
        self._product_id = "JPMAQS"
        self._catalog_dataset = "JPMAQS_METADATA_CATALOG"
        self._notifications_dataset = "JPMAQS_METADATA_NOTIFICATIONS"
        self.simple_fusion_client = SimpleFusionAPIClient(
            oauth_handler=oauth_handler, base_url=base_url, proxies=proxies
        )
        self.failure_messages: List[str] = []
        self.metadata_datesets = [
            self._catalog_dataset,
            self._notifications_dataset,
        ]

    def list_datasets(
        self,
        product_id: str = "JPMAQS",
        fields: List[str] = ["@id", "identifier", "title", "description"],
        include_catalog: bool = False,
        include_notifications: bool = False,
        include_full_datasets: bool = True,
        include_explorer_datasets: bool = False,
        include_delta_datasets: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        List datasets available in the JPMaQS product. Returns a DataFrame with the
        specified fields. This excludes the metadata catalog and the Explorer datasets
        by default.

        Parameters
        ----------
        product_id : str
            The product ID to filter datasets by. Default is "JPMAQS".
        fields : List[str]
            List of fields to include in the returned DataFrame.
        include_catalog : bool
            If True, includes the metadata catalog dataset in the results.
        include_notifications : bool
            If True, includes the notifications dataset in the results.
        include_explorer_datasets : bool
            If True, includes the Explorer datasets in the results.
        include_delta_datasets : bool
            If True, includes the Delta datasets in the results.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing information about the available datasets.
        """
        if not (
            include_catalog
            or include_notifications
            or include_full_datasets
            or include_explorer_datasets
            or include_delta_datasets
        ):
            raise ValueError(
                "At least one of `include_catalog`, `include_notifications`, "
                "`include_full_datasets`, `include_explorer_datasets`, or "
                "`include_delta_datasets` must be True."
            )

        r = self.simple_fusion_client.get_product_details(
            product_id=product_id, **kwargs
        )
        resources_df: pd.DataFrame = get_resources_df(r, keep_fields=None)
        resources_df = resources_df.sort_values(by=["isRestricted", "@id"])

        if not include_catalog:
            resources_df = resources_df[
                resources_df["identifier"] != self._catalog_dataset
            ]

        if not include_notifications:
            resources_df = resources_df[
                resources_df["identifier"] != self._notifications_dataset
            ]

        if not include_explorer_datasets:
            sel_bools = resources_df["identifier"].str.startswith("JPMAQS_EXPLORER_")
            if all(sel_bools):
                warnings.warn(
                    "`include_explorer_datasets` is True, but all datasets are Explorer datasets. Setting it to False."
                )
            resources_df = resources_df[~sel_bools]

        if not include_delta_datasets:
            sel_bools = resources_df["identifier"].str.startswith("JPMAQS_DELTA_")
            resources_df = resources_df[~sel_bools]

        if not include_full_datasets:
            delta_datasets = resources_df[
                resources_df["identifier"].str.startswith("JPMAQS_DELTA_")
            ]
            explorer_datasets = resources_df[
                resources_df["identifier"].str.startswith("JPMAQS_EXPLORER_")
            ]
            other_ds_ids = set(delta_datasets["identifier"]) | set(
                explorer_datasets["identifier"]
            )
            full_datasets = resources_df[resources_df["identifier"].isin(other_ds_ids)]
            resources_df = full_datasets

        resources_df = resources_df[fields].reset_index(drop=True)
        resources_df.index = resources_df.index + 1
        return resources_df

    @cache_decorator(CACHE_TTL)
    def get_metadata_catalog(self, **kwargs) -> pd.DataFrame:
        """
        Get the metadata catalog for JPMaQS. This is a special dataset that contains
        metadata (e.g., dataset identifiers, ticker names and descriptions, etc.)
        Returns a DataFrame with the metadata catalog.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the API request.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the metadata catalog.
        """
        r_bytes = self.simple_fusion_client.get_seriesmember_distribution_details(
            catalog=self._catalog,
            dataset=self._catalog_dataset,
            seriesmember="latest",
            distribution="parquet",
            as_bytes=True,
            **kwargs,
        )
        return read_parquet_from_bytes_to_pandas_dataframe(r_bytes)

    @cache_decorator(CACHE_TTL)
    def get_notifications_distribution(
        self, series_member: Optional[str] = None, **kwargs
    ) -> pd.DataFrame:
        """
        Get the notifications distribution for JPMaQS. This dataset contains notifications
        around updating and refresh times of various series in the JPMaQS product.

        Parameters
        ----------
        series_member : Optional[str]
            The series member identifier for which to retrieve the distribution.
        **kwargs : dict
            Additional keyword arguments to pass to the API request.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the notifications distribution details.
        """
        if series_member is None:
            series_member = self.get_latest_seriesmember_identifier(
                dataset=self._notifications_dataset, **kwargs
            )
        r_json = self.simple_fusion_client.get_seriesmember_distribution_details(
            catalog=self._catalog,
            dataset=self._notifications_dataset,
            seriesmember=series_member,
            distribution="json",
            as_json=True,
            **kwargs,
        )

        # handle timestamp issue
        timestamp_str = r_json["metadata"]["datetime"]
        dt_fmt = "%Y-%d-%mT%H%M%S"
        timestamp = pd.to_datetime(timestamp_str, format=dt_fmt, errors="raise")
        df = pd.DataFrame(r_json["data"])
        df["timestamp"] = timestamp

        # explode report fields
        df["cross_section"] = df["cross_section"].str.replace(" ", "").str.split(",")
        df["category"] = df["category"].str.replace(" ", "").str.split(",")
        df = df.explode("cross_section").explode("category").reset_index(drop=True)

        df = df.sort_values(by=["category", "cross_section", "comment"])
        df = df.reset_index(drop=True)

        return df

    def list_tickers(self, **kwargs) -> List[str]:
        """
        List all tickers available in the JPMaQS product. This method retrieves the
        metadata catalog and extracts the tickers from it.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the API request.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the tickers and their metadata.
        """
        metadata_catalog = self.get_metadata_catalog(**kwargs)
        if "Ticker" not in metadata_catalog.columns:
            raise ValueError("Invalid metadata catalog: 'Ticker' column not found.")
        return sorted(metadata_catalog["Ticker"])

    def get_ticker_metadata(self, ticker: str, **kwargs) -> pd.DataFrame:
        """
        Get metadata for a specific ticker in the JPMaQS product. This method retrieves
        the metadata catalog and filters it for the specified ticker.

        Parameters
        ----------
        ticker : str
            The ticker for which to retrieve metadata.
        **kwargs : dict
            Additional keyword arguments to pass to the API request.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the metadata for the specified ticker.
        """
        metadata_catalog = self.get_metadata_catalog(**kwargs)
        ticker_metadata = metadata_catalog[
            metadata_catalog["Ticker"].str.lower() == ticker.lower()
        ]
        if ticker_metadata.empty:
            raise ValueError(f"No metadata found for ticker '{ticker}'.")
        return ticker_metadata.reset_index(drop=True)

    def get_dataset_available_series(self, dataset: str, **kwargs) -> pd.DataFrame:
        """
        Get the available series for a given dataset in the JPMaQS product. Typically,
        each JPMaQS dataset will have one series for all business days (the JPMaQS release
        for that dataset for that day).

        Parameters
        ----------
        dataset : str
            The dataset identifier for which to retrieve series.
        **kwargs : dict
            Additional keyword arguments to pass to the API request.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the available series for the specified dataset.
        """
        result = self.simple_fusion_client.get_dataset_series(
            catalog=self._catalog, dataset=dataset, **kwargs
        )
        cols = ["@id", "identifier", "createdDate", "fromDate", "toDate"]
        metadata_cols = {
            self._catalog_dataset: ["@id", "identifier"],
            self._notifications_dataset: ["@id", "identifier", "createdDate"],
        }
        if dataset in metadata_cols:
            cols = metadata_cols[dataset]

        result = get_resources_df(result, keep_fields=cols)
        return result

    def get_seriesmember_distributions(
        self, dataset: str, seriesmember: str, **kwargs
    ) -> pd.DataFrame:
        """
        Get the available distributions for a given series member in a dataset.

        Parameters
        ----------
        dataset : str
            The dataset identifier for which to retrieve series member distributions.
        seriesmember : str
            The series member identifier for which to retrieve distributions.
        **kwargs : dict
            Additional keyword arguments to pass to the API request.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the available distributions for the specified series
            member.
        """
        result = self.simple_fusion_client.get_seriesmember_distributions(
            catalog=self._catalog, dataset=dataset, seriesmember=seriesmember, **kwargs
        )
        result = get_resources_df(result)
        return result

    def download_series_member_distribution(
        self,
        dataset: str,
        seriesmember: str,
        distribution: str = "parquet",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Download the distribution for a given series member in a dataset.

        Parameters
        ----------
        dataset : str
            The dataset identifier for which to download the series member distribution.
        seriesmember : str
            The series member identifier for which to download the distribution.
        distribution : str
            The distribution format to download. Default is "parquet".
        **kwargs : dict
            Additional keyword arguments to pass to the API request.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the distribution data for the specified series member.
        """

        result = self.simple_fusion_client.get_seriesmember_distribution_details(
            catalog=self._catalog,
            dataset=dataset,
            seriesmember=seriesmember,
            distribution=distribution,
            as_bytes=True,
            **kwargs,
        )

        result = read_parquet_from_bytes_to_pandas_dataframe(result)
        return result

    def download_series_member_distribution_to_disk(
        self,
        save_directory: str,
        dataset: str,
        seriesmember: str,
        distribution: str = "parquet",
        qdf: bool = False,
        as_csv: bool = False,
        keep_raw_data: bool = False,
        **kwargs,
    ) -> None:
        os.makedirs(save_directory, exist_ok=True)
        is_catalog_dataset = dataset in self.metadata_datesets
        filename = os.path.join(
            save_directory, f"{dataset}-{seriesmember}.{distribution}"
        )
        self.simple_fusion_client.get_seriesmember_distribution_details_to_disk(
            filename=filename,
            catalog=self._catalog,
            dataset=dataset,
            seriesmember=seriesmember,
            distribution=distribution,
            **kwargs,
        )
        ftype = "catalog" if is_catalog_dataset else "series member"
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Failed to download {ftype} distribution to {filename}."
            )
        else:
            print(f"Successfully downloaded {ftype} distribution to {filename}.")

        if is_catalog_dataset:
            return

        convert_ticker_based_parquet_file_to_qdf(
            filename=filename,
            as_csv=as_csv,
            qdf=qdf,
            keep_raw_data=keep_raw_data,
        )
        if qdf:
            msg_str = (
                f"Successfully converted {filename} to Quantamental Data Format (QDF)"
            )
            if as_csv:
                msg_str += " and saved as CSV"
            print(msg_str)

    def get_latest_seriesmember_identifier(
        self,
        dataset: str,
        **kwargs,
    ) -> str:
        """
        Get the latest distribution identifier for a given dataset in the JPMaQS product.

        Parameters
        ----------
        dataset : str
            The dataset identifier for which to get the latest distribution.
        **kwargs : dict
            Additional keyword arguments to pass to the API request.

        Returns
        -------
        str
            The identifier of the latest distribution for the specified dataset.
        """
        series_members = self.get_dataset_available_series(dataset=dataset, **kwargs)
        if series_members.empty:
            raise ValueError(f"No series members found for dataset '{dataset}'.")
        latest_series_member = sorted(series_members["identifier"].tolist())[-1]
        return latest_series_member

    def download_latest_distribution(
        self,
        dataset: str,
        distribution: str = "parquet",
        qdf: bool = True,
        categorical: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Download the latest distribution for a given dataset in the JPMaQS product.

        Parameters
        ----------
        dataset : str
            The dataset identifier for which to download the latest distribution.
        distribution : str
            The distribution format to download. Default is "parquet".
        qdf : bool
            If True, converts the DataFrame to a QuantamentalDataFrame.
        categorical : bool
            If True, converts the DataFrame to a QuantamentalDataFrame with categorical
            data.
        **kwargs : dict
            Additional keyword arguments to pass to the API request.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the latest distribution for the specified dataset.
        """
        latest_series_member = self.get_latest_seriesmember_identifier(
            dataset=dataset,
            **kwargs,
        )

        dist_df = self.download_series_member_distribution(
            dataset=dataset,
            seriesmember=latest_series_member,
            distribution=distribution,
            **kwargs,
        )

        if qdf:
            dist_df = convert_ticker_based_pandas_df_to_qdf(
                df=dist_df,
                categorical=categorical,
            )

        return dist_df

    def download_and_filter_series_member_distribution(
        self,
        dataset: str,
        seriesmember: str,
        tickers: List[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        qdf: bool = False,
        distribution: str = "parquet",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Download and filter the distribution for a given series member in a dataset.

        Parameters
        ----------
        dataset : str
            The dataset identifier for which to download the series member distribution.
        seriesmember : str
            The series member identifier for which to download the distribution.
        tickers : List[str]
            A list of tickers to filter the distribution by. If None, no filtering is done.
        start_date : Optional[str]
            The start date to filter the distribution by (in ISO format). If None, no
            filtering is done.
        end_date : Optional[str]
            The end date to filter the distribution by (in ISO format). If None, no
            filtering is done.
        qdf : bool
            If True, converts the DataFrame to a QuantamentalDataFrame. Default is False.
        distribution : str
            The distribution format to download. Default is "parquet".
        **kwargs : dict
            Additional keyword arguments to pass to the API request.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the filtered distribution for the specified series member.
        """
        result = self.simple_fusion_client.get_seriesmember_distribution_details(
            catalog=self._catalog,
            dataset=dataset,
            seriesmember=seriesmember,
            distribution=distribution,
            as_bytes=True,
            **kwargs,
        )

        result = read_parquet_from_bytes_to_pyarrow_table(result)
        result = filter_parquet_table_as_qdf(
            table=result,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            qdf=qdf,
        )
        return result.to_pandas()

    def download_latest_distribution_to_disk(
        self,
        save_directory: str,
        dataset: str,
        distribution: str = "parquet",
        qdf: bool = False,
        as_csv: bool = False,
        keep_raw_data: bool = False,
        **kwargs,
    ) -> None:
        latest_series_member = self.get_latest_seriesmember_identifier(
            dataset=dataset, **kwargs
        )
        self.download_series_member_distribution_to_disk(
            save_directory=save_directory,
            dataset=dataset,
            seriesmember=latest_series_member,
            distribution=distribution,
            qdf=qdf,
            as_csv=as_csv,
            keep_raw_data=keep_raw_data,
            **kwargs,
        )

    def _download_multiple_distributions_to_disk(
        self,
        folder: str = None,
        qdf: bool = False,
        include_catalog: bool = False,
        include_notifications: bool = False,
        include_full_datasets: bool = False,
        include_explorer_datasets: bool = False,
        include_delta_datasets: bool = False,
        as_csv: bool = False,
        keep_raw_data: bool = False,
        datasets_list: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if folder is None:
            folder = Path.cwd()
        folder: Path = Path(folder).expanduser()
        timestamp = pd.Timestamp.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        folder = folder / f"jpmaqs-download-{timestamp}"
        os.makedirs(folder, exist_ok=True)
        catalog_df = self.get_metadata_catalog()
        metadata_catalog_path = os.path.join(folder, "jpmaqs-metadata-catalog")
        if as_csv:
            catalog_df.to_csv(f"{metadata_catalog_path}.csv", index=False)
        else:
            catalog_df.to_parquet(f"{metadata_catalog_path}.parquet", index=False)

        datasets: List[str] = self.list_datasets(
            include_catalog=include_catalog,
            include_notifications=include_notifications,
            include_full_datasets=include_full_datasets,
            include_explorer_datasets=include_explorer_datasets,
            include_delta_datasets=include_delta_datasets,
        )["identifier"].tolist()
        if datasets_list is not None:
            ds_lower = [ds.lower() for ds in datasets]
            avail_ds = [ds for ds in datasets_list if ds.lower() in ds_lower]
            if not avail_ds:
                raise ValueError(
                    f"No datasets found in the provided `datasets_list`. Available datasets: {', '.join(datasets)}"
                )

        self.failure_messages = []
        failures = []
        with cf.ThreadPoolExecutor() as executor:
            futures: Dict[str, cf.Future] = {}
            for ds in datasets:
                futures[ds] = executor.submit(
                    self.download_latest_distribution_to_disk,
                    save_directory=folder,
                    dataset=ds,
                    qdf=qdf,
                    as_csv=as_csv,
                    keep_raw_data=keep_raw_data,
                )
                time.sleep(FUSION_API_DELAY)
            for ds, future in futures.items():
                try:
                    future.result()
                except Exception as e:
                    e_msg = f"Failed to download dataset {ds}: {e}"
                    print(e_msg)
                    self.failure_messages.append(e_msg)
                    failures.append(ds)

        if failures:
            print(
                f"Failed to download the following datasets: {', '.join(failures)}. "
                "Please check the logs for more details."
            )

        return catalog_df

    def download_latest_delta_distribution(
        self,
        folder: str = None,
        qdf: bool = False,
        as_csv: bool = False,
        keep_raw_data: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Download the latest Delta distribution for all datasets in the JPMaQS product.

        Parameters
        ----------
        folder : str
            The folder where the Delta distribution will be saved. If None, a folder with
            the current date will be created in the current directory.
        qdf : bool
            If True, converts the DataFrame to a QuantamentalDataFrame.
        as_csv : bool
            If True, saves the downloaded datasets as CSV files. Default is False, with
            Parquet as the default format.
        keep_raw_data : bool
            If True, keeps the raw data files after conversion. Default is False.
        **kwargs : dict
            Additional keyword arguments to pass to the API request.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the metadata catalog.
        """
        return self._download_multiple_distributions_to_disk(
            folder=folder,
            qdf=qdf,
            include_catalog=False,
            include_full_datasets=False,
            include_explorer_datasets=False,
            include_delta_datasets=True,
            as_csv=as_csv,
            keep_raw_data=keep_raw_data,
        )

    def download_latest_full_snapshot(
        self,
        folder: str = None,
        qdf: bool = False,
        include_catalog: bool = False,
        include_notifications: bool = False,
        include_explorer_datasets: bool = False,
        include_delta_datasets: bool = False,
        as_csv: bool = False,
        keep_raw_data: bool = False,
        datasets_list: List[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Download the latest full snapshot of all datasets in the JPMaQS product.

        Parameters
        ----------
        folder : str
            The folder where the snapshot will be saved. If None, a folder with the current
            date will be created in the current directory.
        qdf : bool
            If True, converts the DataFrame to a QuantamentalDataFrame.
        include_catalog : bool
            If True, includes the metadata catalog dataset in the snapshot. Default is
            False.
        include_explorer_datasets : bool
            If True, includes Explorer datasets in the snapshot. Default is False.
        include_delta_datasets : bool
            If True, includes Delta datasets in the snapshot. Default is False.
        as_csv : bool
            If True, saves the downloaded datasets as CSV files. Default is False, with
            Parquet as the default format.
        keep_raw_data : bool
            If True, keeps the raw data files after conversion. Default is False.
        datasets_list : Optional[List[str]]
            A list of specific datasets to download. If None, all datasets specified using
            the `include_*` parameters will be downloaded.

        **kwargs : dict
            Additional keyword arguments to pass to the API request.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the metadata catalog.
        """
        start_time = time.time()
        result = self._download_multiple_distributions_to_disk(
            folder=folder,
            qdf=qdf,
            include_catalog=include_catalog,
            include_notifications=include_notifications,
            include_full_datasets=True,
            include_explorer_datasets=include_explorer_datasets,
            include_delta_datasets=include_delta_datasets,
            as_csv=as_csv,
            keep_raw_data=keep_raw_data,
            datasets_list=datasets_list,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Downloaded latest full snapshot of JPMaQS datasets in {elapsed_time:.2f} seconds."
        )
        return result

    def download(
        self,
        folder: str = None,
        tickers: Optional[List[str]] = None,
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        metrics: List[str] = ["all"],
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
        qdf: bool = True,
        as_csv: bool = False,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """
        Download data for specified tickers, `cids`, or `xcats` from the JPMaQS product.
        This method downloads the latest full snapshots of the requested tickers' respective
        datasets and filters them based on the provided parameters.

        Parameters
        ----------
        folder : str
            The folder where the downloaded data will be saved. If None, a dataframe
            will be returned without saving to disk.
        tickers : Optional[List[str]]
            A list of tickers to download data for. This list will be concatenated with
            the tickers generated from the combination of `cids` and `xcats`.
        cids : Optional[List[str]]
            A list of `cids` to download data for. This will be used to generate tickers
            in the format "cid_xcat".
        xcats : Optional[List[str]]
            A list of `xcats` to download data for. This will be used to generate tickers
            in the format "cid_xcat".
        metrics : List[str]
            A list of metrics to include in the downloaded data. Default is ["all"], which
            includes all available metrics.
        start_date : str
            The start date for the data to be downloaded, in "YYYY-MM-DD" format.
            Default is "2000-01-01".
        end_date : Optional[str]
            The end date for the data to be downloaded, in "YYYY-MM-DD" format.
            If None, defaults to the current date.
        qdf : bool
            If True, converts the DataFrame to a QuantamentalDataFrame. Default is True.
        as_csv : bool
            If True, saves the downloaded datasets as CSV files. Default is False, with
            Parquet as the default format.
        **kwargs : dict
            Additional keyword arguments to pass to the API request.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the downloaded data for the specified tickers, `cids`,
            or `xcats`. If `folder` is specified, the data will also be saved to disk.
        """
        save_to_folder = folder is not None
        if folder is None:
            folder = Path.cwd()
        folder: Path = Path(folder).expanduser()

        def vartolist(x: Optional[List[str]]) -> List[str]:
            return [x] if isinstance(x, str) else x

        tickers = vartolist(tickers)
        cids = vartolist(cids)
        xcats = vartolist(xcats)
        metrics = vartolist(metrics)

        if tickers is None:
            tickers = []
        if bool(cids) ^ bool(xcats):
            raise ValueError(
                "Both `cids` and `xcats` must be provided together or neither."
            )
        if cids is not None and xcats is not None:
            tickers += [f"{cid}_{xcat}" for cid in cids for xcat in xcats]

        if not tickers:
            raise ValueError(
                "At least one of `tickers`, `cids`, or `xcats` must be provided."
            )
        tickers = sorted(set(tickers))

        catalog_df = self.get_metadata_catalog()

        all_tickers_lower = catalog_df["Ticker"].str.lower().tolist()
        non_existing = sorted(_ for _ in tickers if _.lower() not in all_tickers_lower)
        tickers = sorted(_ for _ in tickers if _.lower() not in non_existing)
        if non_existing:
            wstr = f"There are {len(non_existing)} tickers that do not exist in the metadata catalog. "
            wstr += "Please check the input tickers against the metadata catalog."
            warnings.warn(wstr)

        tickers_info = catalog_df[
            catalog_df["Ticker"].str.lower().isin([_.lower() for _ in tickers])
        ]
        datasets = tickers_info.drop_duplicates(subset=["Theme"], keep="first")
        if datasets.empty:
            raise ValueError(
                "No datasets found for the specified tickers. Please check the tickers "
                "against the metadata catalog."
            )
        datasets = sorted(
            set(
                "JPMAQS_"
                + datasets["Theme"]
                .str.replace(" ", "_")
                .str.upper()
                .reset_index(drop=True)
            )
        )

        if save_to_folder:
            return self.download_latest_full_snapshot(
                folder=folder,
                qdf=qdf,
                include_catalog=True,
                include_notifications=True,
                as_csv=as_csv,
                keep_raw_data=kwargs.pop("keep_raw_data", False),
                datasets_list=datasets,
                **kwargs,
            )

        if end_date is None:
            end_date = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        if pd.Timestamp(start_date) > pd.Timestamp(end_date):
            start_date, end_date = end_date, start_date

        def _download_df(
            dataset: str,
            tickers: List[str],
            start_date: str,
            end_date: str,
            **kwargs,
        ) -> pd.DataFrame:
            series_member = self.get_latest_seriesmember_identifier(
                dataset=dataset, **kwargs
            )
            df = self.download_and_filter_series_member_distribution(
                dataset=dataset,
                seriesmember=series_member,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                qdf=qdf,
                **kwargs,
            )
            df["dataset"] = dataset
            return df

        _commonargs = dict(tickers=tickers, start_date=start_date, end_date=end_date)

        print(f"downloading {len(datasets)} datasets: {', '.join(datasets)}")
        results: List[pd.DataFrame] = []
        with cf.ThreadPoolExecutor() as executor:
            futures: Dict[str, cf.Future] = {}
            for dataset in datasets:
                futures[dataset] = executor.submit(
                    _download_df, dataset=dataset, **_commonargs, **kwargs
                )
                time.sleep(FUSION_API_DELAY)

            for dataset, future in futures.items():
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Failed to download data for dataset {dataset}: {e}")

        if not len(results) or all(df.empty for df in results):
            results = pd.DataFrame()
        else:
            results = pd.concat(results, ignore_index=True).reset_index(drop=True)
        if results.empty:
            raise ValueError(
                "No data found for the specified tickers, cids, or xcats within the date range."
            )
        return QuantamentalDataFrame(results)


if __name__ == "__main__":
    st = time.time()
    oauth_handler = FusionOAuth.from_credentials_json(
        "data/fusion_client_credentials.json"
    )
    jpmaqs_client = JPMaQSFusionClient(oauth_handler=oauth_handler)

    st = time.time()
    df = jpmaqs_client.get_notifications_distribution()
    print(df.head())
    print(f"Time taken for notifications download: {time.time() - st:.2f} seconds")

    st = time.time()

    df = jpmaqs_client.download(
        # folder="./data",
        cids=["USD", "GBP", "EUR", "JPY", "CHF", "AUD", "CAD"],
        xcats=["FXXR_NSA", "EQXR_NSA", "EQCRY_NSA"],
        tickers=["USD_EQXR_NSA", "GBP_EQXR_NSA"],
        start_date="2025-07-17",
    )
    print(df.head())
    print(f"Time taken for download: {time.time() - st:.2f} seconds")
    df = None
    ds = jpmaqs_client.list_datasets()

    st = time.time()
    jpmaqs_client.download_latest_full_snapshot(
        folder="./data",
        keep_raw_data=False,
        # qdf=True,
        # as_csv=True,
    )
    print(f"Time taken for full snapshots download: {time.time() - st:.2f} seconds")

    st = time.time()

    jpmaqs_client.download_latest_delta_distribution(
        folder="./data",
        # qdf=True,
        # as_csv=True,
        keep_raw_data=False,
    )
    print(f"Time taken for latest delta download: {time.time() - st:.2f} seconds")
