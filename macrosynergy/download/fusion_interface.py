import requests
import json
import datetime
import time
import logging
import os
import io
import warnings
import functools
from typing import Dict, Optional, TypeVar, Any, List, Union, Callable
import pandas as pd

from macrosynergy import __version__ as ms_version_info

from macrosynergy.management.types import QuantamentalDataFrame

FUSION_AUTH_URL: str = "https://authe.jpmorgan.com/as/token.oauth2"
FUSION_ROOT_URL: str = "https://fusion.jpmorgan.com/api/v1"
FUSION_RESOURCE_ID: str = "JPMC:URI:RS-93742-Fusion-PROD"
FUSION_API_DELAY = 1.0  # seconds
CACHE_TTL = 60  # seconds
LAST_API_CALL: Optional[datetime.datetime] = None

logger = logging.getLogger(__name__)


class FusionOAuth(object):
    @staticmethod
    def from_credentials_json(credentials_json: str):
        with open(credentials_json, "r") as f:
            credentials = json.load(f)
        return FusionOAuth.from_credentials(credentials)

    @staticmethod
    def from_credentials(credentials: dict):
        return FusionOAuth(**credentials)

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        resource: str = FUSION_RESOURCE_ID,
        application_name: str = "fusion",
        root_url: str = FUSION_ROOT_URL,
        auth_url: str = FUSION_AUTH_URL,
        proxies: dict = None,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.resource = resource
        self.application_name = application_name
        self.root_url = root_url
        self.auth_url = auth_url
        self.proxies = proxies

        self.token_data = {
            "grant_type": "client_credentials",
            "aud": resource,
            "client_id": client_id,
            "client_secret": client_secret,
        }

        self._stored_token = None

    def _retrieve_token(self):
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
            self._retrieve_token()
        return self._stored_token["access_token"]

    def get_auth(self) -> dict:
        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "User-Agent": f"MacrosynergyPackage/{ms_version_info}",
        }
        return headers


CachedType = TypeVar("CachedType", bound=Callable[..., Any])


def cached(
    ttl: int = 60, *, maxsize: Optional[int] = None
) -> Callable[[CachedType], CachedType]:
    """
    Decorator to cache the result of a function for up to `ttl` seconds total.
    Once any call happens at least `ttl` seconds after the last clear, the ENTIRE
    cache is flushed before proceeding.

    :param ttl: time-to-live for the whole cache, in seconds
    :param maxsize: maximum number of entries to hold in the LRU cache
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


def wait_for_api_call() -> bool:
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
) -> Optional[Union[Dict[str, Any], str, bytes]]:
    assert method in ["GET", "POST", "PUT", "DELETE"], f"Invalid method: {method}"

    if sum(map(bool, [as_bytes, as_text, as_json])):
        as_json = True
    elif sum(map(bool, [as_bytes, as_text, as_json])) > 1:
        raise ValueError("Only one of `as_json`, `as_bytes`, or `as_text` can be True.")
    raw_response: Optional[requests.Response] = None
    try:
        wait_for_api_call()
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

        if response.status_code == 204:
            return None
        if not response.content:
            return None

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
        self.proxies: Optional[Dict[str, str]] = proxies

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        json_payload: Optional[Dict[str, Any]] = None,
        as_json: Optional[bool] = None,
        as_bytes: Optional[bool] = None,
        as_text: Optional[bool] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        url: str = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers: Dict[str, str] = self.oauth_handler.get_auth()
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

    @cached(CACHE_TTL)
    def get_common_catalog(self, **kwargs) -> Optional[Dict[str, Any]]:
        # /v1/catalogs/common
        return self._request(method="GET", endpoint="catalogs/common", **kwargs)

    @cached(CACHE_TTL)
    def get_products(self, **kwargs) -> Optional[Dict[str, Any]]:
        # /v1/catalogs/common/products
        return self._request(
            method="GET", endpoint="catalogs/common/products", **kwargs
        )

    @cached(CACHE_TTL)
    def get_product_details(
        self, product_id: str = "JPMAQS", **kwargs
    ) -> Optional[Dict[str, Any]]:
        # /v1/catalogs/common/products/{product_id}
        return self._request(
            method="GET", endpoint=f"catalogs/common/products/{product_id}", **kwargs
        )

    @cached(CACHE_TTL)
    def get_dataset(
        self, catalog: str, dataset: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        # /v1/catalogs/{catalog}/datasets/{dataset}
        endpoint: str = f"catalogs/{catalog}/datasets/{dataset}"
        return self._request(method="GET", endpoint=endpoint, **kwargs)

    @cached(CACHE_TTL)
    def get_dataset_series(
        self, catalog: str, dataset: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        # /v1/catalogs/{catalog}/datasets/{dataset}/datasetseries
        endpoint: str = f"catalogs/{catalog}/datasets/{dataset}/datasetseries"
        return self._request(method="GET", endpoint=endpoint, **kwargs)

    @cached(CACHE_TTL)
    def get_dataset_seriesmember(
        self, catalog: str, dataset: str, seriesmember: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        # /v1/catalogs/{catalog}/datasets/{dataset}/datasetseries/{seriesmember}
        endpoint: str = (
            f"catalogs/{catalog}/datasets/{dataset}/datasetseries/{seriesmember}"
        )
        return self._request(method="GET", endpoint=endpoint, **kwargs)

    @cached(CACHE_TTL)
    def get_seriesmember_distributions(
        self, catalog: str, dataset: str, seriesmember: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        # /v1/catalogs/{catalog}/datasets/{dataset}/datasetseries/{seriesmember}/distributions
        endpoint: str = (
            f"catalogs/{catalog}/datasets/{dataset}/datasetseries/{seriesmember}/distributions"
        )
        return self._request(method="GET", endpoint=endpoint, **kwargs)

    def get_seriesmember_distribution_details(
        self, catalog: str, dataset: str, seriesmember: str, distribution: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        # /v1/catalogs/{catalog}/datasets/{dataset}/datasetseries/{seriesmember}/distributions/{distribution}
        endpoint: str = (
            f"catalogs/{catalog}/datasets/{dataset}/datasetseries/{seriesmember}/distributions/{distribution}"
        )
        return self._request(method="GET", endpoint=endpoint, **kwargs)


def get_resources_df(
    response_dict: Dict[str, Any],
    resources_key: str = "resources",
    keep_fields: Optional[List[str]] = None,
    custom_sort_columns: bool = True,
) -> pd.DataFrame:
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


def convert_ticker_based_parquet_to_qdf(
    df: pd.DataFrame, categorical: bool = True
) -> pd.DataFrame:
    """
    Convert Parquet DataFrame with ticker entries to a QDF with cid & xcat columns.
    """
    df[["cid", "xcat"]] = df["ticker"].str.split("_", n=1, expand=True)
    df = df.drop(columns=["ticker"])
    df = QuantamentalDataFrame(df, categorical=categorical)
    return df


def read_parquet_from_bytes(r_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_parquet(io.BytesIO(r_bytes))
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            raise e
        raise ValueError(f"Failed to read parquet from bytes: {e}") from e


class JPMaQSFusionClient:
    def __init__(
        self,
        oauth_handler: FusionOAuth,
        base_url: str = FUSION_ROOT_URL,
        proxies: Optional[Dict[str, str]] = None,
    ):
        self._catalog = "common"
        self._product_id = "JPMAQS"
        self._catalog_dataset = "JPMAQS_METADATA_CATALOG"
        self.simple_fusion_client = SimpleFusionAPIClient(
            oauth_handler=oauth_handler, base_url=base_url, proxies=proxies
        )

    def list_datasets(
        self,
        product_id: str = "JPMAQS",
        fields=["@id", "identifier", "title", "description"],
        include_catalog: bool = False,
        include_explorer_datasets: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        r = self.simple_fusion_client.get_product_details(
            product_id=product_id, **kwargs
        )
        resources_df: pd.DataFrame = get_resources_df(r, keep_fields=None)
        resources_df = resources_df[
            resources_df["identifier"] != "JPMAQS_METADATA_CATALOG"
        ].sort_values(by=["isRestricted", "@id"])

        if not include_catalog:
            resources_df = resources_df[
                resources_df["identifier"] != self._catalog_dataset
            ]

        if not include_explorer_datasets:
            sel_bools = resources_df["identifier"].str.startswith("JPMAQS_EXPLORER_")
            if all(sel_bools):
                warnings.warn(
                    "`include_explorer_datasets` is True, but all datasets are Explorer datasets. Setting it to False."
                )
            resources_df = resources_df[~sel_bools]

        resources_df = resources_df[fields].reset_index(drop=True)
        resources_df.index = resources_df.index + 1
        return resources_df

    @cached(CACHE_TTL)
    def get_metadata_catalog(self, **kwargs) -> pd.DataFrame:
        r_bytes = self.simple_fusion_client.get_seriesmember_distribution_details(
            catalog=self._catalog,
            dataset=self._catalog_dataset,
            seriesmember="latest",
            distribution="parquet",
            as_bytes=True,
            **kwargs,
        )
        return read_parquet_from_bytes(r_bytes)

    def get_dataset_available_series(self, dataset: str, **kwargs) -> pd.DataFrame:
        result = self.simple_fusion_client.get_dataset_series(
            catalog=self._catalog, dataset=dataset, **kwargs
        )
        cols = ["@id", "identifier", "createdDate", "fromDate", "toDate"]
        if dataset == self._catalog_dataset:
            cols = cols[:2]

        result = get_resources_df(result, keep_fields=cols)
        return result

    def get_seriesmember_distributions(
        self, dataset: str, seriesmember: str, **kwargs
    ) -> pd.DataFrame:
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
        result = self.simple_fusion_client.get_seriesmember_distribution_details(
            catalog=self._catalog,
            dataset=dataset,
            seriesmember=seriesmember,
            distribution=distribution,
            as_bytes=True,
            **kwargs,
        )

        result = read_parquet_from_bytes(result)
        return result

    def download_latest_distribution(
        self,
        dataset: str,
        distribution: str = "parquet",
        qdf: bool = True,
        categorical: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        series_members = self.get_dataset_available_series(dataset=dataset, **kwargs)

        latest_series_member = sorted(series_members["identifier"].tolist())[-1]
        dist_df = self.download_series_member_distribution(
            dataset=dataset,
            seriesmember=latest_series_member,
            distribution=distribution,
            **kwargs,
        )

        if qdf:
            dist_df = convert_ticker_based_parquet_to_qdf(
                df=dist_df,
                categorical=categorical,
            )

        return dist_df

    def download_latest_full_snapshot(
        self,
        folder: str = None,
        qdf: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        if folder is None:
            _date = datetime.datetime.now().strftime("%Y-%m-%d")
            folder = "./jpmaqs-full-snapshot-" + _date
        os.makedirs(folder, exist_ok=True)

        catalog_df = jpmaqs_client.get_metadata_catalog()
        catalog_df.to_csv(
            os.path.join(folder, "jpmaqs-metadata-catalog.csv"),
            index=False,
        )

        datasets = jpmaqs_client.list_datasets()["identifier"].tolist()
        for ds in datasets:
            dist_df = jpmaqs_client.download_latest_distribution(
                ds, qdf=qdf, categorical=False, **kwargs
            )
            dist_df.to_csv(
                os.path.join(folder, f"{ds}.csv"),
                index=False,
            )
            print(f"Downloaded {ds} to {folder}/{ds}.csv")


if __name__ == "__main__":
    oauth_handler = FusionOAuth.from_credentials_json(
        "data/fusion_client_credentials.json"
    )
    jpmaqs_client = JPMaQSFusionClient(oauth_handler=oauth_handler)

    jpmaqs_client.download_latest_full_snapshot()

    # catalog_df = jpmaqs_client.get_metadata_catalog()
    # print("JPMaQS Catalog:")
    # print(catalog_df.head(5))

    # min_dates, max_dates = [], []

    # tickers_count = []

    # for ds in jpmaqs_client.list_datasets()["identifier"].tolist():
    #     dist_df = jpmaqs_client.download_latest_distribution(ds)
    #     print(f"Dataset: {ds}")
    #     if "ticker" in dist_df.columns:
    #         tickers_count.append(len(dist_df["ticker"].drop_duplicates()))
    #     else:
    #         tickers_count.append(len(dist_df[["cid", "xcat"]].drop_duplicates()))

    #     print(f"Unique tickers: {tickers_count[-1]}")
    #     _min, _max = dist_df["real_date"].min(), dist_df["real_date"].max()
    #     min_dates.append(_min)
    #     max_dates.append(_max)
    #     print(f"Min, Max dates: {_min}, {_max}")
    #     print("\n---")

    # print("Unique tickers count:", sum(tickers_count))
    # print("Min. date: ", min(min_dates), "\t\tMax. date: ", max(max_dates))
