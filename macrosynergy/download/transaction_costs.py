import datetime
import pandas as pd
import warnings
import requests
import numpy as np
from macrosynergy.management.types import QuantamentalDataFrame
from io import StringIO

MAX_RETRIES = 3
CSV_URL = "https://macrosynergy-trading-costs.s3.eu-west-2.amazonaws.com/transaction-costs.csv"


def _request_wrapper(url: str, verbose: bool = True, **kwargs) -> requests.Response:
    """
    Wrapper around `requests.get` to handle errors and retries.
    Any additional keyword arguments are passed to `requests.get`. Purpose is to allow
    users to pass additional headers, proxy settings, cert verification, etc.
    """
    if verbose:
        print(f"Requesting data from {url}")

    for _ in range(MAX_RETRIES):
        try:
            r = requests.get(url, **kwargs)
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"Error downloading {url}: {e}")
            continue
    raise requests.exceptions.RequestException(f"Failed to download {url}")


def download_transaction_costs(
    csv_url: str = CSV_URL,
    verbose: bool = False,
    **kwargs,
) -> QuantamentalDataFrame:
    """
    Download trading costs data from the S3 bucket.

    :param <str> csv_url: URL of the CSV file to download.
    :param <bool> verbose: Print progress information.
    :param **kwargs: Additional keyword arguments to pass to `requests.get`. This can be
        used to pass additional headers, proxy settings, cert verification, etc.
    """
    if verbose:
        print(f"Timestamp (UTC): {datetime.datetime.now(datetime.timezone.utc)}")
        print(f"Downloading trading costs data from {csv_url}")
    dfd: pd.DataFrame = pd.read_csv(
        StringIO(_request_wrapper(csv_url, **kwargs).text),
        parse_dates=["real_date"],
        dtype={"ticker": str, "real_date": str, "value": np.float64},
        usecols=["ticker", "real_date", "value"],
        header=0,
        engine="c",
    )
    if not isinstance(dfd, pd.DataFrame):
        raise ValueError("Failed to download trading costs data")

    if not isinstance(dfd, QuantamentalDataFrame):
        warnings.warn("Downloaded dataframe is not a QuantamentalDataFrame.")

    return dfd


download_transaction_costs()
