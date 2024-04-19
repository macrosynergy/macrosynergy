import datetime
import pandas as pd
import warnings
import requests
import numpy as np
from typing import List
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import get_cid, get_xcat
from io import StringIO

MAX_RETRIES = 3
CSV_URL = "https://macrosynergy-trading-costs.s3.eu-west-2.amazonaws.com/transaction-costs.csv"


def _request_wrapper(url: str, verbose: bool = True, **kwargs) -> str:
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
            return r.text
        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"Error downloading {url}: {e}")
            continue
    raise requests.exceptions.RequestException(f"Failed to download {url}")


AVAILABLE_CTYPES = ["FX", "IRS", "CDS"]
AVAIALBLE_COSTS = ["BIDOFFER", "ROLLCOST", "SIZE"]
AVAILABLE_STATS = ["MEDIAN", "90PCTL"]
AVAILABLE_CATS = [
    f"{c}{t}_{s}"
    for c in AVAILABLE_CTYPES
    for t in AVAIALBLE_COSTS
    for s in AVAILABLE_STATS
]


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
        StringIO(_request_wrapper(csv_url, **kwargs)),
        parse_dates=["real_date"],
    )
    dfd["cid"], dfd["xcat"] = get_cid(dfd["ticker"]), get_xcat(dfd["ticker"])
    dfd = dfd[QuantamentalDataFrame.IndexCols + ["value"]]
    if not isinstance(dfd, pd.DataFrame) or dfd.empty:
        raise ValueError("Failed to download trading costs data")

    if not isinstance(dfd, QuantamentalDataFrame):
        ers = "Downloaded dataframe is not a QuantamentalDataFrame."
        warnings.warn(ers, UserWarning)
        if verbose:
            print(ers)

    return dfd


if __name__ == "__main__":
    print(download_transaction_costs(verbose=True).head())
