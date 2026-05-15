import datetime
import pandas as pd
import warnings
import requests
from typing import List, Literal
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import get_cid, get_xcat
from io import StringIO

MAX_RETRIES = 3
TRANSACTION_COSTS_FILE_URL = None


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


AVAILABLE_CTYPES: List[str] = ["FX", "IRS", "CDS"]
AVAIALBLE_COSTS: List[str] = ["BIDOFFER", "ROLLCOST", "SIZE"]
AVAILABLE_STATS: List[str] = ["MEDIAN", "90PCTL"]
AVAILABLE_CATS: List[str] = [
    f"{c}{t}_{s}"
    for c in AVAILABLE_CTYPES
    for t in AVAIALBLE_COSTS
    for s in AVAILABLE_STATS
]


def download_transaction_costs(
    file_url: str = TRANSACTION_COSTS_FILE_URL,
    verbose: bool = False,
    categorical: bool = False,
    file_format: Literal["auto", "csv", "parquet"] = "auto",
    **kwargs,
) -> QuantamentalDataFrame:
    """
    Download trading costs data from the S3 bucket.

    Parameters
    ----------
    file_url : str
        URL of the file to download. If the file format is "auto", the function will try
        to infer the file format from the URL extension.
    verbose : bool
        Print progress information.
    file_format : Literal["auto", "csv", "parquet"]
        Format of the file to download. If "auto", the function will try to infer the format
        from the URL extension. Supported formats are "csv" and "parquet".
    **kwargs : args
        Additional keyword arguments to pass to `requests.get`. This can be used to pass
        additional headers, proxy settings, cert verification, etc.
    """
    if file_url is None or not isinstance(file_url, str):
        raise ValueError(f"Invalid file URL provided: {file_url}")
    if verbose:
        print(f"Timestamp (UTC): {datetime.datetime.now(datetime.timezone.utc)}")
        print(f"Downloading trading costs data from {file_url}")
    if file_format == "csv" or (file_format == "auto" and file_url.endswith(".csv")):
        dfd: pd.DataFrame = pd.read_csv(
            StringIO(_request_wrapper(file_url, **kwargs)),
            parse_dates=["real_date"],
        )
    elif file_format == "parquet" or (
        file_format == "auto" and file_url.endswith(".parquet")
    ):
        dfd: pd.DataFrame = pd.read_parquet(file_url)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    dfd["cid"], dfd["xcat"] = get_cid(dfd["ticker"]), get_xcat(dfd["ticker"])
    dfd = dfd[QuantamentalDataFrame.IndexCols + ["value"]]
    if not isinstance(dfd, pd.DataFrame) or dfd.empty:
        raise ValueError("Failed to download trading costs data")

    if not isinstance(dfd, QuantamentalDataFrame):
        if verbose:
            warnings.warn(
                "Downloaded data could not be converted to QuantamentalDataFrame",
                UserWarning,
            )
    else:
        if verbose:
            print("Downloaded data and transformed data successfully")
        dfd = QuantamentalDataFrame(dfd, categorical=categorical)

    return dfd


if __name__ == "__main__":
    print(download_transaction_costs(verbose=True).head())
