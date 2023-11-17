import argparse
import json
import logging
import os
from typing import Any, Dict, List

import time
from joblib import Parallel, delayed

from multiprocessing import Pool

from macrosynergy.download.jpmaqs import JPMaQSDownload
from macrosynergy.download.local import DownloadTimeseries, LocalCache


logger = logging.getLogger(__name__)


def load_credentials(path: str) -> Dict[str, Dict[str, str]]:
    # load the credentials
    with open(path, "r") as f:
        credentials: dict = json.load(f)

    creds: set = set()
    for kx, vx in credentials.items():
        if not isinstance(vx, dict):
            raise ValueError(f"Value for key {kx} is not a dict")
        creds.add(vx["client_id"] + vx["client_secret"])

    if len(creds) != len(credentials):
        raise ValueError("Duplicate credentials detected")

    return credentials


def _create_store(
    store_path: str,
    client_id: str,
    client_secret: str,
    expressions: List[str],
    fmt: str = "json",
    test_mode: bool = False,
) -> None:
    try:
        DownloadTimeseries(
            store_path=store_path,
            client_id=client_id,
            client_secret=client_secret,
            store_format=fmt,
            test_mode=test_mode,
        ).download_data(
            show_progress=True,
            expressions=expressions,
            # delay_param=1.0,
        )
    except Exception as exc:
        logger.error(f"Error downloading data for {client_id}: {exc}")
        return False

    return True


def _mc_jobs(
    jobs_list: List[Dict[str, Any]],
    max_retry: int = 3,
    njobs: int = 4,
):
    # njobs may never be > 6
    if njobs > 6:
        njobs = 6

    if max_retry < 1:
        raise ValueError("Exhausted all retries")

    results: List[bool] = Parallel(n_jobs=njobs)(
        delayed(_create_store)(**job) for job in jobs_list
    )
    with Pool(njobs) as p:
        for job in jobs_list:
            p.apply_async(_create_store, kwds=job)
        p.close()
        p.join()

    # failed_jobs: List[Dict[str, Any]] = [
    #     job for job, res in zip(jobs_list, results) if not res
    # ]

    # if len(failed_jobs) > 0:
    #     logger.info(f"Retrying {len(failed_jobs)} jobs")
    #     _mc_jobs(failed_jobs, max_retry=max_retry - 1)


def multicred_download(
    store_path: str,
    credentials: Dict[str, Dict[str, str]],
    fmt: str = "json",
    test_mode: bool = False,
    njobs: int = 4,
):
    store_path = os.path.expanduser(store_path)
    os.makedirs(store_path, exist_ok=True)
    os.makedirs(os.path.join(store_path, fmt, "/"), exist_ok=True)

    if not isinstance(credentials, dict) and not len(credentials) > 0:
        raise ValueError("Credentials must be a non-empty dict")

    fcred: Dict[str, str] = list(credentials.values())[0]
    all_tickers_list: List[str] = JPMaQSDownload(
        client_id=fcred["client_id"],
        client_secret=fcred["client_secret"],
    ).get_catalogue()

    if test_mode:
        if isinstance(test_mode, bool):
            test_mode = 1000
        all_tickers_list = all_tickers_list[:test_mode]

    all_expressions_list: List[str] = JPMaQSDownload.construct_expressions(
        tickers=all_tickers_list,
        metrics=[
            "value",
            "grading",
            "eop_lag",
            "mop_lag",
        ],
    )
    # divide the expressions into batches for each credential
    ncreds: int = len(credentials)
    nexpr: int = len(all_expressions_list)
    nexpr_per_cred: int = nexpr // ncreds
    cred_expr: Dict[str, List[str]] = {}
    job_batches: List[Dict[str, Any]] = []
    for kx, vx in credentials.items():
        cred_expr[kx] = all_expressions_list[:nexpr_per_cred]
        if cred_expr[kx] == []:
            continue
        all_expressions_list = all_expressions_list[nexpr_per_cred:]
        job_batches.append(
            {
                "store_path": store_path,
                "client_id": vx["client_id"],
                "client_secret": vx["client_secret"],
                "fmt": fmt,
                "expressions": cred_expr[kx],
            }
        )

    # add the remaining expressions to the last credential
    job_batches[-1]["expressions"] += all_expressions_list

    # run the jobs
    _mc_jobs(jobs_list=job_batches, max_retry=3, njobs=njobs)

    lc: LocalCache = LocalCache(local_path=store_path, fmt=fmt)
    found_tickers: List[str] = lc.get_catalogue()
    missing_tickers: List[str] = list(set(all_tickers_list) - set(found_tickers))
    if len(missing_tickers) > 0:
        logger.warning(f"Missing tickers: {missing_tickers}")
    else:
        logger.info("All tickers found")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Download data from JPMaQS and store locally"
    )

    # get the path to the credentials file
    parser.add_argument(
        "--credentials",
        "-c",
        type=str,
        help="Path to the credentials file",
        # required=True,
        default="C:/Users/PalashTyagi/Code/ms_copy/macrosynergy/macrosynergy/download/dq_aws_multicred.json",
    )

    # get the path to the store
    parser.add_argument(
        "--store",
        "-s",
        type=str,
        help="Path to the store",
        # required=True,
        default="~/Documents/data/JPMaQSData/",
    )

    parser.add_argument(
        "--test_mode",
        "-t",
        help="Test mode",
        action="store_true",
        default=False,
    )

    # add njobs argument
    parser.add_argument(
        "--njobs",
        "-n",
        type=int,
        help="Number of jobs to run in parallel",
        default=4,
    )

    args = parser.parse_args()

    credentials: Dict[str, Dict[str, str]] = load_credentials(args.credentials)
    store_path: str = args.store

    if not isinstance(args.njobs, int):
        raise ValueError("njobs must be a positive integer")

    mstart: float = time.time()
    multicred_download(
        store_path=store_path,
        credentials=credentials,
        fmt="json",
        test_mode=1000,
        njobs=args.njobs,
    )

    mend: float = time.time()

    logger.info(f"Total time for multicred download: {mend - mstart:.2f} seconds")
    print(f"Total time for multicred download: {mend - mstart:.2f} seconds")
