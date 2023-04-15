from typing import Dict, List, Optional, Union
from macrosynergy.download import JPMaQSDownload
from macrosynergy.download.dataquery import OAUTH_BASE_URL
from macrosynergy.management.utils import Config
import pandas as pd
from timeit import default_timer as timer
from tqdm import tqdm
from joblib import Parallel, delayed

###############
# Class MultiCredentials


class MultiCredentialDownload:
    def __init__(
        self, config: Config, base_url: str = OAUTH_BASE_URL, max_workers: int = 4
    ):
        self.config = config
        self._base_url = base_url
        self._max_workers = max_workers

    def _download_chunk(
        self,
        credentials: Config,
        download_args: dict,
    ):
        # force the download skip the dataframe conversion
        download_args["as_dataframe"] = False

        try:
            with JPMaQSDownload(**credentials.oauth(mask=False)) as jpm:
                return jpm.download(**download_args)
        except Exception as e:
            print(e)
            return -1

    def download(
        self,
        tickers: Optional[List[str]] = None,
        cids: Optional[List[str]] = None,
        xcats: Optional[List[str]] = None,
        metrics: List[str] = ["all"],
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
        report_time_taken: bool = True,
        report_egress: bool = True,
    ) -> Dict[str, dict]:
        if metrics==["all"]:
            metrics = ["value", "grading", "eop_lag", "mop_lag"]

        all_expressions = JPMaQSDownload.construct_expressions(
            tickers=tickers, cids=cids, xcats=xcats, metrics=metrics
        )

        # split the expressions into n chunks, where n is the number of credentials
        mcred: List[Config] = self.config.multi_credentials()

        chunk_size = len(all_expressions) // len(mcred)
        chunks = [
            all_expressions[i : i + chunk_size]
            for i in range(0, len(all_expressions), chunk_size)
        ]
        results = []
        start: float = timer()
        results = Parallel(n_jobs=self._max_workers)(
            delayed(self._download_chunk)(
                credentials=credx,
                download_args={
                    "expressions": chunk,
                    "start_date": start_date,
                    "end_date": end_date,
                    "report_time_taken": report_time_taken,
                    "report_egress": report_egress,
                },
            )
            for credx, chunk in tqdm(zip(mcred, chunks), total=len(chunks))
        )
        end: float = timer()
        if report_time_taken:
            print(f"MultiCred time taken: {end - start} seconds")

        failed = [i for i, x in enumerate(results) if x == -1]
        if len(failed) > 0:
            # form the failed expressions
            print(f"Failed to download {len(failed)} chunks")
            print("Retrying...")
            failed_chunks = [chunks[i] for i in failed]
            f_mcred = mcred[: len(failed_chunks)]
            start: float = timer()
            f_results = Parallel(n_jobs=self._max_workers)(
                delayed(self._download_chunk)(
                    credentials=credx,
                    download_args={
                        "expressions": chunk,
                        "start_date": start_date,
                        "end_date": end_date,
                        "report_time_taken": report_time_taken,
                        "report_egress": report_egress,
                    },
                )
                for credx, chunk in tqdm(
                    zip(f_mcred, failed_chunks), total=len(failed_chunks)
                )
            )

            end: float = timer()
            if report_time_taken:
                print(f"MultiCred time taken: {end - start} seconds")
            results = [x if x != -1 else y for x, y in zip(results, f_results)]

            # print the failed expressions
            print(f"Failed to download {len(failed)} chunks. No longer retrying.")

        r = [x for x in results if x != -1]
        r = [item for sublist in r for item in sublist]
        return r


if __name__ == "__main__":
    from macrosynergy.management.utils import Config

    tickers: List[str] = [
        "USD_IMPINFB1Y_NSA",
        "SGD_GGPBGDPRATIO_NSA",
        "AUD_DU02YCRY_NSA",
        "AUD_DU02YCRY_VT10",
        "AUD_DU02YXR_NSA",
        "AUD_DU02YXR_VT10",
        "AUD_DU02YXRxEASD_NSA",
        "AUD_DU02YXRxLEV10_NSA",
        "AUD_DU05YCRY_NSA",
        "AUD_DU05YCRY_VT10",
        "AUD_DU05YXR_NSA",
        "AUD_DU05YXR_VT10",
        "AUD_DU05YXRxEASD_NSA",
        "AUD_DU05YXRxLEV10_NSA",
        "CHF_DU02YCRY_NSA",
        "CHF_DU02YCRY_VT10",
        "CHF_DU02YXR_NSA",
        "CHF_DU02YXR_VT10",
        "CHF_DU02YXRxEASD_NSA",
    ]

    mcd = MultiCredentialDownload(config=Config("./config_multicred.yml"))

    x = mcd._download(
        tickers=tickers,
        metrics=["all"],
        start_date="1990-01-01",
        end_date="2020-01-01",
    )
