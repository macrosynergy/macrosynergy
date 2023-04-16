from typing import Dict, List, Optional, Union, Any
from macrosynergy.download import JPMaQSDownload
from macrosynergy.management.utils import Config
import pandas as pd
import os
import pickle
from timeit import default_timer as timer
from tqdm import tqdm
from joblib import Parallel, delayed

###############
# Class MultiCredentials


class MultiCredentialDownload:
    def __init__(
        self, config: Config, max_workers: int = 4, save_path: str = "./pickledata"
    ):
        self.config = config
        self._max_workers = max_workers
        self.save_path = save_path
        self.chunk_len = 2000
        os.makedirs(self.save_path, exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _download_chunk(
        self,
        credentials: Config,
        download_args: dict,
        save_id: str,
    ):
        # force the download skip the dataframe conversion
        download_args["as_dataframe"] = False
        try:
            start = timer()
            with open(f"{self.save_path}/{save_id}.pkl", "wb") as f:
                pickle.dump(
                    obj=JPMaQSDownload(**credentials.oauth(mask=False)).download(
                        **download_args
                    ),
                    file=f,
                )
            end = timer()
            print(f"Download and save from {save_id} : {end - start} seconds")

            return True
        except Exception as e:
            print(e)
            return False

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
        if metrics == ["all"]:
            metrics: List[str] = ["value", "grading", "eop_lag", "mop_lag"]

        all_expressions: List[str] = JPMaQSDownload.construct_expressions(
            tickers=tickers, cids=cids, xcats=xcats, metrics=metrics
        )

        # form "chunks" of 1000
        chunks: List[List[str]] = [
            all_expressions[i : i + self.chunk_len]
            for i in range(0, len(all_expressions), self.chunk_len)
        ]

        download_args_list: List[dict] = [
            {
                "expressions": chunk,
                "start_date": start_date,
                "end_date": end_date,
                "as_dataframe": False,
                "report_time_taken": report_time_taken,
                "report_egress": report_egress,
            }
            for chunk in chunks
        ]

        save_ids: List[str] = [f"{i}" for i in range(len(download_args_list))]

        # create a list of creds = [cred1, cred,2, ... credn, cred1, cred2, ... credn] to the length of the download args. there are a limited number of creds, so we will just loop through them

        creds: List[Config] = []
        available_creds_count = len(self.config.multi_credentials())
        for i in range(len(download_args_list)):
            creds.append(self.config.multi_credentials()[i % available_creds_count])

        results: List[bool] = Parallel(n_jobs=self._max_workers)(
            delayed(self._download_chunk)(
                credentials=cred,
                download_args=download_args,
                save_id=save_id,
            )
            for cred, download_args, save_id in tqdm(
                zip(creds, download_args_list, save_ids), total=len(download_args_list)
            )
        )

        # if any of the results are false, then we need to re-run the failed ones
        if not all(results):
            failed_ids: List[str] = [
                "F" + save_id
                for save_id, result in zip(save_ids, results)
                if not result
            ]
            failed_args: List[dict] = [
                download_args
                for download_args, result in zip(download_args_list, results)
                if not result
            ]
            creds: List[Config] = []
            for i in range(len(failed_args)):
                creds.append(self.config.multi_credentials()[i % available_creds_count])

            results: List[bool] = Parallel(n_jobs=self._max_workers)(
                delayed(self._download_chunk)(
                    credentials=cred,
                    download_args=download_args,
                    save_id=save_id,
                )
                for cred, download_args, save_id in tqdm(
                    zip(creds, failed_args, failed_ids), total=len(failed_args)
                )
            )

            if not all(results):
                print(
                    f"Failed to download {len(failed_ids)*self.chunk_len} expressions"
                )

            return


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

    mcd = MultiCredentialDownload(config=Config("./config_multicred.yml"), 
                                    save_path="./pickle_data",)

    x = mcd.download(
        tickers=tickers,
        metrics=["all"],
        start_date="1990-01-01",
        end_date="2020-01-01",
    )
