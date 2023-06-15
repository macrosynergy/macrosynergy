from typing import List
import pandas as pd
import os
import glob
from functools import lru_cache
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from macrosynergy.download.jpmaqs import JPMaQSDownload, DataQueryInterface

logger = logging.getLogger(__name__)
cache = lru_cache(maxsize=None)


class LocalDataQueryInterface(DataQueryInterface):
    def __init__(self, local_path: str, fmt="pkl"):
        self.local_path = os.path.abspath(local_path)
        # check if the local path exists
        if not os.path.exists(self.local_path):
            raise FileNotFoundError(
                f"The local path provided : {self.local_path}, does not exist."
            )
        self.store_format = fmt
        logger.info(f"LocalDataQuery initialized with local_path: {self.local_path}")

    @cache
    def _find_ticker_files(
        self,
    ) -> List[str]:
        """
        Returns a list of files in the local path
        """
        # get all files in the local path with the correct extension, at any depth

        files: List[str] = glob.glob(
            os.path.join(self.local_path, f"*.{self.store_format}"), recursive=True
        )
        return files

    @cache
    def _get_ticker_path(self, ticker: str) -> str:
        """
        Returns the absolute path to the ticker file.

        :param ticker: The ticker to find the path for.
        :return: The absolute path to the ticker file.
        :raises FileNotFoundError: If the ticker is not found in the local path.
        """
        files: List[str] = self._find_ticker_files()
        for f in files:
            if ticker == f.split(os.sep)[-1].split(".")[0]:
                return f
        raise FileNotFoundError(f"Ticker {ticker} not found in {self.local_path}")

    def get_catalogue(self, *args, **kwargs) -> List[str]:
        """
        Returns a list of tickers available in the local
        tickerstore.
        """
        tickers: List[str] = [
            os.path.basename(f).split(".")[0] for f in self._find_ticker_files()
        ]
        return tickers

    def check_connection(self, verbose=False) -> bool:
        # check if _find_ticker_files returns anything
        if len(self._find_ticker_files()) > 0:
            return True
        else:
            fmt_long: str = (
                "pickle (*.pkl)" if self.store_format == "pkl" else "csv (*.csv)"
            )
            raise FileNotFoundError(
                f"The local path provided : {self.local_path}, "
                f"does not contain {fmt_long} files."
            )

    def load_data(
        self,
        expressions: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        def _load_df(ticker: str) -> pd.DataFrame:
            if self.store_format == "pkl":
                return pd.read_pickle(os.path.join(self.local_path, ticker + ".pkl"))
            elif self.store_format == "csv":
                return pd.read_csv(os.path.join(self.local_path, ticker + ".csv"))

        def _get_df(
            cid: str, xcat: str, metrics: List[str], start_date: str, end_date: str
        ) -> pd.DataFrame:
            df: pd.DataFrame = _load_df(ticker=f"{cid}_{xcat}")
            df = df[["real_date"] + metrics]
            df["real_date"] = pd.to_datetime(df["real_date"])
            df = df.loc[(df["real_date"] >= start_date) & (df["real_date"] <= end_date)]
            df["cid"] = cid
            df["xcat"] = xcat
            return df[["real_date", "cid", "xcat"] + metrics]

        # is overloaded to accept a list of expressions
        deconstr_expressions: List[List[str]] = JPMaQSDownload.deconstruct_expression(
            expression=expressions
        )

        pd.DataFrame = pd.concat(
            [
                _get_df(
                    cid=cidx,
                    xcat=xcatx,
                    metrics=metricsx,
                    start_date=start_date,
                    end_date=end_date,
                )
                for cidx, xcatx, metricsx in deconstr_expressions
            ],
            ignore_index=True,
            axis=0,
        )
        return pd.DataFrame

    def download_data(
        self,
        expressions: List[str],
        start_date: str = "2000-01-01",
        end_date: str = None,
        show_progress: bool = False,
        endpoint: str = ...,
        calender: str = "CAL_ALLDAYS",
        frequency: str = "FREQ_DAY",
        conversion: str = "CONV_LASTBUS_ABS",
        nan_treatment: str = "NA_NOTHING",
        reference_data: str = "NO_REFERENCE_DATA",
        retry_counter: int = 0,
        delay_param: float = ...,
    ) -> pd.DataFrame:
        # divide expressions into batches of 10
        batched_expressions: List[List[str]] = [
            expressions[i : i + 10] for i in range(0, len(expressions), 10)
        ]

        with Pool(cpu_count() - 1) as p:
            df: pd.DataFrame = pd.concat(
                list(
                    tqdm(
                        p.imap(
                            self.load_data,
                            batched_expressions,
                        ),
                        total=len(batched_expressions),
                        disable=not show_progress,
                        desc="Downloading data",
                    )
                ),
                ignore_index=True,
                axis=0,
            )

        return df


class LocalCache(JPMaQSDownload):
    def __init__(self, local_path: str, fmt="pkl"):
        self.local_path = os.path.abspath(local_path)
        self.store_format = fmt
        super().__init__(
            client_id="<local>",
            client_secret=f"<{self.local_path}>",
            check_connection=False,
        )
        self.dq_interface = LocalDataQueryInterface(
            local_path=self.local_path, fmt=self.store_format
        )

    def download(
        self,
        jpmaqs_df=True,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        kwargs.update({"as_dataframe": True, "get_catalogue": True})
        df: pd.DataFrame = super().download(*args, **kwargs)
        if jpmaqs_df:
            return df
        else:
            df["ticker"]: str = df["cid"] + "_" + df["xcat"]
            df = df.drop(["cid", "xcat"], axis=1)
            return df


class DownloadSnapshot(JPMaQSDownload):
    def __init__(self, store_path: str, store_format: str = "pkl", *args, **kwargs):
        """
        Downloads a snapshot of the entire database and stores it locally.

        Parameters
        :param <str> store_path: The path to store the data. Prefer absolute paths.
        :param <str> store_format: The format to store the data in. Either "pkl", "csv" or "all".
            "all" will store both formats.
        :param <str> client_id: The client id for the JPMaQS API.
        :param <str> client_secret: The client secret for the JPMaQS API.
        """
        self.store_path: str = os.path.abspath(store_path)
        self.store_format: str = store_format
        super().__init__(*args, **kwargs)

    def _save_df(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Saves the dataframe to the store_path.

        :param df: The dataframe to save.


        """
        save_csv: bool = self.store_format == "csv" or self.store_format == "all"
        save_pkl: bool = self.store_format == "pkl" or self.store_format == "all"
        if save_csv:
            os.makedirs(os.path.join(self.store_path, "csv"), exist_ok=True)
            csvs_path: str = os.path.join(self.store_path, "csv")
        if save_pkl:
            os.makedirs(os.path.join(self.store_path, "pkl"), exist_ok=True)
            pkls_path: str = os.path.join(self.store_path, "pkl")

        for (cid, xcat), dfx in df.groupby(["cid", "xcat"]):
            dfx = (
                dfx.drop(["cid", "xcat"], axis=1)
                .dropna(axis=0, how="any")
                .reset_index(drop=True)
            )
            if save_pkl:
                dfx.to_pickle(os.path.join(pkls_path, f"{cid}_{xcat}.pkl"))
            if save_csv:
                dfx.to_csv(os.path.join(csvs_path, f"{cid}_{xcat}.csv"), index=False)

    def download(
        self,
        batch_size: int = 500,
        show_progress: bool = False,
    ) -> None:
        """
        Downloads the entire database and stores it locally.
        All tickers are downloaded in batches of `batch_size`.
        This is to avoid memory and connectivity issues.
        NOTE: This will take a long time to run - 1hr+.

        :param <int> batch_size: The batch size to use when downloading the data.
        :param <bool> show_progress: Whether to show a progress bar or not.
        :return: None
        """

        all_tickers: List[str] = self.get_catalogue()
        start_date: str = "1990-01-01"
        metrics: List[str] = self.valid_metrics
        end_date: str = None

        # batch the expressions into 500
        batched_tickers: List[List[str]] = [
            all_tickers[i : i + batch_size]
            for i in range(0, len(all_tickers), batch_size)
        ]

        # download the data
        for batch in tqdm(
            batched_tickers, disable=not show_progress, desc="Downloading snapshot"
        ):
            df: pd.DataFrame = super().download(
                tickers=batch,
                metrics=metrics,
                start_date=start_date,
                end_date=end_date,
                show_progress=show_progress,
            )
            self._save_df(df=df)
