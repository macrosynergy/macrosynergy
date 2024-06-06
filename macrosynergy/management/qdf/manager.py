from typing import List, Optional, TypeVar
import os
import datetime
import pandas as pd
import numpy as np
import warnings
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.qdf.classes import QDFManagerBase, DateLike
from macrosynergy.management.qdf.load import Loader
from macrosynergy.management.qdf.save import Saver
from macrosynergy.management.qdf.methods import (
    qdf_to_df_dict,
    df_dict_to_qdf,
    ticker_df_to_df_dict,
    expression_df_to_df_dict,
    get_tickers_from_df_dict,
    get_date_range_from_df_dict,
)

from macrosynergy.management.qdf.query import query_df_dict

from macrosynergy.management.utils import get_cid, get_xcat


class QDFManager(QDFManagerBase):
    """
    A class to manage a large `QuantamentalDataFrame` object.
    """

    def __init__(
        self,
        qdf: Optional[QuantamentalDataFrame] = None,
        expression_df: Optional[pd.DataFrame] = None,
        df_dict: Optional[dict[str, pd.DataFrame]] = None,
        ticker_df: Optional[pd.DataFrame] = None,
        metric: Optional[str] = None,
    ) -> None:
        """
        Initialise the `QDFManager` object.
        """
        if not any([qdf, expression_df, df_dict, ticker_df]):
            raise ValueError(
                "Either `qdf`, `expression_df`, `df_dict`, or `ticker_df` must be provided."
            )
        if expression_df is not None:
            self.df_dict = expression_df_to_df_dict(expression_df)
        elif qdf is not None:
            self.df_dict = qdf_to_df_dict(qdf)
        elif df_dict is not None:
            self.df_dict = df_dict
        elif ticker_df is not None:
            if metric is None:
                warnings.warn("No metric provided. Defaulting to 'value'.")
                metric = "value"
            self.df_dict = ticker_df_to_df_dict(ticker_df, metric)
        else:
            raise ValueError(
                "Either `qdf`, `expression_df`, or `df_dict` must be provided."
            )

        self._tickers = get_tickers_from_df_dict(self.df_dict, common_metrics=True)
        self.cids = sorted(set(get_cid(self._tickers)))
        self.xcats = sorted(set(get_xcat(self._tickers)))
        self.metrics = sorted(self.df_dict.keys())
        self.date_range = get_date_range_from_df_dict(self.df_dict)
        self.start_date: pd.Timestamp = self.date_range.min()
        self.end_date: pd.Timestamp = self.date_range.max()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback): ...

    @staticmethod
    def load_from_disk(
        path: str,
        format: str = "csvs",
    ) -> "QDFManager":
        """
        Load a `QuantamentalDataFrame` from disk.

        Parameters
        :param <str> path: The path to the directory containing the data.
        :param <str> format: The format of the data. Options are "csvs",
        """
        return QDFManager(df_dict=Loader.load_from_disk(path, format=format))

    def save_to_disk(
        self,
        path: str,
        format: str = "pkl",
    ) -> None:
        """
        Save the `QuantamentalDataFrame` to disk.

        Parameters
        :param <str> path: The path to the directory to save the data.
        :param <str> format: The format of the data. Options are "csvs",
        """
        if format == "pkl":
            Saver.save_pkl_to_disk(self.df_dict, path)
        else:
            for metric, df in self.df_dict.items():
                Saver.save_single_csv_to_disk(df, os.path.join(path, f"{metric}.csv"))

    def _get_dict(
        self,
        *args,
        **kwargs,
        # cid: Optional[str] = None,
        # cids: Optional[List[str]] = None,
        # xcat: Optional[str] = None,
        # xcats: Optional[List[str]] = None,
        # ticker: Optional[str] = None,
        # tickers: Optional[List[str]] = None,
        # start: Optional[DateLike] = None,
        # end: Optional[DateLike] = None,
        # metric: Optional[str] = None,
        # metrics: Optional[List[str]] = None,
        # cross_section_groups: Optional[List[str]] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Query the `QuantamentalDataFrame`.
        """
        return query_df_dict(*args, **kwargs, qdf_manager=self)

    def qdf(
        self,
        cid: Optional[str] = None,
        cids: Optional[List[str]] = None,
        xcat: Optional[str] = None,
        xcats: Optional[List[str]] = None,
        ticker: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        metrics: Optional[str] = "value",
        cross_section_groups: Optional[List[str]] = None,
    ) -> QuantamentalDataFrame:
        """
        Query the `QuantamentalDataFrame`.
        """
        return df_dict_to_qdf(
            self._get_dict(
                cid=cid,
                cids=cids,
                xcat=xcat,
                xcats=xcats,
                ticker=ticker,
                tickers=tickers,
                start=start,
                end=end,
                metrics=metrics,
                cross_section_groups=cross_section_groups,
            )
        )

    def obj_query(
        self,
        query: str,
        *args,
        **kwargs,
    ) -> QuantamentalDataFrame:
        """
        Query the `QuantamentalDataFrame`.
        """
        tx = kwargs.get("tickers", [])
        txnew = tx + [f"*{query}*".upper()]
        kwargs["tickers"] = txnew

        return self.qdf(*args, **kwargs)

    def query(
        self,
        query: str,
        *args,
        **kwargs,
    ):
        """
        Query the `QuantamentalDataFrame`.
        """
        tx = kwargs.get("tickers", [])
        txnew = tx + [f"*{query}*".upper()]
        kwargs["tickers"] = txnew

        return self._get_dict(*args, **kwargs)

    def ticker_df(
        self,
        cid: Optional[str] = None,
        cids: Optional[List[str]] = None,
        xcat: Optional[str] = None,
        xcats: Optional[List[str]] = None,
        ticker: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        metric: str = "value",
        cross_section_groups: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Query the `QuantamentalDataFrame`.
        """
        if metric not in self.metrics:
            raise ValueError(f"Invalid metric: {metric}. Options are {self.metrics}")

        # shortcut if no query
        if all(
            [
                _x is None
                for _x in [
                    cid,
                    cids,
                    xcat,
                    xcats,
                    ticker,
                    tickers,
                    cross_section_groups,
                ]
            ]
        ):
            start = self.start_date if start is None else start
            end = self.end_date if end is None else end
            return self.df_dict[metric].loc[start:end]

        return self._get_dict(
            cid=cid,
            cids=cids,
            xcat=xcat,
            xcats=xcats,
            ticker=ticker,
            tickers=tickers,
            start=start,
            end=end,
            metric=metric,
            cross_section_groups=cross_section_groups,
        )[metric]
