from typing import List, Optional, TypeVar, Union, Dict, Any
import os
import datetime
import pandas as pd
import numpy as np
import warnings
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.qdf.base import QDFManagerBase, DateLike
from macrosynergy.management.qdf.load import Loader
from macrosynergy.management.qdf.save import Saver


import macrosynergy.management.qdf as qdfx
from macrosynergy.management.utils import get_cid, get_xcat


class QDFManager(QDFManagerBase):
    """
    A class to manage a large `QuantamentalDataFrame` object.
    """

    def __init__(
        self,
        qdf: Optional[QuantamentalDataFrame] = None,
        expression_df: Optional[pd.DataFrame] = None,
        df_dict: Optional[Dict[str, pd.DataFrame]] = None,
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
            self.df_dict = qdfx.methods.expression_df_to_df_dict(expression_df)
        elif qdf is not None:
            self.df_dict = qdfx.methods.qdf_to_df_dict(qdf)
        elif df_dict is not None:
            self.df_dict = df_dict
        elif ticker_df is not None:
            if metric is None:
                warnings.warn("No metric provided. Defaulting to 'value'.")
                metric = "value"
            self.df_dict = qdfx.methods.ticker_df_to_df_dict(ticker_df, metric)
        else:
            raise ValueError(
                "Either `qdf`, `expression_df`, or `df_dict` must be provided."
            )

        self.__init_properties__()

    def __init_properties__(self):
        """
        Initialise the properties of the `QDFManager` object.
        """

        self.tickers = qdfx.methods.get_tickers_from_df_dict(
            self.df_dict, common_metrics=False
        )
        self.ticker_dict = qdfx.methods.get_ticker_dict_from_df_dict(self.df_dict)

        self.date_range = qdfx.methods.get_date_range_from_df_dict(self.df_dict)
        self.start_date: pd.Timestamp = self.date_range.min()
        self.end_date: pd.Timestamp = self.date_range.max()
        dict_keys = list(self.df_dict.keys())
        for ki, kx in enumerate(dict_keys):
            if self.df_dict[kx].empty:
                del self.df_dict[kx]
        if self.tickers != []:
            self.cids = sorted(set(get_cid(self.tickers)))
            self.xcats = sorted(set(get_xcat(self.tickers)))
            self.metrics = sorted(self.df_dict.keys())
        else:
            self.cids = []
            self.xcats = []
            self.metrics = []

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
    ) -> Dict[str, pd.DataFrame]:
        """
        Query the `QuantamentalDataFrame`.
        """
        return qdfx.query.query_df_dict(*args, **kwargs, qdf_manager=self)

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
        return qdfx.methods.df_dict_to_qdf(
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

    def ticker_df(
        self,
        ticker: Optional[str] = None,
        cid: Optional[str] = None,
        cids: Optional[List[str]] = None,
        xcat: Optional[str] = None,
        xcats: Optional[List[str]] = None,
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
            if self.metrics == []:
                return pd.DataFrame(index=self.date_range)
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

    def update(
        self,
        qdfm: Optional["QDFManager"] = None,
        qdf: Optional[QuantamentalDataFrame] = None,
        expression_df: Optional[pd.DataFrame] = None,
        df_dict: Optional[Dict[str, pd.DataFrame]] = None,
        ticker_df: Optional[pd.DataFrame] = None,
        metric: Optional[str] = None,
    ) -> None:
        """
        Update the `QuantamentalDataFrame`.
        """
        # only one of qdfm, qdf, expression_df, df_dict, or ticker_df should be provided. only one
        if (
            sum([x is not None for x in [qdfm, qdf, expression_df, df_dict, ticker_df]])
            != 1
        ):
            raise ValueError(
                "Exactly one of `qdfm`, `qdf`, `expression_df`, `df_dict`, or `ticker_df` must be provided."
            )

        if qdfm is not None:
            self.df_dict = qdfx.methods.update_df_dict(self.df_dict, qdfm.df_dict)

        elif qdf is not None:
            self.df_dict = qdfx.methods.update_df_dict(
                self.df_dict,
                qdfx.methods.qdf_to_df_dict(qdf),
            )

        elif expression_df is not None:
            self.df_dict = qdfx.methods.update_df_dict(
                self.df_dict,
                qdfx.methods.expression_df_to_df_dict(expression_df),
            )

        elif df_dict is not None:
            self.df_dict = qdfx.methods.update_df_dict(self.df_dict, df_dict)

        elif ticker_df is not None:
            if metric is None:
                warnings.warn("No metric provided. Defaulting to 'value'.")
                metric = "value"
            self.df_dict = qdfx.methods.update_df_dict(
                self.df_dict,
                qdfx.methods.ticker_df_to_df_dict(ticker_df, metric),
            )

        # reinstiate with df dict
        self.__init_properties__()

    def drop(
        self,
        invert_selection: bool = False,
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
    ) -> None:
        """
        Drop the tickers from the `QuantamentalDataFrame`.
        """
        # query first
        query_dict: Dict[str, Dict[str, Union[List[str], pd.Timestamp]]] = (
            qdfx.query.get_query_dict_from_args(
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

        if invert_selection:
            for mtr in query_dict.keys():
                query_dict[mtr]["tickers"] = sorted(
                    set(self.tickers) - set(query_dict[mtr]["tickers"])
                )

        # drop the tickers
        self.df_dict = qdfx.methods.drop_tickers_from_df_dict(self.df_dict, query_dict)

    def query(
        self,
        query: str,
        *args,
        **kwargs,
    ) -> "QDFManager":
        """
        Query the `QuantamentalDataFrame`.
        """
        tx = kwargs.get("tickers", [])
        txnew = tx + [f"*{query}*".upper()]
        kwargs["tickers"] = txnew

        return QDFManager(df_dict=self._get_dict(*args, **kwargs))

    def iquery(
        self,
        query: str = None,
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
        *args,
        **kwargs,
    ) -> "QDFQueryView":
        if query is not None:
            kwargs["tickers"] = [f"*{query.upper()}*"] + kwargs.get("tickers", [])
        return QDFQueryView(manager=self).iquery(*args, **kwargs)


class QDFQueryView(QDFManager):
    """
    Query the `QuantamentalDataFrame`.
    """

    def __init__(
        self,
        manager: "QDFManager",
        view: Optional[Dict[str, Dict[str, Union[List[str], pd.Timestamp]]]] = None,
    ):
        """
        Initialise the `QDFQueryView` object.
        """
        self.manager = manager
        if view is None:
            view = qdfx.query.get_query_dict(qdf_manager=self.manager)
        self.view = view

        # set the properties
        self.tickers = qdfx.query.get_tickers_from_query_dict(
            self.view, common_metrics=False
        )
        self.ticker_dict = qdfx.query.get_ticker_dict_from_query_dict(self.view)
        self.metrics = list(self.view.keys())
        self.date_range = pd.bdate_range(
            min([d["start"] for d in self.view.values()]),
            max([d["end"] for d in self.view.values()]),
        )
        self.start_date = self.date_range.min()
        self.end_date = self.date_range.max()

    @property
    def df_dict(self):
        return self.manager.df_dict

    def iquery(
        self,
        query: str = None,
        *args,
        **kwargs,
    ) -> "QDFQueryView":
        """
        Return a new `QDFQueryView` object with the query applied.
        """
        assert (
            "ticker_dict" not in kwargs
        ), "ticker_dict is not a valid argument for `oquery`."
        if query is not None:
            kwargs["tickers"] = [f"*{query.upper()}*"] + kwargs.get("tickers", [])

        qdict: Dict[str, Dict[str, Union[List[str], pd.Timestamp]]] = (
            qdfx.query.get_query_dict_from_args(
                ticker_dict=self.ticker_dict, *args, **kwargs
            )
        )

        return QDFQueryView(manager=self.manager, view=qdict)

    def qdf(
        self,
        *args,
        **kwargs,
    ) -> QuantamentalDataFrame:
        """
        Query the `QuantamentalDataFrame`.
        """
        return self.compile().qdf(*args, **kwargs)

    def ticker_df(
        self,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Query the `QuantamentalDataFrame`.
        """
        return self.compile().ticker_df(*args, **kwargs)

    def query(
        self,
        *args,
        **kwargs,
    ) -> "QDFQueryView":
        """
        Query the `QuantamentalDataFrame`.
        """
        return self.compile().query(*args, **kwargs)

    def compile(self) -> QDFManager:
        """
        Compile the `QDFQueryView` into a `QuantamentalDataFrame`.
        """
        return QDFManager(
            df_dict=qdfx.query.get_query_df_dict(
                query_dict=self.view,
                qdf_manager=self.manager,
            )
        )
