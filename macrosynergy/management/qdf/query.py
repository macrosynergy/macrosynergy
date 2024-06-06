from typing import Dict, Iterable, List, Optional, Union
from collections.abc import Mapping, Iterable
from .classes import QDFManagerBase, DateLike

from .methods import get_ticker_dict_from_df_dict
import macrosynergy.management.constants as msy_constants
import pandas as pd
import fnmatch
import itertools


def _substring_matcher(
    substrings: Iterable[str],
    all_available_tickers: Iterable[str],
) -> List[str]:
    """
    Get the tickers from the query parameters.
    """
    assert isinstance(substrings, Iterable)
    assert isinstance(all_available_tickers, Iterable)
    return list(
        itertools.chain.from_iterable(
            fnmatch.filter(all_available_tickers, substring) for substring in substrings
        )
    )


def get_tickers_from_query_dict(
    query_dict: Dict[str, Dict[str, Union[List[str], pd.Timestamp]]],
    common_metrics: bool = False,
) -> List[str]:
    """
    Get the tickers from a query dictionary.
    """
    if common_metrics:
        tickers: List[str] = list(
            set.intersection(*map(set, [t["tickers"] for t in query_dict.values()]))
        )
    else:
        tickers: List[str] = list(
            itertools.chain.from_iterable(t["tickers"] for t in query_dict.values())
        )
    return tickers


def get_ticker_dict_from_query_dict(
    query_dict: Dict[str, Dict[str, Union[List[str], pd.Timestamp]]],
) -> Dict[str, List[str]]:
    """
    Get a dictionary of tickers from a query dictionary.
    """
    return {metric: query_dict[metric]["tickers"] for metric in query_dict}


def _run_ticker_query(
    ticker_dict: Dict[str, List[str]],
    all_available_tickers: Iterable[str],
    all_cids: Iterable[str],
    all_xcats: Iterable[str],
    cids: Optional[Iterable[str]] = None,
    xcats: Optional[Iterable[str]] = None,
    tickers: Optional[Iterable[str]] = None,
    metrics: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Get the tickers from the query parameters.
    """

    if tickers is None:
        if cids is None or cids == []:
            cids = all_cids
        if xcats is None or xcats == []:
            xcats = all_xcats
        tickers = [f"{cid}_{xcat}" for cid in cids for xcat in xcats]

    _contains_wildcard = lambda x: "*" in x or "?" in x
    wtickers = list(filter(_contains_wildcard, tickers))
    if wtickers:
        tickers = list(set(tickers) - set(wtickers))
        tickers += _substring_matcher(wtickers, all_available_tickers)

    metrics = list(set(metrics) & set(ticker_dict.keys()))
    if cids is None or xcats is None:
        cids: List[str] = []
        xcats: List[str] = []

    if tickers is None:
        tickers: List[str] = []

    tickers += [f"{cid}_{xcat}" for cid in cids for xcat in xcats]
    tickers = set(tickers)

    return {m: sorted(set(ticker_dict[m]).intersection(tickers)) for m in metrics}


def get_query_dict_from_args(
    ticker_dict: Dict[str, List[str]],
    cid: Optional[str] = None,
    cids: Optional[List[str]] = None,
    xcat: Optional[str] = None,
    xcats: Optional[List[str]] = None,
    ticker: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
    date_range: Optional[pd.DatetimeIndex] = None,
    metric: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    cross_section_groups: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Union[List[str], pd.Timestamp]]]:
    metrics = [metric] if isinstance(metric, str) else []
    metrics = [metrics] if isinstance(metrics, str) else []
    metrics = metrics + (metrics or [])

    if metrics == ["all"] or not metrics:
        metrics = ticker_dict.keys()
    start: DateLike = pd.Timestamp("1900-01-01") if start is None else start
    end: DateLike = pd.Timestamp("2100-01-01") if end is None else end
    if date_range is not None:
        date_range: pd.DatetimeIndex = pd.DatetimeIndex(map(pd.Timestamp, date_range))
        start = date_range.min()
        end = date_range.max()

    if not isinstance(start, pd.Timestamp):
        start = pd.Timestamp(start)
    if not isinstance(end, pd.Timestamp):
        end = pd.Timestamp(end)

    if start < pd.Timestamp("1900-01-01"):
        start = pd.Timestamp("1900-01-01")
    if end > pd.Timestamp("2100-01-01"):
        end = pd.Timestamp("2100-01-01")

    if start > end:
        start, end = end, start
    all_available_tickers = list(
        set(itertools.chain.from_iterable(ticker_dict.values()))
    )
    all_cids = list(set(map(lambda x: x.split("_")[0], all_available_tickers)))
    all_xcats = list(set(map(lambda x: x.split("_")[1], all_available_tickers)))

    if isinstance(cross_section_groups, str):
        cross_section_groups = [cross_section_groups]
    cross_section_groups = cross_section_groups or []
    if cids is None:
        cids = []
    for cxg in cross_section_groups or []:
        _cx = cxg.lower()
        if _cx in msy_constants.cross_section_groups:
            cids = list(set(cids + msy_constants.cross_section_groups[_cx]))
    cids = [cids] if isinstance(cids, str) else cids
    xcats = [xcats] if isinstance(xcats, str) else xcats

    if isinstance(cid, str):
        cids = [cid] + (cids or [])
    if isinstance(xcat, str):
        xcats = [xcat] + (xcats or [])
    if isinstance(ticker, str):
        tickers = [ticker] + (tickers or [])

    ticker_query: Dict[str, List[str]] = _run_ticker_query(
        ticker_dict=ticker_dict,
        all_available_tickers=all_available_tickers,
        all_cids=all_cids,
        all_xcats=all_xcats,
        cids=cids,
        xcats=xcats,
        tickers=tickers,
        metrics=metrics,
    )

    return {
        metric: {
            "tickers": tickers,
            "start": start,
            "end": end,
        }
        for metric, tickers in ticker_query.items()
    }


def get_query_dict(
    qdf_manager: QDFManagerBase,
    cid: Optional[str] = None,
    cids: Optional[List[str]] = None,
    xcat: Optional[str] = None,
    xcats: Optional[List[str]] = None,
    ticker: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
    date_range: Optional[pd.DatetimeIndex] = None,
    metric: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    cross_section_groups: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Union[List[str], pd.Timestamp]]]:
    metrics = [metric] if isinstance(metric, str) else []
    metrics = [metrics] if isinstance(metrics, str) else []
    metrics = metrics + (metrics or [])

    if metrics == ["all"] or not metrics:
        metrics = qdf_manager.metrics
    start: DateLike = qdf_manager.start_date if start is None else start
    end: DateLike = qdf_manager.end_date if end is None else end
    if date_range is not None:
        date_range: pd.DatetimeIndex = pd.DatetimeIndex(map(pd.Timestamp, date_range))
        start = date_range.min()
        end = date_range.max()

    if not isinstance(start, pd.Timestamp):
        start = pd.Timestamp(start)
    if not isinstance(end, pd.Timestamp):
        end = pd.Timestamp(end)

    if start < qdf_manager.start_date:
        start = qdf_manager.start_date
    if end > qdf_manager.end_date:
        end = qdf_manager.end_date

    if start > end:
        start, end = end, start
    all_available_tickers = qdf_manager.tickers
    all_cids = qdf_manager.cids
    all_xcats = qdf_manager.xcats
    ticker_dict = get_ticker_dict_from_df_dict(qdf_manager.df_dict)
    (qdf_manager.df_dict)
    if isinstance(cross_section_groups, str):
        cross_section_groups = [cross_section_groups]
    cross_section_groups = cross_section_groups or []
    if cids is None:
        cids = []
    for cxg in cross_section_groups or []:
        _cx = cxg.lower()
        if _cx in msy_constants.cross_section_groups:
            cids = list(set(cids + msy_constants.cross_section_groups[_cx]))
    cids = [cids] if isinstance(cids, str) else cids
    xcats = [xcats] if isinstance(xcats, str) else xcats

    if isinstance(cid, str):
        cids = [cid] + (cids or [])
    if isinstance(xcat, str):
        xcats = [xcat] + (xcats or [])
    if isinstance(ticker, str):
        tickers = [ticker] + (tickers or [])

    ticker_query: Dict[str, List[str]] = _run_ticker_query(
        ticker_dict=ticker_dict,
        all_available_tickers=all_available_tickers,
        all_cids=all_cids,
        all_xcats=all_xcats,
        cids=cids,
        xcats=xcats,
        tickers=tickers,
        metrics=metrics,
    )

    return {
        metric: {
            "tickers": tickers,
            "start": start,
            "end": end,
        }
        for metric, tickers in ticker_query.items()
    }


def get_query_df_dict(
    query_dict: Dict[str, Dict[str, Union[List[str], pd.Timestamp]]],
    qdf_manager: QDFManagerBase,
) -> Dict[str, pd.DataFrame]:
    return {
        metric: qdf_manager.df_dict[metric].loc[
            query_dict[metric]["start"] : query_dict[metric]["end"],
            query_dict[metric]["tickers"],
        ]
        for metric in query_dict
    }


def query_df_dict(
    qdf_manager: QDFManagerBase,
    cid: Optional[str] = None,
    cids: Optional[List[str]] = None,
    xcat: Optional[str] = None,
    xcats: Optional[List[str]] = None,
    ticker: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
    date_range: Optional[pd.DatetimeIndex] = None,
    # substring: Optional[str] = None,
    # substrings: Optional[List[str]] = None,
    metric: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    cross_section_groups: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Query a dictionary of `pd.DataFrame`s.
    """
    query_dict: Dict[str, Dict[str, Union[List[str], pd.Timestamp]]] = get_query_dict(
        qdf_manager,
        cid=cid,
        cids=cids,
        xcat=xcat,
        xcats=xcats,
        ticker=ticker,
        tickers=tickers,
        start=start,
        end=end,
        date_range=date_range,
        metric=metric,
        metrics=metrics,
        cross_section_groups=cross_section_groups,
    )
    return get_query_df_dict(query_dict=query_dict, qdf_manager=qdf_manager)
