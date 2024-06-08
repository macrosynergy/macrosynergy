import pandas as pd
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import (
    deconstruct_expression,
    qdf_to_ticker_df,
    concat_qdfs,
    ticker_df_to_qdf,
    get_ticker,
)
import itertools
from typing import List, Dict


def qdf_to_df_dict(qdf: QuantamentalDataFrame) -> Dict[str, pd.DataFrame]:
    """
    Convert a `QuantamentalDataFrame` to a dictionary of `pd.DataFrame`s.
    """
    metrics: List[str] = qdf.columns.difference(
        QuantamentalDataFrame.IndexCols
    ).to_list()

    df_dict: Dict[str, pd.DataFrame] = {
        metric: qdf_to_ticker_df(qdf, metric) for metric in metrics
    }

    return df_dict


def df_dict_to_qdf(df_dict: Dict[str, pd.DataFrame]) -> QuantamentalDataFrame:
    """
    Convert a dictionary of `pd.DataFrame`s to a `QuantamentalDataFrame`.
    """

    def _is_empty(df: pd.DataFrame) -> bool:
        return df is None or (df.empty)

    r = concat_qdfs(
        [
            ticker_df_to_qdf(df, metric)
            for metric, df in df_dict.items()
            if not _is_empty(df)
        ]
    )
    if r is None:
        return pd.DataFrame()

    return r


def ticker_df_to_df_dict(
    ticker_df: pd.DataFrame, metric: str
) -> Dict[str, pd.DataFrame]:
    """
    Convert a dictionary of tickers to a dictionary of `pd.DataFrame`s.
    """
    return {metric: ticker_df}


def expression_df_to_df_dict(
    expression_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Convert an expression dataframe to a dictionary of `pd.DataFrame`s.
    """
    d_exprs: List[List[str]] = deconstruct_expression(expression_df.columns.to_list())
    de_df = pd.DataFrame(d_exprs, columns=["cid", "xcat", "metric"])

    unique_metrics: List[str] = list(set(d_e[-1] for d_e in d_exprs))

    df_dict: Dict[str, pd.DataFrame] = {
        metric: expression_df[expression_df.columns[de_df["metric"] == metric]]
        for metric in unique_metrics
    }
    for metric in unique_metrics:
        df_dict[metric].columns = [get_ticker(col) for col in df_dict[metric].columns]

    return df_dict


def get_ticker_dict_from_df_dict(
    df_dict: Dict[str, pd.DataFrame]
) -> Dict[str, List[str]]:
    """
    Get a dictionary of tickers from a dictionary of `pd.DataFrame`s.
    """
    return {metric: df.columns.to_list() for metric, df in df_dict.items()}


def get_tickers_from_df_dict(
    df_dict: Dict[str, pd.DataFrame], common_metrics: bool = False
) -> List[str]:
    """
    Get the tickers from a dictionary of `pd.DataFrame`s.
    """
    ticker_dict: Dict[str, List[str]] = get_ticker_dict_from_df_dict(df_dict)

    if common_metrics:
        tickers: List[str] = list(set.intersection(*map(set, ticker_dict.values())))
    else:
        tickers: List[str] = list(
            set(itertools.chain.from_iterable(ticker_dict.values()))
        )

    return sorted(tickers)


def get_date_range_from_df_dict(df_dict: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """
    Get the date range from a dictionary of `pd.DataFrame`s.
    """
    dts = {metric: set(df.index) for metric, df in df_dict.items()}
    return pd.DatetimeIndex(sorted(set.union(*dts.values())))


def update_ticker_df(
    left: pd.DataFrame,
    right: pd.DataFrame,
    overwrite: bool = True,
) -> pd.DataFrame:
    """
    Update the `left` dataframe with the data from the `right` dataframe.
    """
    if overwrite:
        new_cols: pd.Index = right.columns
    else:
        new_cols: pd.Index = right.columns.difference(left.columns)

    if new_cols.empty:
        return left

    return pd.concat([left, right[new_cols]], axis=1)


def drop_tickers_from_ticker_df(
    ticker_df: pd.DataFrame, tickers: List[str]
) -> pd.DataFrame:
    """
    Drop tickers from a ticker dataframe.
    """
    return ticker_df.drop(columns=tickers, errors="ignore")


def update_df_dict(
    df_dict: Dict[str, pd.DataFrame],
    new_df_dict: Dict[str, pd.DataFrame],
    overwrite: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Update the `df_dict` structure with new data.
    """
    for metric, df in new_df_dict.items():
        if metric in df_dict:
            df_dict[metric] = update_ticker_df(
                left=df_dict[metric], right=df, overwrite=overwrite
            )
        else:
            df_dict[metric] = df

    return df_dict


def drop_tickers_from_df_dict(
    df_dict: Dict[str, pd.DataFrame], tickers: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Drop tickers from a dictionary of `pd.DataFrame`s.
    """
    return {
        metric: drop_tickers_from_ticker_df(df, tickers)
        for metric, df in df_dict.items()
    }
