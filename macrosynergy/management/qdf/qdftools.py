from macrosynergy.management.utils import ticker_df_to_qdf, qdf_to_ticker_df
from macrosynergy.management.decorators import argvalidation
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.constants import JPMAQS_METRICS
from typing import List, Tuple, Union, Dict
from numbers import Number
import pandas as pd


def _reduce_to_ticker_df(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    # each column is a ticker, so we can just select the columns we want
    return df.loc[:, tickers]


def _update_ticker_df(df_left: pd.DataFrame, df_right: pd.DataFrame) -> pd.DataFrame:
    for ticker in df_right.columns:
        df_left.loc[:, ticker] = df_right.loc[:, ticker]
    return df_left


def deconstruct_expression(expression: str) -> Tuple[str, str, str]:
    """
    Deconstruct a list of expressions into a list of expressions into cid,xcat,metric.
    JPMaQS expressions are of the form DB(JPMAQS,CID_XCAT,METRIC).
    Expressions that are not of this form are returned as (EXPRESSION, EXPRESSION, EXPRESSION)
    """
    result = (expression, expression, expression)
    try:
        assert expression.startswith("DB(JPMAQS,") and expression.endswith(")")
        expression = expression.replace("DB(JPMAQS,", "").replace(")", "")
        cid, xcat = expression.split(",", 1)
        xcat, metric = xcat.split(",", 1)
        result = (cid, xcat, metric)
        assert all([len(x) > 0 for x in result])
    except:
        pass
    return result


# implement a metaclass that applies the argvalidation decorator to all methods of the class
class ArgValidationMeta(type):
    def __new__(cls, name, bases, dct: dict):
        for key, value in dct.items():
            if callable(value):
                dct[key] = argvalidation(value)
        return super().__new__(cls, name, bases, dct)


class QDFTools:
    def __init__(self, *args, **kwargs):
        pass


class QDFWide(QDFTools):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reduce_df(self, *args, **kwargs):
        return _reduce_to_ticker_df(*args, **kwargs)


class ExpressionsDataFrame(QDFTools):
    def __init__(self, df: pd.DataFrame = pd.DataFrame(), *args, **kwargs):
        self.df = df

    def download(self, *args, **kwargs):
        raise NotImplementedError

    def _info(self, *args, **kwargs) -> Dict[str, List[str]]:
        if self.df.empty:
            return {}
        expr_tuples = [deconstruct_expression(expr) for expr in self.df.columns]
        _cids, _xcats, _metrics = zip(*expr_tuples)
        tickers = sorted(set([f"{cid}_{xcat}" for cid, xcat in zip(_cids, _xcats)]))
        return {
            "cids": sorted(set(_cids)),
            "xcats": sorted(set(_xcats)),
            "metrics": sorted(set(_metrics)),
            "tickers": tickers,
            "total_exprs": len(self.df.columns),
            "total_tickers": len(tickers),
        }

    @property
    def info(self, *args, **kwargs):
        d = self._info(*args, **kwargs)
        d.pop("tickers")
        return d
