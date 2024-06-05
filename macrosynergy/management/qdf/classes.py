from typing import List, TypeVar
import datetime
import pandas as pd
import numpy as np


DateLike = TypeVar("DateLike", str, pd.Timestamp, np.datetime64, datetime.datetime)


class QDFManagerBase:
    """
    Base class for the `QDFManager` and related classes.
    """

    def __init__(self):
        self._tickers: List[str] = None
        self.cids: List[str] = None
        self.xcats: List[str] = None
        self.metrics: List[str] = None
        self.date_range: pd.DatetimeIndex = None
        self.start_date: pd.Timestamp = None
        self.end_date: pd.Timestamp = None
        self.df_dict: dict[str, pd.DataFrame] = None

    @property
    def tickers(self):
        return self._tickers

    @tickers.setter
    def tickers(self, value):
        self._tickers = value

    @property
    def start_date(self):
        return self._start_date

    @start_date.setter
    def start_date(self, value):
        self._start_date = value

    @property
    def end_date(self):
        return self._end_date

    @end_date.setter
    def end_date(self, value):
        self._end_date = value

    @property
    def date_range(self):
        return self._date_range

    @date_range.setter
    def date_range(self, value):
        self._date_range = value
