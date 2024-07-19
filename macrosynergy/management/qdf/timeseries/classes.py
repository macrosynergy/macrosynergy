import macrosynergy.management.qdf.timeseries.methods as tsmethods
from macrosynergy.management.types import ArgValidationMeta
from numbers import Number

import pandas as pd
import numpy as np

from typing import List, Tuple, Union, Optional


class TimeSeries(metaclass=ArgValidationMeta):
    """
    Class to handle time series data.
    """

    def __init__(self, ts: Union[pd.Series, "TimeSeries"]):
        if isinstance(ts, TimeSeries):
            self = ts
        else:
            self.ts = ts

    def information_state_changes(self, threshold: float = 0.0) -> pd.Series:
        """Get the information state changes of a time series."""
        return tsmethods.information_state_changes(self.ts, threshold)

    def z_score(self) -> pd.Series:
        return tsmethods.z_score(self.ts)

    def infer_frequency(self) -> str:
        return tsmethods.infer_frequency(self.ts)


class SparseTimeSeries(TimeSeries):
    """Class to handle sparse time series data."""

    def __init__(self, ts: Union[pd.Series, TimeSeries]):
        self.ts = TimeSeries(ts).information_state_changes()

    def to_daily(
        self,
        value: Optional[Number] = None,
        limit: Optional[int] = None,
    ) -> pd.Series:
        _sd, _ed = self.ts.index.min(), self.ts.index.max()
        idx = pd.bdate_range(_sd, _ed)
        ts = self.ts.reindex(idx)
        return tsmethods.forward_fill(ts=ts, value=value, limit=limit)
