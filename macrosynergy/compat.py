from . import PYTHON_3_8_OR_LATER
import pandas as pd
import polars as pl
from packaging import version

if PYTHON_3_8_OR_LATER:
    RESAMPLE_NUMERIC_ONLY = {"numeric_only": True}
    JOBLIB_RETURN_AS = {"return_as": "generator"}
    from sklearn.base import OneToOneFeatureMixin
else:
    RESAMPLE_NUMERIC_ONLY = {}
    JOBLIB_RETURN_AS = {}
    from sklearn.base import _OneToOneFeatureMixin as OneToOneFeatureMixin


if version.parse(pd.__version__) > version.parse("2.1.0"):
    PD_FUTURE_STACK = {"future_stack": True}
else:
    PD_FUTURE_STACK = {"dropna": False}

PD_NEW_DATE_FREQ: bool = version.parse(pd.__version__) > version.parse("2.1.4")

PD_OLD_RESAMPLE: bool = version.parse(pd.__version__) < version.parse("1.5.0")

PD_2_0_OR_LATER: bool = version.parse(pd.__version__) >= version.parse("2.0.0")

# Availability of pd.DataFrame.applymap/map
# https://pandas.pydata.org/pandas-docs/version/2.1/reference/api/pandas.DataFrame.map.html
PD_NEW_MAP: bool = version.parse(pd.__version__) >= version.parse("2.1.0")

# Polars' `LazyFrame.pivot` is not available in the last Python3.8 version (polars==1.8.2)
PYTHON_3_8_POLARS_PIVOT: bool = not hasattr(pl.LazyFrame, "pivot")
