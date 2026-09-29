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

# `select_dtypes` rebuilds via `type(self)(mgr)` before pandas 2.2, breaking subclasses
PD_SUBCLASS_SAFE_SELECT_DTYPES: bool = hasattr(pd.DataFrame, "_constructor_from_mgr")

# Polars' `LazyFrame.pivot` is not available in the last Python3.8 version (polars==1.8.2)
PYTHON_3_8_POLARS_PIVOT: bool = not hasattr(pl.LazyFrame, "pivot")

# `scan_parquet(include_file_paths=...)` arrived in polars 1.2.0
POLARS_SCAN_FILE_PATHS: bool = version.parse(pl.__version__) >= version.parse("1.2.0")

# polars' parquet reader keeps memory across the files it reads in these releases,
# whatever the engine or batching: 1.0-1.1 leak, 1.15-1.27 creep (measured on JPMaQS
# delta histories). Long multi-file loads can run out of memory there
POLARS_PARQUET_READER_RETAINS_MEMORY: bool = version.parse(
    pl.__version__
) < version.parse("1.2.0") or (
    version.parse("1.15.0") <= version.parse(pl.__version__) < version.parse("1.28.0")
)
