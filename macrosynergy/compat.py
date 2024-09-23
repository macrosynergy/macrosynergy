from . import PYTHON_3_8_OR_LATER
import pandas as pd
from packaging import version

if PYTHON_3_8_OR_LATER:
    RESAMPLE_NUMERIC_ONLY = {"numeric_only": True}
else:
    RESAMPLE_NUMERIC_ONLY = {}


PD_FUTURE_STACK = (
    dict(future_stack=True)
    if version.parse(pd.__version__) > version.parse("2.1.0")
    else dict(dropna=False)
)

PD_NEW_DATE_FREQ = version.parse(pd.__version__) > version.parse("2.1.4")
