"""example/macrosynergy/panel/granger_causality_test.py"""

import pandas as pd

from macrosynergy.management.simulate import make_test_df
from macrosynergy.panel.granger_causality_test import granger_causality_test

cids = ["AUD"]
xcats = ["FX", "EQ"]

df = make_test_df(
    cids=cids,
    xcats=xcats,
)

gct = granger_causality_test(
    df=df,
    cids=cids,
    xcats=xcats,
)

cids = ["AUD", "CAD"]
xcats = "FX"

df = make_test_df(
    cids=cids,
    xcats=xcats,
)

gct = granger_causality_test(
    df=df,
    tickers=["AUD_FX", "CAD_FX"],
)

print(gct)
