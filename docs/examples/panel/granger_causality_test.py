"""example/macrosynergy/panel/granger_causality_test.py"""


import pandas as pd
from macrosynergy.management.simulate import make_test_df
from macrosynergy.panel.granger_causality_test import granger_causality_test

## Set the currency areas (cross-sectional identifiers) and categories
cids = ["AUD", "CAD"]
xcats = ["FX", "EQ"]

## Creating the mock data
df = make_test_df(
    cids=cids,
    xcats=xcats,
)

## Example 1
# Run the test if AUD_FX Granger causes AUD_EQ
test_cid = "AUD"
test_xcats = ["FX", "EQ"]

gct = granger_causality_test(
    df=df,
    cids=test_cid,
    xcats=test_xcats,
)

print(gct)

## Example 2
# Run the test if AUD_FX Granger causes CAD_EQ
# tickerA = "AUD_FX"
# tickerB = "CAD_FX"

gct = granger_causality_test(
    df=df,
    tickers=["AUD_FX", "CAD_FX"],
)

print(gct)
