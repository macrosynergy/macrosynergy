"""example/macrosynergy/panel/linear_composite.py"""
import pandas as pd
from macrosynergy.management.simulate import make_test_df
from macrosynergy.panel.linear_composite import linear_composite

cids = ["AUD", "CAD", "GBP"]


xcats = ["XR", "CRY", "INFL"]


df = pd.concat(
    [
        make_test_df(
            cids=cids,
            xcats=xcats[:-1],
            start="2000-01-01",
            end="2000-02-01",
            style="linear",
        ),
        make_test_df(
            cids=cids,
            xcats=["INFL"],
            start="2000-01-01",
            end="2000-02-01",
            style="decreasing-linear",
        ),
    ]
)


# all infls are now decreasing-linear, while everything else is increasing-linear


df.loc[
    (df["cid"] == "GBP") & (df["xcat"] == "INFL") & (df["real_date"] == "2000-01-17"),
    "value",
] = pd.NA


df.loc[
    (df["cid"] == "AUD") & (df["xcat"] == "CRY") & (df["real_date"] == "2000-01-17"),
    "value",
] = pd.NA


# there are now missing values for AUD-CRY and GBP-INFL on 2000-01-17


lc_cid = linear_composite(df=df, xcats="XR", weights="INFL", normalize_weights=False)


lc_xcat = linear_composite(
    df=df,
    cids=["AUD", "CAD"],
    xcats=["XR", "CRY", "INFL"],
    weights=[1, 2, 1],
    signs=[1, -1, 1],
)
