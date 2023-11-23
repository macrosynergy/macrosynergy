"""example/macrosynergy/visuals/timelines.py"""
from macrosynergy.management.simulate import make_test_df
from macrosynergy.visuals import timelines
import pandas as pd

cids = [
    "USD",
    "EUR",
    "GBP",
    "AUD",
    "CAD",
    "JPY",
    "CHF",
    "NZD",
    "SEK",
    "NOK",
    "DKK",
    "INR",
]


xcats = [
    "FXXR",
    "EQXR",
    "RIR",
    "IR",
    "REER",
    "CPI",
    "PPI",
    "M2",
    "M1",
    "M0",
    "FXVOL",
    "FX",
]


sel_cids = ["USD", "EUR", "GBP"]


sel_xcats = ["FXXR", "EQXR", "RIR", "IR"]


r_styles = [
    "linear",
    "decreasing-linear",
    "sharp-hill",
    "sine",
    "four-bit-sine",
]


df = make_test_df(
    cids=list(set(cids) - set(sel_cids)),
    xcats=xcats,
    start="2000-01-01",
)


for rstyle, xcatx in zip(r_styles, sel_xcats):
    dfB = make_test_df(
        cids=sel_cids,
        xcats=[xcatx],
        start="2000-01-01",
        style=rstyle,
    )

    df = pd.concat([df, dfB], axis=0)


for ix, cidx in enumerate(sel_cids):
    df.loc[df["cid"] == cidx, "value"] = (
        ((df[df["cid"] == cidx]["value"]) * (ix + 1)).reset_index(drop=True).copy()
    )


for ix, xcatx in enumerate(sel_xcats):
    df.loc[df["xcat"] == xcatx, "value"] = (
        ((df[df["xcat"] == xcatx]["value"]) * (ix * 10 + 1))
        .reset_index(drop=True)
        .copy()
    )


# timer_start: float = time.time()


timelines(
    df=df,
    xcats=sel_xcats,
    xcat_grid=True,
    xcat_labels=["ForEx", "Equity", "Real Interest Rates", "Interest Rates"],
    square_grid=True,
    cids=sel_cids[1],
    # single_chart=True,
)


timelines(
    df=df,
    xcats=sel_xcats[0],
    cids=sel_cids,
    # cs_mean=True,
    # xcat_grid=False,
    single_chart=True,
    cs_mean=True,
)


timelines(
    df=df,
    same_y=False,
    xcats=sel_xcats[0],
    cids=sel_cids,
    title="Plotting multiple cross sections for a single category \n with different y-axis!",
)
