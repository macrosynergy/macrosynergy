"""example/macrosynergy/panel/category_relations.py"""
from macrosynergy.management.simulate import make_qdf
from macrosynergy.panel.category_relations import CategoryRelations

import pandas as pd


cids = ["AUD", "CAD", "GBP", "NZD", "USD"]
xcats = ["XR", "CRY", "GROWTH", "INFL"]
df_cids = pd.DataFrame(
    index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
)
df_cids.loc["AUD"] = ["2000-01-01", "2020-12-31", 0.1, 1]
df_cids.loc["CAD"] = ["2001-01-01", "2020-11-30", 0, 1]
df_cids.loc["BRL"] = ["2001-01-01", "2020-11-30", -0.1, 2]
df_cids.loc["GBP"] = ["2002-01-01", "2020-11-30", 0, 2]
df_cids.loc["NZD"] = ["2002-01-01", "2020-09-30", -0.1, 2]
df_cids.loc["USD"] = ["2003-01-01", "2020-12-31", -0.1, 2]

cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]
df_xcats = pd.DataFrame(index=xcats, columns=cols)
df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
df_xcats.loc["CRY"] = ["2000-01-01", "2020-10-30", 1, 2, 0.95, 1]
df_xcats.loc["GROWTH"] = ["2001-01-01", "2020-10-30", 1, 2, 0.9, 1]
df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]

dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
dfd["grading"] = 1
black = {"AUD": ["2000-01-01", "2003-12-31"], "GBP": ["2018-01-01", "2100-01-01"]}

# All AUD GROWTH locations.
filt1 = (dfd["xcat"] == "GROWTH") & (dfd["cid"] == "AUD")
filt2 = (dfd["xcat"] == "INFL") & (dfd["cid"] == "NZD")

# Reduced DataFrame.
dfdx = dfd[~(filt1 | filt2)].copy()
dfdx["ERA"]: str = "before 2007"
dfdx.loc[dfdx["real_date"].dt.year > 2007, "ERA"] = "from 2010"

cidx = ["AUD", "CAD", "GBP", "USD"]

cr = CategoryRelations(
    dfdx,
    xcats=["CRY", "XR"],
    freq="M",
    lag=1,
    cids=cidx,
    xcat_aggs=["mean", "sum"],
    start="2001-01-01",
    blacklist=black,
    years=None,
)

cr.reg_scatter(
    labels=False,
    separator=None,
    title="Carry and Return",
    xlab="Carry",
    ylab="Return",
    coef_box="lower left",
    prob_est="map",
)

cr = CategoryRelations(
    dfdx,
    xcats=["CRY", "XR"],
    freq="M",
    years=5,
    lag=0,
    cids=cidx,
    xcat_aggs=["mean", "sum"],
    start="2001-01-01",
    blacklist=black,
)

cr.reg_scatter(
    labels=False,
    separator=None,
    title="Carry and Return, 5-year periods",
    xlab="Carry",
    ylab="Return",
    coef_box="lower left",
    prob_est="map",
)

cr = CategoryRelations(
    dfdx,
    xcats=["CRY", "XR"],
    xcat1_chg="diff",
    freq="M",
    lag=1,
    cids=cidx,
    xcat_aggs=["mean", "sum"],
    start="2001-01-01",
    blacklist=black,
    years=None,
)

cr.reg_scatter(
    labels=False,
    separator=cids,
    title="Carry and Return",
    xlab="Carry",
    ylab="Return",
    coef_box="lower left",
)

cr.ols_table(type="pool")
cr.ols_table(type="re")
