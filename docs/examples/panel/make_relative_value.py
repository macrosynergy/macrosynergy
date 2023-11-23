"""example/macrosynergy/panel/make_relative_value.py"""


# Simulate DataFrame.


cids = ["AUD", "CAD", "GBP", "NZD"]


xcats = ["XR", "CRY", "GROWTH", "INFL"]


df_cids = pd.DataFrame(
    index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
)


df_cids.loc["AUD"] = ["2000-01-01", "2020-12-31", 0.1, 1]


df_cids.loc["CAD"] = ["2001-01-01", "2020-11-30", 0, 1]


df_cids.loc["GBP"] = ["2002-01-01", "2020-11-30", 0, 2]


df_cids.loc["NZD"] = ["2002-01-01", "2020-09-30", -0.1, 2]


df_xcats = pd.DataFrame(
    index=xcats,
    columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
)


df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]


df_xcats.loc["CRY"] = ["2000-01-01", "2020-10-30", 1, 2, 0.95, 1]


df_xcats.loc["GROWTH"] = ["2001-01-01", "2020-10-30", 1, 2, 0.9, 1]


df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]


dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)


# Simulate blacklist


black = {"AUD": ["2000-01-01", "2003-12-31"], "GBP": ["2018-01-01", "2100-01-01"]}


# Applications


dfd_1 = make_relative_value(
    dfd,
    xcats=["GROWTH", "INFL"],
    cids=None,
    blacklist=None,
    rel_meth="subtract",
    rel_xcats=None,
    postfix="RV",
)


rel_xcats = ["GROWTH_sRV", "INFL_sRV"]


dfd_1_black = make_relative_value(
    dfd,
    xcats=["GROWTH", "INFL"],
    cids=None,
    blacklist=black,
    rel_meth="subtract",
    rel_xcats=rel_xcats,
    postfix="RV",
)
