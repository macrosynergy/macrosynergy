"""example/macrosynergy/panel/make_zn_scores.py"""


cids = ["AUD", "CAD", "GBP", "USD", "NZD"]


xcats = ["XR", "CRY", "GROWTH", "INFL"]


df_cids = pd.DataFrame(
    index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
)


df_cids.loc["AUD"] = ["2010-01-01", "2020-12-31", 0.5, 2]


df_cids.loc["CAD"] = ["2006-01-01", "2020-11-30", 0, 1]


df_cids.loc["GBP"] = ["2008-01-01", "2020-11-30", -0.2, 0.5]


df_cids.loc["USD"] = ["2007-01-01", "2020-09-30", -0.2, 0.5]


df_cids.loc["NZD"] = ["2002-01-01", "2020-09-30", -0.1, 2]


df_xcats = pd.DataFrame(
    index=xcats,
    columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
)


df_xcats.loc["XR"] = ["2008-01-01", "2020-12-31", 0, 1, 0, 0.3]


df_xcats.loc["CRY"] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]


df_xcats.loc["GROWTH"] = ["2012-01-01", "2020-10-30", 1, 2, 0.9, 1]


df_xcats.loc["INFL"] = ["2013-01-01", "2020-10-30", 1, 2, 0.8, 0.5]


# Apply a blacklist period from series' start date.


black = {"AUD": ["2010-01-01", "2013-12-31"], "GBP": ["2018-01-01", "2100-01-01"]}


dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)


dfd["grading"] = np.ones(dfd.shape[0])


# Monthly: panel + cross.


dfzm = make_zn_scores(
    dfd,
    xcat="XR",
    sequential=True,
    cids=cids,
    blacklist=black,
    iis=True,
    neutral="mean",
    pan_weight=0.75,
    min_obs=261,
    est_freq="m",
)


print(dfzm)


# Weekly: panel + cross.


dfzw = make_zn_scores(
    dfd,
    xcat="XR",
    sequential=True,
    cids=cids,
    blacklist=black,
    iis=False,
    neutral="mean",
    pan_weight=0.5,
    min_obs=261,
    est_freq="w",
)


# Daily: panel. Neutral and standard deviation will be computed daily.


dfzd = make_zn_scores(
    dfd,
    xcat="XR",
    sequential=True,
    cids=cids,
    blacklist=black,
    iis=True,
    neutral="mean",
    pan_weight=1.0,
    min_obs=261,
    est_freq="d",
)


# Daily: cross.


dfd["ticker"] = dfd["cid"] + "_" + dfd["xcat"]


dfzd = make_zn_scores(
    dfd,
    xcat="XR",
    sequential=True,
    cids=cids,
    blacklist=black,
    iis=True,
    neutral="mean",
    pan_weight=0.0,
    min_obs=261,
    est_freq="d",
)


panel_df = make_zn_scores(
    dfd,
    "CRY",
    cids,
    start="2010-01-04",
    blacklist=black,
    sequential=False,
    min_obs=0,
    neutral="mean",
    iis=True,
    thresh=None,
    pan_weight=0.75,
    postfix="ZN",
)
