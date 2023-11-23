"""example/macrosynergy/panel/view_correlations.py"""


np.random.seed(0)


# Un-clustered correlation matrices.


cids = ["AUD", "CAD", "GBP", "USD", "NZD", "EUR"]


cids_dmsc = ["CHF", "NOK", "SEK"]


cids_dmec = ["DEM", "ESP", "FRF", "ITL", "NLG"]


cids += cids_dmec


cids += cids_dmsc


xcats = ["XR", "CRY"]


df_cids = pd.DataFrame(
    index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
)


df_cids.loc["AUD"] = ["2010-01-01", "2020-12-31", 0.5, 2]


df_cids.loc["CAD"] = ["2011-01-01", "2020-11-30", 0, 1]


df_cids.loc["GBP"] = ["2012-01-01", "2020-11-30", -0.2, 0.5]


df_cids.loc["USD"] = ["2010-01-01", "2020-12-30", -0.2, 0.5]


df_cids.loc["NZD"] = ["2002-01-01", "2020-09-30", -0.1, 2]


df_cids.loc["EUR"] = ["2002-01-01", "2020-09-30", -0.2, 2]


df_cids.loc["DEM"] = ["2003-01-01", "2020-09-30", -0.3, 2]


df_cids.loc["ESP"] = ["2003-01-01", "2020-09-30", -0.1, 2]


df_cids.loc["FRF"] = ["2003-01-01", "2020-09-30", -0.2, 2]


df_cids.loc["ITL"] = ["2004-01-01", "2020-09-30", -0.2, 0.5]


df_cids.loc["NLG"] = ["2003-01-01", "2020-12-30", -0.1, 0.5]


df_cids.loc["CHF"] = ["2003-01-01", "2020-12-30", -0.3, 2.5]


df_cids.loc["NOK"] = ["2010-01-01", "2020-12-30", -0.1, 0.5]


df_cids.loc["SEK"] = ["2010-01-01", "2020-09-30", -0.1, 0.5]


df_xcats = pd.DataFrame(
    index=xcats,
    columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
)


df_xcats.loc["XR",] = ["2010-01-01", "2020-12-31", 0.1, 1, 0, 0.3]


df_xcats.loc["CRY",] = ["2010-01-01", "2020-10-30", 1, 2, 0.95, 0.5]


dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)


start = "2012-01-01"


end = "2020-09-30"


lag_dict = {"XR": [0, 2, 5]}


# Clustered correlation matrices. Test hierarchical clustering.


correl_matrix(
    df=dfd,
    xcats=["XR"],
    xcats_secondary=None,
    cids=cids,
    cids_secondary=None,
    start=start,
    end=end,
    val="value",
    freq=None,
    cluster=True,
    title="Correlation Matrix",
    size=(14, 8),
    max_color=None,
    lags=None,
    lags_secondary=None,
)
