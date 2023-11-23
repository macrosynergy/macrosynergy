"""example/macrosynergy/panel/view_ranges.py"""


cids = ["AUD", "CAD", "GBP", "USD"]


xcats = ["XR", "CRY"]


df_cids = pd.DataFrame(
    index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
)


df_cids.loc["AUD",] = ["2010-01-01", "2020-12-31", 0.5, 0.2]


df_cids.loc["CAD",] = ["2011-01-01", "2020-11-30", 0, 1]


df_cids.loc["GBP",] = ["2012-01-01", "2020-11-30", 0, 2]


df_cids.loc["USD",] = ["2012-01-01", "2020-11-30", 1, 2]


df_xcats = pd.DataFrame(
    index=xcats,
    columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
)


df_xcats.loc["XR",] = ["2010-01-01", "2020-12-31", 0, 1, 0, 0.3]


df_xcats.loc["CRY",] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]


dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)


view_ranges(
    dfd,
    xcats=["XR"],
    kind="box",
    start="2012-01-01",
    end="2018-01-01",
    sort_cids_by="std",
)


filter_1 = (dfd["xcat"] == "XR") & (dfd["cid"] == "AUD")


dfd = dfd[~filter_1]


view_ranges(
    dfd,
    xcats=["XR", "CRY"],
    cids=cids,
    kind="box",
    start="2012-01-01",
    end="2018-01-01",
    sort_cids_by=None,
    xcat_labels=["EQXR_NSA", "CRY_NSA"],
)
