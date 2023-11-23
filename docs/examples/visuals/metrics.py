"""example/macrosynergy/visuals/metrics.py"""


test_cids = ["USD", "EUR", "GBP"]


test_xcats = ["FX", "IR"]


dfE = make_test_df(cids=test_cids, xcats=test_xcats, style="sharp-hill")


dfM = make_test_df(cids=test_cids, xcats=test_xcats, style="four-bit-sine")


dfG = make_test_df(cids=test_cids, xcats=test_xcats, style="sine")


dfE.rename(columns={"value": "eop_lag"}, inplace=True)


dfM.rename(columns={"value": "mop_lag"}, inplace=True)


dfG.rename(columns={"value": "grading"}, inplace=True)


mergeon = ["cid", "xcat", "real_date"]


dfx = pd.merge(pd.merge(dfE, dfM, on=mergeon), dfG, on=mergeon)


view_metrics(
    df=dfx,
    xcat="FX",
)


view_metrics(
    df=dfx,
    xcat="IR",
    metric="mop_lag",
)


view_metrics(
    df=dfx,
    xcat="IR",
    metric="grading",
)
