"""example/macrosynergy/pnl/naive_pnl.py"""


cids = ["AUD", "CAD", "GBP", "NZD", "USD", "EUR"]


xcats = ["EQXR_NSA", "CRY", "GROWTH", "INFL", "DUXR"]


cols_1 = ["earliest", "latest", "mean_add", "sd_mult"]


df_cids = pd.DataFrame(index=cids, columns=cols_1)


df_cids.loc["AUD", :] = ["2008-01-03", "2020-12-31", 0.5, 2]


df_cids.loc["CAD", :] = ["2010-01-03", "2020-11-30", 0, 1]


df_cids.loc["GBP", :] = ["2012-01-03", "2020-11-30", -0.2, 0.5]


df_cids.loc["NZD"] = ["2002-01-03", "2020-09-30", -0.1, 2]


df_cids.loc["USD"] = ["2015-01-03", "2020-12-31", 0.2, 2]


df_cids.loc["EUR"] = ["2008-01-03", "2020-12-31", 0.1, 2]


cols_2 = cols_1 + ["ar_coef", "back_coef"]


df_xcats = pd.DataFrame(index=xcats, columns=cols_2)


df_xcats.loc["EQXR_NSA"] = ["2000-01-03", "2020-12-31", 0.1, 1, 0, 0.3]


df_xcats.loc["CRY"] = ["2000-01-01", "2020-10-30", 1, 2, 0.95, 1]


df_xcats.loc["GROWTH"] = ["2010-01-03", "2020-10-30", 1, 2, 0.9, 1]


df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]


df_xcats.loc["DUXR"] = ["2000-01-01", "2020-12-31", 0.1, 0.5, 0, 0.1]


black = {"AUD": ["2006-01-01", "2015-12-31"], "GBP": ["2022-01-01", "2100-01-01"]}


dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)


# Instantiate a new instance to test the long-only functionality.


# Benchmarks are used to calculate correlation against PnL series.


pnl = NaivePnL(
    dfd,
    ret="EQXR_NSA",
    sigs=["CRY", "GROWTH", "INFL"],
    cids=cids,
    start="2000-01-01",
    blacklist=black,
    bms=["EUR_EQXR_NSA", "USD_EQXR_NSA"],
)


pnl.make_pnl(
    sig="GROWTH",
    sig_op="zn_score_pan",
    sig_neg=True,
    sig_add=0.5,
    rebal_freq="monthly",
    vol_scale=5,
    rebal_slip=1,
    min_obs=250,
    thresh=2,
)


pnl.make_long_pnl(vol_scale=10, label="Long")


df_eval = pnl.evaluate_pnls(
    pnl_cats=["PNL_GROWTH_NEG"], start="2015-01-01", end="2020-12-31"
)


pnl.agg_signal_bars(
    pnl_name="PNL_GROWTH_NEG",
    freq="m",
    metric="direction",
    title=None,
)


pnl.plot_pnls(
    pnl_cats=["PNL_GROWTH_NEG", "Long"],
    facet=False,
    xlab="date",
    ylab="%",
)


pnl.plot_pnls(
    pnl_cats=["PNL_GROWTH_NEG", "Long"],
    facet=False,
    xcat_labels=["S_1", "S_2"],
    xlab="date",
    ylab="%",
)


pnl.plot_pnls(
    pnl_cats=["PNL_GROWTH_NEG", "Long"], facet=True, xcat_labels=["S_1", "S_2"]
)


pnl.plot_pnls(
    pnl_cats=["PNL_GROWTH_NEG", "Long"],
    facet=True,
)


pnl.plot_pnls(pnl_cats=["PNL_GROWTH_NEG"], pnl_cids=cids, xcat_labels=None)


pnl.plot_pnls(pnl_cats=["PNL_GROWTH_NEG"], pnl_cids=cids, facet=True, xcat_labels=None)


pnl.plot_pnls(
    pnl_cats=["PNL_GROWTH_NEG"],
    pnl_cids=cids,
    same_y=True,
    facet=True,
    xcat_labels=None,
    share_axis_labels=False,
    xlab="Date",
    ylab="PnL",
    y_label_adj=0.1,
)
