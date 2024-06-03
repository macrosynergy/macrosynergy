"""example/macrosynergy/signal/target_positions.py"""


# A. Example dataframe


cids = ["AUD", "GBP", "NZD", "USD"]


xcats = ["FXXR_NSA", "EQXR_NSA", "SIG_NSA"]


ccols = ["earliest", "latest", "mean_add", "sd_mult"]


df_cids = pd.DataFrame(index=cids, columns=ccols)


df_cids.loc["AUD"] = ["2010-01-01", "2020-12-31", 0, 1]


df_cids.loc["GBP"] = ["2010-01-01", "2020-12-31", 0, 2]


df_cids.loc["NZD"] = ["2010-01-01", "2020-12-31", 0, 3]


df_cids.loc["USD"] = ["2010-01-01", "2020-12-31", 0, 4]


xcols = ccols + ["ar_coef", "back_coef"]


df_xcats = pd.DataFrame(index=xcats, columns=xcols)


df_xcats.loc["FXXR_NSA"] = ["2010-01-01", "2020-12-31", 0, 1, 0, 0.2]


df_xcats.loc["EQXR_NSA"] = ["2010-01-01", "2020-12-31", 0.5, 2, 0, 0.2]


df_xcats.loc["SIG_NSA"] = ["2010-01-01", "2020-12-31", 0, 10, 0.4, 0.2]


random.seed(2)


dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)


dfd_copy = dfd.copy()


black = {"AUD": ["2000-01-01", "2003-12-31"], "GBP": ["2018-01-01", "2100-01-01"]}


# B. Target positions without basket


df1 = target_positions(
    df=dfd,
    cids=cids,
    xcat_sig="SIG_NSA",
    ctypes=["FX", "EQ"],
    sigrels=[1, 0.5],
    ret="XR_NSA",
    start="2012-01-01",
    end="2020-10-30",
    scale="prop",
    min_obs=252,
    cs_vtarg=5,
    posname="POS",
)


df2 = target_positions(
    df=dfd,
    cids=cids,
    xcat_sig="FXXR_NSA",
    ctypes=["FX", "EQ"],
    sigrels=[1, -1],
    ret="XR_NSA",
    start="2012-01-01",
    end="2020-10-30",
    scale="dig",
    cs_vtarg=0.1,
    posname="POS",
)


df3 = target_positions(
    df=dfd,
    cids=cids,
    xcat_sig="FXXR_NSA",
    ctypes=["FX", "EQ"],
    sigrels=[1, -1],
    ret="XR_NSA",
    start="2010-01-01",
    end="2020-12-31",
    scale="prop",
    cs_vtarg=None,
    posname="POS",
)


# C. Target position with one basket


apc_contracts = ["AUD_FX", "NZD_FX"]


basket_1 = Basket(
    df=dfd, contracts=apc_contracts, ret="XR_NSA", cry=None, blacklist=black
)


basket_1.make_basket(weight_meth="equal", max_weight=0.55, basket_name="APC_FX")


df_weight = basket_1.return_weights("APC_FX")


df_weight = df_weight[["cid", "xcat", "real_date", "value"]]


dfd = dfd[["cid", "xcat", "real_date", "value"]]


dfd_concat = pd.concat([dfd_copy, df_weight])


df4 = target_positions(
    df=dfd_concat,
    cids=cids,
    xcat_sig="SIG_NSA",
    ctypes=["FX", "EQ"],
    basket_names=["APC_FX"],
    sigrels=[1, -1, -0.5],
    ret="XR_NSA",
    start="2010-01-01",
    end="2020-12-31",
    scale="prop",
    cs_vtarg=10,
    posname="POS",
)
