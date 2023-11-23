"""example/macrosynergy/signal/signal_return_relations.py"""

cids = ["AUD", "CAD", "GBP", "NZD"]
xcats = ["XR", "XRH", "CRY", "GROWTH", "INFL"]
df_cids = pd.DataFrame(
    index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
)
df_cids.loc["AUD"] = ["2000-01-01", "2020-12-31", 0, 1]
df_cids.loc["CAD"] = ["2001-01-01", "2020-11-30", 0, 1]
df_cids.loc["GBP"] = ["2002-01-01", "2020-11-30", 0, 2]
df_cids.loc["NZD"] = ["2007-01-01", "2020-09-30", 0.0, 2]

df_xcats = pd.DataFrame(
    index=xcats,
    columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
)
df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
df_xcats.loc["XRH"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
df_xcats.loc["CRY"] = ["2000-01-01", "2020-10-30", 0, 2, 0.95, 1]
df_xcats.loc["GROWTH"] = ["2001-01-01", "2020-10-30", 0, 2, 0.9, 1]
df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 0, 2, 0.8, 0.5]

black = {"AUD": ["2006-01-01", "2015-12-31"], "GBP": ["2012-01-01", "2100-01-01"]}

dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

# Additional signals.
srn = SignalReturnRelations(
    dfd,
    rets="XR",
    sigs="CRY",
    rival_sigs=None,
    sig_neg=True,
    cosp=True,
    freqs="M",
    start="2002-01-01",
)

dfsum = srn.summary_table()
print(dfsum)

r_sigs = ["INFL", "GROWTH"]
srn = SignalReturnRelations(
    dfd,
    rets="XR",
    sigs="CRY",
    rival_sigs=r_sigs,
    sig_neg=True,
    cosp=True,
    freqs="M",
    start="2002-01-01",
)
dfsum = srn.summary_table()
print(dfsum)

df_sigs = srn.signals_table(sigs=["CRY_NEG", "INFL_NEG"])
df_sigs_all = srn.signals_table()
print(df_sigs)
print(df_sigs_all)

srn.accuracy_bars(
    type="signals",
    title="Accuracy measure between target return, XR,"
    " and the respective signals, ['CRY', 'INFL'"
    ", 'GROWTH'].",
)

sr = SignalReturnRelations(
    dfd,
    rets="XR",
    sigs="CRY",
    freqs="M",
    start="2002-01-01",
    agg_sigs="last",
)

srt = sr.single_relation_table()
mrt = sr.multiple_relations_table()
sst = sr.single_statistic_table(stat="accuracy")

print(srt)
print(mrt)
print(sst)

# Basic Signal Returns showing for multiple input values

sr = SignalReturnRelations(
    dfd,
    rets=["XR", "XRH"],
    sigs=["CRY", "INFL", "GROWTH"],
    sig_neg=[False, True],
    cosp=True,
    freqs=["M", "Q"],
    agg_sigs=["last", "mean"],
    blacklist=black,
)

srt = sr.single_relation_table()
mrt = sr.multiple_relations_table()
sst = sr.single_statistic_table(stat="accuracy", show_heatmap=True)

print(srt)
print(mrt)
print(sst)

# Specifying specific arguments for each of the Signal Return Functions

srt = sr.single_relation_table(ret="XR", xcat="CRY", freq="Q", agg_sigs="last")
print(srt)

mrt = sr.multiple_relations_table(
    rets=["XR", "GROWTH"], xcats="INFL", freqs=["M", "Q"], agg_sigs=["last", "mean"]
)
print(mrt)

sst = sr.single_statistic_table(
    stat="accuracy",
    rows=["ret", "xcat", "freq"],
    columns=["agg_sigs"],
)
print(sst)
