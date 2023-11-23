"""example/macrosynergy/management/simulate/simulate_quantamental_data.py"""

ser_ar = simulate_ar(100, mean=0, sd_mult=1, ar_coef=0.75)

cids = ["AUD", "CAD", "GBP"]
xcats = ["XR", "CRY"]
df_cids = pd.DataFrame(
    index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
)
df_cids.loc["AUD",] = ["2010-01-01", "2020-12-31", 0.5, 2]
df_cids.loc["CAD",] = ["2011-01-01", "2020-11-30", 0, 1]
df_cids.loc["GBP",] = ["2011-01-01", "2020-11-30", -0.2, 0.5]

df_xcats = pd.DataFrame(
    index=xcats,
    columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
)
df_xcats.loc["XR",] = ["2010-01-01", "2020-12-31", 0, 1, 0, 0.3]
df_xcats.loc["CRY",] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]

dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
