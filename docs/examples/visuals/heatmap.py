"""example/macrosynergy/visuals/heatmap.py"""

test_cids: List[str] = [
    "USD",
]  # "EUR", "GBP"]
test_xcats: List[str] = ["FX", "IR"]
dfE: pd.DataFrame = make_test_df(
    cids=test_cids, xcats=test_xcats, style="sharp-hill"
)

dfM: pd.DataFrame = make_test_df(
    cids=test_cids, xcats=test_xcats, style="four-bit-sine"
)

dfG: pd.DataFrame = make_test_df(cids=test_cids, xcats=test_xcats, style="sine")

dfE.rename(columns={"value": "eop_lag"}, inplace=True)
dfM.rename(columns={"value": "mop_lag"}, inplace=True)
dfG.rename(columns={"value": "grading"}, inplace=True)
mergeon = ["cid", "xcat", "real_date"]
dfx: pd.DataFrame = pd.merge(pd.merge(dfE, dfM, on=mergeon), dfG, on=mergeon)

heatmap = Heatmap(df=dfx, xcats=["FX"])

heatmap.df["real_date"]: pd.Series = heatmap.df["real_date"].dt.strftime("%Y-%m-%d")
heatmap.df = heatmap.df.pivot_table(
    index="cid", columns="real_date", values="grading"
)

heatmap._plot(heatmap.df, title="abc", rotate_xticks=90)
