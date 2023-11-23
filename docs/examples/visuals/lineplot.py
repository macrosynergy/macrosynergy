"""example/macrosynergy/visuals/lineplot.py"""


cids: List[str] = [
    "USD",
    "EUR",
    "GBP",
    "AUD",
    "CAD",
    "JPY",
    "CHF",
    "NZD",
    "SEK",
    "INR",
]
# Quantamental categories of interest

xcats = [
    "NIR_NSA",
    "RIR_NSA",
    "DU05YXR_NSA",
    "DU05YXR_VT10",
    "FXXR_NSA",
    "EQXR_NSA",
    "DU05YXR_NSA",
]  # market links

sel_cids: List[str] = ["USD", "EUR", "GBP"]
sel_xcats: List[str] = ["NIR_NSA", "RIR_NSA", "FXXR_NSA", "EQXR_NSA"]

with JPMaQSDownload(
    local_path=r"~\Macrosynergy\Macrosynergy - Documents\SharedData\JPMaQSTickers"
) as jpmaqs:
    df: pd.DataFrame = jpmaqs.download(
        cids=cids,
        xcats=xcats,
        start_date="2016-01-01",
    )


random = SystemRandom()

# random.seed(42)

# for cidx, xcatx in df[["cid", "xcat"]].drop_duplicates().values.tolist():
#     # if random() > 0.5 multiply by random.random()*10
#     _bools = (df["cid"] == cidx) & (df["xcat"] == xcatx)
#     r = max(random.random(), 0.1)
#     df.loc[_bools, "value"] = df.loc[_bools, "value"] * r

# FacetPlot(df).lineplot()

print("From same object:")
timer_start: float = time.time()

LinePlot(df, cids=cids, xcats=xcats).plot(
    title="Test Title with a very long title to see how it looks, \n and a new line - why not?",
    legend_fontsize=8,
    compare_series="USD_RIR_NSA",
)

# facet_size=(5, 4),
print(f"Time taken: {time.time() - timer_start}")
