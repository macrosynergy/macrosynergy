"""example/macrosynergy/visuals/facetplot.py"""

# from macrosynergy.visuals import FacetPlot


cids_A: List[str] = ["AUD", "CAD", "EUR", "GBP", "USD"]
cids_B: List[str] = ["CHF", "INR", "JPY", "NOK", "NZD", "SEK"]
cids_C: List[str] = ["CHF", "EUR", "INR", "JPY", "NOK", "NZD", "SEK", "USD"]

xcats_A: List[str] = [
    "CPIXFE_SA_P1M1ML12",
    "CPIXFE_SJA_P3M3ML3AR",
    "CPIXFE_SJA_P6M6ML6AR",
    "CPIXFE_SA_P1M1ML12_D1M1ML3",
    "CPIXFE_SA_P1M1ML12_D1M1ML3",
]
xcats_B: List[str] = [
    "CPIC_SA_P1M1ML12",
    "CPIC_SJA_P3M3ML3AR",
    "CPIC_SJA_P6M6ML6AR",
    "CPIC_SA_P1M1ML12_D1M1ML3",
    "CPIH_SA_P1M1ML12",
    "EXALLOPENNESS_NSA_1YMA",
    "EXMOPENNESS_NSA_1YMA",
]
xcats_C: List[str] = ["DU05YXR_NSA", "DU05YXR_VT10"]
xcats_D: List[str] = [
    "FXXR_NSA",
    "EQXR_NSA",
    "FXTARGETED_NSA",
    "FXUNTRADABLE_NSA",
]
all_cids: List[str] = list(set(cids_A + cids_B + cids_C))
all_xcats: List[str] = list(set(xcats_A + xcats_B + xcats_C + xcats_D))

df: pd.DataFrame = make_test_df(
    cids=all_cids,
    xcats=all_xcats,
)
# remove data for USD_FXXR_NSA and CHF _EQXR_NSA and _FXXR_NSA
df: pd.DataFrame = df[
    ~((df["cid"] == "USD") & (df["xcat"] == "FXXR_NSA"))
].reset_index(drop=True)
df: pd.DataFrame = df[
    ~((df["cid"] == "CHF") & (df["xcat"].isin(["EQXR_NSA", "FXXR_NSA"])))
].reset_index(drop=True)
df: pd.DataFrame = df[
    ~((df["cid"] == "NOK") & (df["xcat"] == "FXUNTRADABLE_NSA"))
].reset_index(drop=True)

timer_start: float = time.time()

with FacetPlot(
    df,
) as fp:
    fp.lineplot(
        cids=cids_A,
        share_x=True,
        xcat_grid=True,
        ncols=2,
        title="Test Title with a very long title to see how it looks, \n and a new line - why not?",
        # save_to_file="test_0.png",
        ax_hline=75,
        show=True,
    )
    fp.lineplot(
        cids=cids_B,
        xcats=xcats_A,
        attempt_square=True,
        share_y=True,
        cid_grid=True,
        title="Another test title",
        # save_to_file="test_1.png",
        show=True,
    )
    fp.lineplot(
        cids=cids_C,
        xcats=xcats_D,
        cid_xcat_grid=True,
        title="Another test title",
        show=True,
    )

print(f"Time taken: {time.time() - timer_start}")
