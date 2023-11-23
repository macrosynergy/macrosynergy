"""example/macrosynergy/panel/granger_causality_test.py"""

cids: List[str] = ["AUD"]
xcats: List[str] = ["FX", "EQ"]

df: pd.DataFrame = make_test_df(
    cids=cids,
    xcats=xcats,
)

gct: Dict[Any, Any] = granger_causality_test(
    df=df,
    cids=cids,
    xcats=xcats,
)

cids: List[str] = ["AUD", "CAD"]
xcats: str = "FX"
# tickers =  AUD_FX, CAD_FX
df: pd.DataFrame = make_test_df(
    cids=cids,
    xcats=xcats,
)

gct: Dict[Any, Any] = granger_causality_test(
    df=df,
    tickers=["AUD_FX", "CAD_FX"],
)

print(gct)
