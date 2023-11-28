"""example/macrosynergy/panel/view_metrics.py"""

# %% [markdown]
# ## Imports
# %%
from macrosynergy.management.simulate import make_test_df
from macrosynergy.panel.view_metrics import view_metrics
import pandas as pd

# %% [markdown]
# ## Set the currency areas (cross-sectional identifiers) and categories
# %%
test_cids = ["USD", "EUR", "GBP"]
test_xcats = ["FX", "IR"]

# %% [markdown]
# ## Creating the mock data
# %%
df1 = make_test_df(cids=test_cids, xcats=test_xcats, style="sharp-hill")
df2 = make_test_df(cids=test_cids, xcats=test_xcats, style="four-bit-sine")
df3 = make_test_df(cids=test_cids, xcats=test_xcats, style="sine")
df1.rename(columns={"value": "eop_lag"}, inplace=True)
df2.rename(columns={"value": "mop_lag"}, inplace=True)
df3.rename(columns={"value": "grading"}, inplace=True)
mergeon = ["cid", "xcat", "real_date"]
dfx = pd.merge(pd.merge(df1, df2, on=mergeon), df3, on=mergeon)

# %% [markdown]
# ## Example 1 - View the EOP lag for all FX time series in the dataframes
# %%
view_metrics(
    df=dfx,
    xcat="FX",
)


# %% [markdown]
# ## Example 2 - View the EOP lag for all IR time series in the dataframes
# %%
view_metrics(
    df=dfx,
    xcat="IR",
    metric="mop_lag",
)

# %% [markdown]
# ## Example 3 - View the grading for all IR time series in the dataframes
# %%

view_metrics(
    df=dfx,
    xcat="IR",
    metric="grading",
)
