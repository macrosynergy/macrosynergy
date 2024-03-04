"""example/macrosynergy/panel/linear_composite.py"""
import pandas as pd
from macrosynergy.management.simulate import make_test_df
from macrosynergy.panel.linear_composite import linear_composite

# %% [markdown]
# ## Set the currency areas (cross-sectional identifiers) and categories
# %%
cids = ["AUD", "CAD", "GBP"]
xcats = ["XR", "CRY", "INFL"]

# %% [markdown]
# ## Creating the mock data
# %%
df1 = make_test_df(
    cids=cids,
    xcats=["XR", "CRY"],
    start="2000-01-01",
    end="2000-02-01",
    style="linear",
)
df2 = make_test_df(
    cids=cids,
    xcats=["INFL"],
    start="2000-01-01",
    end="2000-02-01",
    style="decreasing-linear",
)

df = pd.concat([df1, df2], axis=0)

missing_date = df["real_date"] == "2000-01-17"

df.loc[(df["cid"] == "GBP") & (df["xcat"] == "INFL") & missing_date, "value"] = pd.NA
df.loc[(df["cid"] == "AUD") & (df["xcat"] == "CRY") & missing_date, "value"] = pd.NA

# %% [markdown]
# For this example:
# - All *_INFL are decreasing-linear (downward sloping)
# - All *_XR and *_CRY are increasing-linear (upward sloping)
# - AUD_CRY and GBP_INFL are missing data on 2000-01-17

# %% [markdown]
# ## Example 1 - Calculate the linear composite by adding the same category across all currencies
# %%
# In this example, we add the XR category across all currencies.
# This effectively creates a "new" currency area, which we call "GLOBAL".
lc_cid = linear_composite(
    df=df,
    xcats="XR",
    weights="INFL",
    new_cid="GLOBAL",
)
# %% [markdown]
# ## Example 2 - Calculate the linear composite by a set of categories for each currency
# %%
# In this example, we add the XR, CRY and INFL categories for each currency (AUD and CAD).
# This effectively creates a "new" category for each currency, which we call "NEWXCAT".
lc_xcat = linear_composite(
    df=df,
    cids=["AUD", "CAD"],
    xcats=["XR", "CRY", "INFL"],
    weights=[1, 2, 1],
    signs=[1, -1, 1],
    new_xcat="NEWXCAT",
)
