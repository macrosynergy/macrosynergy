"""example/macrosynergy/panel/view_ranges.py"""
# %% [markdown]
# ## Imports
# %%
from macrosynergy.management.simulate import make_qdf
from macrosynergy.panel.view_ranges import view_ranges
import pandas as pd

# %% [markdown]
# ## Set the currency areas (cross-sectional identifiers) and categories
# %%
cids = ["AUD", "CAD", "GBP", "USD"]
xcats = ["XR", "CRY"]
# %% [markdown]
# ## Creating the mock data
# %%
cols = ["earliest", "latest", "mean_add", "sd_mult"]
df_cids = pd.DataFrame(index=cids, columns=cols)
df_cids.loc["AUD",] = ["2010-01-01", "2020-12-31", 0.5, 0.2]
df_cids.loc["CAD",] = ["2011-01-01", "2020-11-30", 0, 1]
df_cids.loc["GBP",] = ["2012-01-01", "2020-11-30", 0, 2]
df_cids.loc["USD",] = ["2012-01-01", "2020-11-30", 1, 2]

cols += ["ar_coef", "back_coef"]
df_xcats = pd.DataFrame(index=xcats, columns=cols)

df_xcats.loc["XR",] = ["2010-01-01", "2020-12-31", 0, 1, 0, 0.3]
df_xcats.loc["CRY",] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]

df = make_qdf(df_cids, df_xcats, back_ar=0.75)

# %% [markdown]
# ## Example 1 - `view_ranges` with a single XCAT, boxplot
# %%
view_ranges(
    df=df,
    xcats=["XR"],
    kind="box",
    start="2012-01-01",
    end="2018-01-01",
    sort_cids_by="std",
)


# %% [markdown]
# ## Example 2 - `view_ranges` with multiple XCATs, barplots
# %%
# Filter the df to specific timeseries.
filter_1 = (df["xcat"] == "XR") & (df["cid"] == "AUD")

df = df[~filter_1]


view_ranges(
    df=df,
    xcats=["XR", "CRY"],
    cids=cids,
    kind="box",
    start="2012-01-01",
    end="2018-01-01",
    sort_cids_by=None,
    xcat_labels=["EQXR_NSA", "CRY_NSA"],
)
