"""example/macrosynergy/panel/view_grades.py"""


# %% [markdown]
# ## Imports
# %%
from macrosynergy.management.simulate import make_qdf
from macrosynergy.panel.view_grades import heatmap_grades
import pandas as pd

# %% [markdown]
# ## Set the currency areas (cross-sectional identifiers) and categories
# %%
cids = ["NZD", "AUD", "CAD", "GBP"]
xcats = ["XR", "CRY", "GROWTH", "INFL"]

# %% [markdown]
# ## Creating the mock data
# %%
cols = ["earliest", "latest", "mean_add", "sd_mult"]
df_cids = pd.DataFrame(index=cids, columns=cols)

df_cids.loc["AUD",] = ["2000-01-01", "2020-12-31", 0.1, 1]
df_cids.loc["CAD",] = ["2001-01-01", "2020-11-30", 0, 1]
df_cids.loc["GBP",] = ["2002-01-01", "2020-11-30", 0, 2]
df_cids.loc["NZD",] = ["2002-01-01", "2020-09-30", -0.1, 2]

cols += ["ar_coef", "back_coef"]
df_xcats = pd.DataFrame(index=xcats, columns=cols)

df_xcats.loc["XR",] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
df_xcats.loc["CRY",] = ["2000-01-01", "2020-10-30", 1, 2, 0.95, 1]
df_xcats.loc["GROWTH",] = ["2001-01-01", "2020-10-30", 1, 2, 0.9, 1]
df_xcats.loc["INFL",] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]

df = make_qdf(df_cids, df_xcats, back_ar=0.75)

df["grading"] = "3"

filter_date = df["real_date"] >= pd.to_datetime("2010-01-01")
filter_cid = df["cid"].isin(["NZD", "AUD"])
df.loc[filter_date & filter_cid, "grading"] = "1"
filter_date = df["real_date"] >= pd.to_datetime("2013-01-01")
filter_xcat = df["xcat"].isin(["CRY", "GROWTH"])
df.loc[filter_date & filter_xcat, "grading"] = "2.1"
filter_xcat = df["xcat"] == "XR"
df.loc[filter_xcat, "grading"] = 1

# %% [markdown]
# ## Example 1 - View the grading of the specified CIDs and XCATs as a heatmap
# %%
heatmap_grades(df, xcats=["CRY", "GROWTH", "INFL"], cids=cids)

df.info()
