"""example/macrosynergy/panel/view_timelines.py"""


# %% [markdown]
# ## Imports
# %%
from macrosynergy.management.simulate import make_qdf
from macrosynergy.panel.view_timelines import view_timelines
import pandas as pd
import numpy as np

# %% [markdown]
# ## Set the currency areas (cross-sectional identifiers) and categories
# %%
cids = ["AUD", "CAD", "GBP", "NZD"]
xcats = ["XR", "CRY", "INFL", "FXXR"]
# %% [markdown]
# ## Creating the mock data
# %%
cols = ["earliest", "latest", "mean_add", "sd_mult"]
df_cids = pd.DataFrame(index=cids, columns=cols)


df_cids.loc["AUD"] = ["2010-01-01", "2020-12-31", 0.2, 0.2]
df_cids.loc["CAD"] = ["2011-01-01", "2020-11-30", 0, 1]
df_cids.loc["GBP"] = ["2012-01-01", "2020-11-30", 0, 2]
df_cids.loc["NZD"] = ["2012-01-01", "2020-09-30", -0.1, 3]

cols += ["ar_coef", "back_coef"]
df_xcats = pd.DataFrame(index=xcats, columns=cols)

df_xcats.loc["XR"] = ["2010-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
df_xcats.loc["INFL"] = ["2015-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
df_xcats.loc["CRY"] = ["2013-01-01", "2020-10-30", 1, 2, 0.95, 0.5]
df_xcats.loc["FXXR"] = ["2013-01-01", "2020-10-30", 1, 2, 0.95, 0.5]

dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

dfd = dfd.sort_values(["cid", "xcat", "real_date"])

ctr = -1
for xcat in xcats[:2]:
    for cid in cids[:2]:
        ctr *= -1
        mask = (dfd["cid"] == cid) & (dfd["xcat"] == xcat)
        dfd.loc[mask, "value"] = (
            10
            * ctr
            * np.arange(dfd.loc[mask, "value"].shape[0])
            / (dfd.loc[mask, "value"].shape[0] - 1)
        )



view_timelines(
    dfd,
    xcats=["XR", "CRY"],
    cids=cids[0],
    size=(10, 5),
    title="AUD Return and Carry",
)


view_timelines(
    dfd,
    xcats=["XR", "CRY", "INFL"],
    cids=cids[0],
    xcat_grid=True,
    title_adj=0.8,
    xcat_labels=["Return", "Carry", "Inflation"],
    title="AUD Return, Carry & Inflation",
)


view_timelines(dfd, xcats=["CRY"], cids=cids, ncol=2, title="Carry", cs_mean=True)


view_timelines(
    dfd, xcats=["XR"], cids=cids[:2], ncol=2, cumsum=True, same_y=False, aspect=2
)


dfd = dfd.set_index("real_date")


view_timelines(
    dfd,
    xcats=["XR"],
    cids=cids[:2],
    ncol=2,
    cumsum=True,
    same_y=False,
    aspect=2,
    single_chart=True,
)
