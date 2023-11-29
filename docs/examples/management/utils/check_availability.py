"""example/macrosynergy/management/utils/check_availability.py"""

# %% [markdown]
# ## Imports
# %%
from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import check_availability
import pandas as pd

# %% [markdown]
# ## Set the currency areas (cross-sectional identifiers) and categories
# %%
cids = ["AUD", "CAD", "GBP"]
xcats = ["XR", "CRY"]

# %% [markdown]
# ## Creating the mock data
# %%
cols = ["earliest", "latest", "mean_add", "sd_mult"]

df_cids = pd.DataFrame(index=cids, columns=cols)
df_cids.loc["AUD",] = ["2010-01-01", "2020-12-31", 0.5, 2]
df_cids.loc["CAD",] = ["2010-01-01", "2020-11-30", 0, 1]
df_cids.loc["GBP",] = ["2012-01-01", "2020-11-30", -0.2, 0.5]

cols += ["ar_coef", "back_coef"]

df_xcats = pd.DataFrame(index=xcats, columns=cols)
df_xcats.loc["XR",] = ["2010-01-01", "2020-12-31", 0, 1, 0, 0.3]
df_xcats.loc["CRY",] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]

dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

filt_na = (dfd["cid"] == "CAD") & (dfd["real_date"] < "2011-01-01")
dfd.loc[filt_na, "value"] = pd.NA

# %% [markdown]
# ## Checking availability of data
# %%
check_availability(
    df=dfd,
    xcats=xcats,
    cids=cids,
    start_size=(10, 5),
    end_size=(10, 8),
)
