"""example/macrosynergy/management/simulate/simulate_quantamental_data.py"""
# %% [markdown]
# ## Imports
# %%
from macrosynergy.management.simulate import make_qdf, simulate_ar
import pandas as pd

# %% [markdown]
# ## Create an auto-correlated data-series as numpy array
# %%
ser_ar = simulate_ar(nobs=100, mean=0, sd_mult=1, ar_coef=0.75)

# %% [markdown]
# ## Create a mock dataframe using `make_qdf`
# %%
# Define the cross-sectional identifiers and categories
cids = ["AUD", "CAD", "GBP"]
xcats = ["XR", "CRY"]

# Define a dataframe with the parameters for the cross-sectional identifiers
# Each row is a cross-sectional identifier

cols = ["earliest", "latest", "mean_add", "sd_mult"]
df_cids = pd.DataFrame(index=cids, columns=cols)

# Populate the dataframe with desired parameters
df_cids.loc["AUD",] = ["2010-01-01", "2020-12-31", 0.5, 2]
df_cids.loc["CAD",] = ["2011-01-01", "2020-11-30", 0, 1]
df_cids.loc["GBP",] = ["2011-01-01", "2020-11-30", -0.2, 0.5]

# Define a dataframe with the parameters for the categories
# Also defining the auto-correlation coefficient, and a communal background factor coefficient
cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]
df_xcats = pd.DataFrame(index=xcats, columns=cols)

# Populate the dataframe with desired parameters
df_xcats.loc["XR",] = ["2010-01-01", "2020-12-31", 0, 1, 0, 0.3]
df_xcats.loc["CRY",] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]

# Call the `make_qdf` function
dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

# %% [markdown]
# ## Inspect the dataframe
# %%
dfd.head()
# %% [markdown]
# |    | cid   | xcat   | real_date           |     value |
# |---:|:------|:-------|:--------------------|----------:|
# |  0 | AUD   | XR     | 2010-01-01 00:00:00 |  2.468    |
# |  1 | AUD   | XR     | 2010-01-04 00:00:00 | -1.82449  |
# |  2 | AUD   | XR     | 2010-01-05 00:00:00 |  0.680561 |
# |  3 | AUD   | XR     | 2010-01-06 00:00:00 |  1.49684  |
# |  4 | AUD   | XR     | 2010-01-07 00:00:00 |  0.257214 |
