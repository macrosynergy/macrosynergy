"""example/macrosynergy/panel/basket.py"""

## Imports

from macrosynergy.panel.basket import Basket
from macrosynergy.management.simulate import make_qdf
import pandas as pd
import random

# %% [markdown]
# ## Set the currency areas (cross-sectional identifiers) and categories
# %%
cids = ["AUD", "GBP", "NZD", "USD"]
xcats = [
    "FXXR_NSA",
    "FXCRY_NSA",
    "FXCRR_NSA",
    "EQXR_NSA",
    "EQCRY_NSA",
    "EQCRR_NSA",
    "FXWBASE_NSA",
]
# %% [markdown]
# ## Creating the mock data
# %%
random.seed(42)

df_cids = pd.DataFrame(
    index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
)

df_cids.loc["AUD"] = ["2000-01-01", "2022-03-14", 0, 1]
df_cids.loc["GBP"] = ["2001-01-01", "2022-03-14", 0, 2]
df_cids.loc["NZD"] = ["2002-01-01", "2022-03-14", 0, 3]
df_cids.loc["USD"] = ["2000-01-01", "2022-03-14", 0, 4]

df_xcats = pd.DataFrame(
    index=xcats,
    columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
)

df_xcats.loc["FXXR_NSA"] = ["2010-01-01", "2022-03-14", 0, 1, 0, 0.2]
df_xcats.loc["FXCRY_NSA"] = ["2010-01-01", "2022-03-14", 1, 1, 0.9, 0.2]
df_xcats.loc["FXCRR_NSA"] = ["2010-01-01", "2022-03-14", 0.5, 0.8, 0.9, 0.2]
df_xcats.loc["EQXR_NSA"] = ["2010-01-01", "2022-03-14", 0.5, 2, 0, 0.2]
df_xcats.loc["EQCRY_NSA"] = ["2010-01-01", "2022-03-14", 2, 1.5, 0.9, 0.5]
df_xcats.loc["EQCRR_NSA"] = ["2010-01-01", "2022-03-14", 1.5, 1.5, 0.9, 0.5]
df_xcats.loc["FXWBASE_NSA"] = ["2010-01-01", "2022-02-01", 1, 1.5, 0.8, 0.5]
df_xcats.loc["EQWBASE_NSA"] = ["2010-01-01", "2022-02-01", 1, 1.5, 0.9, 0.5]

# Call `make_qdf` to create the mock data
dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
dfd["grading"] = 1

# Create a blacklist for dates with no/incomplete data
black = {
    "AUD": ["2010-01-01", "2013-12-31"],
    "GBP": ["2010-01-01", "2013-12-31"],
}

# List of contracts
contracts = ["AUD_FX", "AUD_EQ", "NZD_FX", "GBP_EQ", "USD_EQ"]

# List of contracts
contracts_1 = ["AUD_FX", "GBP_FX", "NZD_FX", "USD_EQ"]

# %% [markdown]
# ## Instantiate the `Basket` class with multiple contracts, a return and two carry categories
# %%
basket_1 = Basket(
    df=dfd,
    contracts=contracts_1,
    ret="XR_NSA",
    cry=["CRY_NSA", "CRR_NSA"],
    blacklist=black,
)
# %% [markdown]
# ## Example 1 - Create a basket with equal weights
# %%
basket_1.make_basket(
    weight_meth="equal",
    max_weight=0.55,
    basket_name="GLB_EQUAL",
)
# %% [markdown]
# ## Example 2 - Create a basket with fixed weights
# %%
custom_weights = [1 / 6, 1 / 6, 1 / 6, 1 / 2]

basket_1.make_basket(
    weight_meth="fixed",
    max_weight=0.55,
    weights=custom_weights,
    basket_name="GLB_FIXED",
)
# %% [markdown]
# ## Example 3 - Create a basket with fixed weights
# %%
basket_1.weight_visualiser(basket_name="GLB_FIXED", subplots=False, size=(10, 5))
