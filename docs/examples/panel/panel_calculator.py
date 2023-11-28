"""example/macrosynergy/panel/panel_calculator.py"""

# ## Imports
from macrosynergy.management.simulate import make_qdf
from macrosynergy.panel.panel_calculator import panel_calculator
import pandas as pd
import random

# %% [markdown]
# ## Set the currency areas (cross-sectional identifiers) and categories
# %%
cids = ["AUD", "CAD", "GBP", "USD", "NZD"]
xcats = ["XR", "CRY", "GROWTH", "INFL"]


# %% [markdown]
# ## Creating the mock data
# %%
cols = ["earliest", "latest", "mean_add", "sd_mult"]
df_cids = pd.DataFrame(index=cids, columns=cols)
df_cids.loc["AUD"] = ["2010-01-01", "2020-12-31", 0.5, 2]
df_cids.loc["CAD"] = ["2011-01-01", "2020-11-30", 0, 1]
df_cids.loc["GBP"] = ["2012-01-01", "2020-11-30", -0.2, 0.5]
df_cids.loc["USD"] = ["2010-01-01", "2020-12-30", -0.2, 0.5]
df_cids.loc["NZD"] = ["2002-01-01", "2020-09-30", -0.1, 2]
df_cids.loc["EUR"] = ["2002-01-01", "2020-09-30", -0.2, 2]

cols += ["ar_coef", "back_coef"]
df_xcats = pd.DataFrame(index=xcats, columns=cols)
df_xcats.loc["XR"] = ["2012-01-01", "2020-12-31", 0, 1, 0, 0.3]
df_xcats.loc["CRY"] = ["2010-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
df_xcats.loc["GROWTH"] = ["2012-01-01", "2020-10-30", 1, 2, 0.9, 1]
df_xcats.loc["INFL"] = ["2012-01-01", "2020-09-30", 1, 2, 0.8, 0.5]

random.seed(2)
dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

# Example blacklist.
black = {"AUD": ["2000-01-01", "2003-12-31"]}
start = "2010-01-01"
end = "2020-12-31"

# %% [markdown]
# ## Example 1 - Calculating a new XCAT
# %%
f1 = "NEW_VAR1 = GROWTH - iEUR_INFL"
formulas = [f1]
cidx = ["AUD", "CAD"]
df_calc = panel_calculator(
    df=dfd,
    calcs=formulas,
    cids=cidx,
    start=start,
    end=end,
    blacklist=black,
)
# %% [markdown]
# ## Example 2 - Calculating new XCATs with multiple formulas
# %%
cids = ["AUD", "CAD", "GBP", "USD", "NZD"]
formula = "NEW1 = XR - iUSD_XR"
formula_2 = "NEW2 = GROWTH - iEUR_INFL"
formulas = [formula, formula_2]
df_calc = panel_calculator(
    df=dfd,
    calcs=formulas,
    cids=cids,
    start=start,
    end=end,
    blacklist=black,
)
