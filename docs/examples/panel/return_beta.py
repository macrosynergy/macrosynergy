"""example/macrosynergy/panel/return_beta.py"""
# %% [markdown]
# ## Imports
# %%
from macrosynergy.management.simulate import make_qdf
from macrosynergy.panel.return_beta import return_beta, beta_display
import pandas as pd


# %% [markdown]
# ## Set the currency areas (cross-sectional identifiers) and categories
# %%
cids = ["IDR", "INR", "KRW", "MYR", "PHP", "USD"]
xcats = ["FXXR_NSA", "GROWTHXR_NSA", "INFLXR_NSA", "EQXR_NSA"]

cols = ["earliest", "latest", "mean_add", "sd_mult"]
df_cids = pd.DataFrame(index=cids, columns=cols)
# %% [markdown]
# ## Creating the mock data
# %%
df_cids.loc["IDR"] = ["2010-01-01", "2020-12-31", 0.5, 2]
df_cids.loc["INR"] = ["2011-01-01", "2020-11-30", 0, 1]
df_cids.loc["KRW"] = ["2012-01-01", "2020-11-30", -0.2, 0.5]
df_cids.loc["MYR"] = ["2013-01-01", "2020-09-30", -0.2, 0.5]
df_cids.loc["PHP"] = ["2002-01-01", "2020-09-30", -0.1, 2]
df_cids.loc["USD"] = ["2000-01-01", "2022-03-14", 0, 1.25]

cols += ["ar_coef", "back_coef"]
df_xcats = pd.DataFrame(index=xcats, columns=cols)

df_xcats.loc["FXXR_NSA"] = ["2010-01-01", "2020-10-30", 1, 2, 0.9, 1]
df_xcats.loc["GROWTHXR_NSA"] = ["2012-01-01", "2020-10-30", 1, 2, 0.9, 1]
df_xcats.loc["INFLXR_NSA"] = ["2013-01-01", "2020-10-30", 1, 2, 0.8, 0.5]
df_xcats.loc["EQXR_NSA"] = ["2010-01-01", "2022-03-14", 0.5, 2, 0, 0.2]


dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

black = {"IDR": ["2010-01-01", "2014-01-04"], "INR": ["2010-01-01", "2013-12-31"]}

# %% [markdown]
# ## Example 1 - Calculating the beta of *_EQXR_NSA with respect to USD_EQXR_NSA
# %%
# S&P500.
xcat_hedge = "EQXR_NSA"
benchmark_return = "USD_EQXR_NSA"

df_hedge = return_beta(
    df=dfd,
    xcat=xcat_hedge,
    cids=cids,
    benchmark_return=benchmark_return,
    start="2010-01-01",
    end="2020-10-30",
    blacklist=black,
    meth="ols",
    oos=True,
    refreq="w",
    min_obs=24,
    hedged_returns=True,
)


print(df_hedge)

# %% [markdown]
# ## Example 2 - Using the beta_display function to plot Example 1
# %%
beta_display(df_hedge=df_hedge, subplots=False)

# %% [markdown]
# ## Example 3 - Calculating the beta for USD_FXXR_NSA with respect to USD_EQXR_NSA
# %%
# Long position in S&P500 or the Nasdaq, and subsequently using US FX to hedge the
# long position.

xcats = "FXXR_NSA"
cids = ["USD"]

benchmark_return = "USD_EQXR_NSA"

xcat_hedge_two = return_beta(
    df=dfd,
    xcat=xcats,
    cids=cids,
    benchmark_return=benchmark_return,
    start="2010-01-01",
    end="2020-10-30",
    blacklist=black,
    meth="ols",
    oos=True,
    refreq="m",
    min_obs=24,
)

print(xcat_hedge_two)
