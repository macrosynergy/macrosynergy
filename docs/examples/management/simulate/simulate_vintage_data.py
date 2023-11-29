"""example/macrosynergy/management/simulate/simulate_vintage_data.py"""
# %% [markdown]
# ## Import the VintageData class from macrosynergy.management.simulate
# %%
from macrosynergy.management.simulate import VintageData

# %% [markdown]
# ## Set the `VintageData` class
# %%
vins_m = VintageData(
    "USD_INDX_SA",
    cutoff="2019-06-30",
    release_lags=[3, 20, 25],
    number_firsts=12,
    shortest=12,
    sd_ar=5,
    trend_ar=20,
    seasonal=10,
    added_dates=6,
)

# %% [markdown]
# ## Example 1: Make a grade 1 vintage data set
# %%
dfm1 = vins_m.make_grade1()

dfm1.groupby("release_date").agg(["mean", "count"])

# %% [markdown]
# ## Example 2: Make a grade 2 vintage data set
# %%
dfm2 = vins_m.make_grade2()

# %% [markdown]
# ## Example 3: Make a "graded" data set
# %%
dfmg = vins_m.make_graded(grading=[3, 2.1, 1], upgrades=[12, 24])


vins_q = VintageData(
    "USD_INDX_SA",
    release_lags=[3, 20, 25],
    number_firsts=2,
    shortest=8,
    freq="Q",
    seasonal=10,
    added_dates=4,
)
