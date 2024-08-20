import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

import macrosynergy.management as msm
import macrosynergy.panel as msp
import macrosynergy.signal as mss
import macrosynergy.pnl as msn
import macrosynergy.learning as msl
import macrosynergy.visuals as msv

from macrosynergy.download import JPMaQSDownload

from sklearn.ensemble import VotingRegressor

import scipy.stats as stats

from timeit import default_timer as timer
from datetime import timedelta, date, datetime

import warnings

warnings.simplefilter("ignore")

# Cross-sections of interest

cids_dm = [
    "AUD",
    "CAD",
    "CHF",
    "EUR",
    "GBP",
    "JPY",
    "NOK",
    "NZD",
    "SEK",
    "USD",
]  # DM currency areas
cids_latm = ["BRL", "COP", "CLP", "MXN", "PEN"]  # Latam countries
cids_emea = ["CZK", "HUF", "ILS", "PLN", "RON", "RUB", "TRY", "ZAR"]  # EMEA countries
cids_emas = [
    "IDR",
    "INR",
    "KRW",
    "MYR",
    "PHP",
    "SGD",
    "THB",
    "TWD",
]  # EM Asia countries
cids_em = cids_latm + cids_emea + cids_emas
cids = sorted(cids_dm + cids_em)

# FX-specific cross sections

cids_nofx = [
    "EUR",
    "USD",
    "SGD",
]  # not small or suitable for this analysis for lack of data
cids_fx = list(set(cids) - set(cids_nofx))
cids_emfx = list(set(cids_em).intersection(cids_fx))

# Categories of interest

main = ["FXXR_NSA", "FXXRHvGDRB_NSA"] # FX forward return, % of notional: dominant cross / against USD and Return on FX forward, hedged against market direction risk
econ = []
xtra = [
    "FXXRBETAvGDRB_NSA", #  FX forward return beta with respect to a global directional risk basket.
    "FXCRR_NSA",  #  Nominal forward-implied carry vs. dominant cross: % ar
    "FXTARGETED_NSA", # Exchange rate target dummy
    "FXUNTRADABLE_NSA", # FX untradability dummy
]  # related market categories

xcats = main + econ + xtra

xtix = [
    "GLB_DRBXR_NSA", # cross-asset directional risk basket return (GLB)
    "GLB_DRBCRR_NSA", # cross-asset basket carry (GLB)
    "GEQ_DRBXR_NSA", # equity index future basket return (GEQ)
    "GEQ_DRBCRR_NSA", # equity index future basket carry (GEQ)
    "USD_EQXR_NSA", # USD equity index future return
    "USD_EQCRR_NSA", # USD equity index future carry
]

# Download series from J.P. Morgan DataQuery by tickers

start_date = "1995-01-01"
tickers = [cid + "_" + xcat for cid in cids for xcat in xcats] + xtix
print(f"Maximum number of tickers is {len(tickers)}")

# Retrieve credentials

client_id: str = os.getenv("DQ_CLIENT_ID")
client_secret: str = os.getenv("DQ_CLIENT_SECRET")

# Download from DataQuery

with JPMaQSDownload(client_id=client_id, client_secret=client_secret) as downloader:
    start = timer()
    assert downloader.check_connection()
    df = downloader.download(
        tickers=tickers,
        start_date=start_date,
        metrics=["value"],
        suppress_warning=True,
        show_progress=True,
    )
    end = timer()

dfx = df.copy()

print("Download time from DQ: " + str(timedelta(seconds=end - start)))

dfb = df[df["xcat"].isin(["FXTARGETED_NSA", "FXUNTRADABLE_NSA"])].loc[
    :, ["cid", "xcat", "real_date", "value"]
]
dfba = (
    dfb.groupby(["cid", "real_date"])
    .aggregate(value=pd.NamedAgg(column="value", aggfunc="max"))
    .reset_index()
)
dfba["xcat"] = "FXBLACK"
fxblack = msp.make_blacklist(dfba, "FXBLACK")
fxblack

# CV splitters

Xy_long = msm.categories_df(
    df=dfx,
    xcats=["FXCRR_NSA", "FXXR_NSA"],
    cids=cids,
    lag=1,
    xcat_aggs=["mean", "sum"],
    start="1999-01-01",
).dropna()

X = Xy_long.iloc[:, 0]
y = Xy_long.iloc[:, 1]

# Models and hyperparameters

models = {
    "CORRVOL": msl.predictors.CorrelationVolatilitySystem(
        min_xs_samples=21,
    ),
}
grid = {
    "CORRVOL": [
        {
            "correlation_lookback": [
                21 * 12,
                21 * 12 * 2,
                21 * 12 * 5,
            ],
            "correlation_type": ["pearson", "spearman"],
            "volatility_lookback": [5, 10, 21, 21 * 3, 21 * 6, 21 * 12],
            "volatility_window_type": ["exponential", "rolling"],
            "data_freq": ["unadjusted"],
        },
        {
            "correlation_lookback": [4 * 12, 4 * 12 * 2, 4 * 12 * 5],
            "correlation_type": ["pearson", "spearman"],
            "volatility_lookback": [4, 4 * 3, 4 * 6, 4 * 12],
            "volatility_window_type": ["exponential", "rolling"],
            "data_freq": ["W"],
        },
    ]
}

# Cross-validation splitter and evaluation score

inner_splitter = msl.ExpandingKFoldPanelSplit(n_splits=5)
eval_score = msl.neg_mean_abs_corr

# Class instantiation

cidx = cids_emfx
dfxx = dfx[dfx['real_date'] >= '1999-01-01']
be = msl.BetaEstimator(
    df=dfxx, xcat="FXXR_NSA", cids=cidx, benchmark_return="GLB_DRBXR_NSA"
)

# Produce betas and hedged returns by learning

from pyinstrument import Profiler

def profile_model_fitting(be):
    be.estimate_beta(
        beta_xcat="BETAvGDRB_CV",
        hedged_return_xcat="FXXRHvGDRB_CV",
        inner_splitter=inner_splitter,
        scorer=eval_score,
        models=models,
        hparam_grid=grid,
        min_cids=1,
        min_periods=21 * 12,
        est_freq="Q",  # Change to "Q" if time is an important factor
        n_jobs_outer=1,
    )
"""be.estimate_beta(
    beta_xcat="BETAvGDRB_OQR",
    hedged_return_xcat="FXXRHvGDRB_OQR",
    inner_splitter=inner_splitter,
    scorer=eval_score,
    models=models,
    hparam_grid=grid,
    min_cids=1,
    min_periods=21 * 12,
    est_freq="Q", # Change to "M", "W" or "D" if more computational resources are available
    n_jobs_outer=-1,
)
be.models_heatmap(
    beta_xcat="BETAvGDRB_OQR",
    title="Models heatmap for beta estimation, single-frequency method",
    cap=20,
    figsize=(12, 4),
)"""

profiler = Profiler()
profiler.start()
profile_model_fitting(be)
profiler.stop()
with open('betacorrvol_report.html', 'w') as f:
    f.write(profiler.output_html())