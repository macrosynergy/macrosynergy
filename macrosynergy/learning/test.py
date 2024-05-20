import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

import json
import yaml

import macrosynergy.management as msm
import macrosynergy.panel as msp
import macrosynergy.signal as mss
import macrosynergy.pnl as msn
import macrosynergy.learning as msl

from macrosynergy.download import JPMaQSDownload

import scipy.stats as stats

from timeit import default_timer as timer
from datetime import timedelta, date, datetime

from joblib import Parallel, delayed

from tqdm import tqdm

import warnings

warnings.simplefilter("ignore")

# Cross-sections of interest

cids_dmca = [
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
cids_dmec = ["DEM", "ESP", "FRF", "ITL", "NLG"]  # DM euro area countries
cids_latm = ["BRL", "COP", "CLP", "MXN", "PEN"]  # Latam countries
cids_emea = ["CZK", "HUF", "ILS", "PLN", "RON", "RUB", "TRY", "ZAR"]  # EMEA countries
cids_emas = [
    "CNY",
    # "HKD",
    "IDR",
    "INR",
    "KRW",
    "MYR",
    "PHP",
    "SGD",
    "THB",
    "TWD",
]  # EM Asia countries

cids_dm = cids_dmca + cids_dmec
cids_em = cids_latm + cids_emea + cids_emas

cids = sorted(cids_dm + cids_em)

main = ["FXXR_NSA", "FXXR_VT10", "FXXRHvGDRB_NSA"]

econ = []

mark = [
    "EQXR_NSA",
    "FXXRBETAvGDRB_NSA",
    "FXTARGETED_NSA",
    "FXUNTRADABLE_NSA",
]  # related market categories

xcats = main + econ + mark

xtix = ["GLB_DRBXR_NSA", "GEQ_DRBXR_NSA"]

# Download series from J.P. Morgan DataQuery by tickers

start_date = "1990-01-01"
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
        metrics=["value", "eop_lag", "mop_lag", "grading"],
        suppress_warning=True,
        show_progress=True,
    )
    end = timer()

dfd = df.copy()

print("Download time from DQ: " + str(timedelta(seconds=end - start)))

inner_splitter = msl.ExpandingKFoldPanelSplit(n_splits=5)

models = {
    "LR_ROLL": msl.LinearRegressionSystem(
        min_xs_samples=21 * 12, fit_intercept=True, positive=False
    ),
    # "LAD_ROLL": msl.LADRegressionSystem(
    #    min_xs_samples=21 * 12, fit_intercept=True, positive=False
    # ),
    #"RIDGE_ROLL": msl.RidgeRegressionSystem(
    #    min_xs_samples=21 * 12, fit_intercept=True, positive=False
    #),
}

grid = {
    "LR_ROLL": [
        {"roll": [21, 21 * 3, 21 * 6, 21 * 12, 21 * 24, 21 * 60], "data_freq": ["D"]},
        {"roll": [4, 4 * 3, 4 * 6, 4 * 12, 4 * 24, 4 * 60], "data_freq": ["W"]},
    ],
    # "LAD_ROLL": [
    #    {"roll": [21, 21 * 3, 21 * 6, 21 * 12, 21 * 24, 21 * 60], "data_freq": ["D"]},
    #    {"roll": [4, 4 * 3, 4 * 6, 4 * 12, 4 * 24, 4 * 60], "data_freq": ["W"]},
    # ],
    #"RIDGE_ROLL": [
    #    {
    #        "roll": [21, 21 * 3, 21 * 6, 21 * 12, 21 * 24, 21 * 60],
    #        "data_freq": ["D"],
    #        "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
    #    },
    #    {
    #        "roll": [4, 4 * 3, 4 * 6, 4 * 12, 4 * 24, 4 * 60],
    #        "data_freq": ["W"],
    #        "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
    #    },
    #],
}

cidx = cids_em

be = msl.BetaEstimator(
    df=dfd, xcat="FXXR_NSA", cids=cidx, benchmark_return="GLB_DRBXR_NSA"
)

be.estimate_beta(
    beta_xcat="EMBETA_NSA",
    hedged_return_xcat="EMFXXRH_NSA",
    inner_splitter=inner_splitter,
    scorer=msl.neg_mean_abs_corr,
    models=models,
    hparam_grid=grid,
    min_cids=4,
    min_periods=21 * 12 * 2,
    est_freq="Q",
)
be.models_heatmap("EMBETA_NSA")