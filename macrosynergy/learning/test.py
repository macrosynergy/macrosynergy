import os
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.base import BaseEstimator, TransformerMixin

from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.tools import add_constant

from sklearn.metrics import (
    make_scorer,
    balanced_accuracy_score,
    r2_score,
)

import macrosynergy.management as msm
import macrosynergy.panel as msp
import macrosynergy.pnl as msn
import macrosynergy.signal as mss
import macrosynergy.learning as msl
from macrosynergy.download import JPMaQSDownload

# download 

# Cross-sections of interest

cids_dm = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NOK", "NZD", "SEK", "USD"]
cids_em = [
    "CLP",
    "COP",
    "CZK",
    "HUF",
    "IDR",
    "ILS",
    "INR",
    "KRW",
    "MXN",
    "PLN",
    "THB",
    "TRY",
    "TWD",
    "ZAR",
]
cids = cids_dm + cids_em
cids_du = cids_dm + cids_em
cids_dux = list(set(cids_du) - set(["IDR", "NZD"]))
cids_xg2 = list(set(cids_dux) - set(["EUR", "USD"]))

main = [
    "RYLDIRS05Y_NSA",
    "INTRGDPv5Y_NSA_P1M1ML12_3MMA",
    "CPIC_SJA_P6M6ML6AR",
    "CPIH_SA_P1M1ML12",
    "INFTEFF_NSA",
    "PCREDITBN_SJA_P1M1ML12",
    "RGDP_SA_P1Q1QL4_20QMA",
]

mkts = [
    "DU05YXR_VT10",
    "FXTARGETED_NSA", 
    "FXUNTRADABLE_NSA"
]


xcats = main + mkts

tickers = [cid + "_" + xcat for cid in cids for xcat in xcats]

# download 

start_date = "2000-01-01"
end_date = None

oauth_id = os.getenv("DQ_CLIENT_ID")  # Replace with own client ID
oauth_secret = os.getenv("DQ_CLIENT_SECRET")  # Replace with own secret

with JPMaQSDownload(client_id=oauth_id, client_secret=oauth_secret) as downloader:
    df = downloader.download(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        metrics=["value"],
        suppress_warning=True,
        show_progress=True,
    )

dfx = df.copy()

# Create blacklisting dictionary

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

# feature transformations

calcs = [
    "XGDP_NEG = - INTRGDPv5Y_NSA_P1M1ML12_3MMA",
    "XCPI_NEG =  - ( CPIC_SJA_P6M6ML6AR + CPIH_SA_P1M1ML12 ) / 2 + INFTEFF_NSA",
    "XPCG_NEG = - PCREDITBN_SJA_P1M1ML12 + INFTEFF_NSA + RGDP_SA_P1Q1QL4_20QMA",
]

dfa = msp.panel_calculator(dfx, calcs=calcs, cids=cids)
dfx = msm.update_df(dfx, dfa)

macros = ["XGDP_NEG", "XCPI_NEG", "XPCG_NEG", "RYLDIRS05Y_NSA"]
xcatx = macros

for xc in xcatx:
    dfa = msp.make_zn_scores(
        dfx,
        xcat=xc,
        cids=cids,
        neutral="zero",
        thresh=3,
        est_freq="M",
        pan_weight=1,
        postfix="_ZN4",
    )
    dfx = msm.update_df(dfx, dfa)

dfa = msp.linear_composite(
    df=dfx,
    xcats=[xc + "_ZN4" for xc in xcatx],
    cids=cids,
    new_xcat="MACRO_AVGZ",
)

dfx = msm.update_df(dfx, dfa)

macroz = [m + "_ZN4" for m in macros]
xcatx = macroz

xcatx = macroz + ["DU05YXR_VT10"]

dfw = msm.categories_df(
    df=dfx,
    xcats=xcatx,
    cids=cids_dux,
    freq="M",
    lag=1,
    blacklist=fxblack,
    xcat_aggs=["last", "sum"],
)

dfw.dropna(inplace=True)
X = dfw.iloc[:, :-1]
y = dfw.iloc[:, -1]

# required classes

def panel_significance_probability(y_true, y_pred):
    # regress ground truth against predictions
    X = add_constant(y_pred)
    if np.all(y_pred == 0):
        return 0
    
    groups = y_true.index.get_level_values(1)
    # fit model
    re = MixedLM(y_true, X, groups=groups).fit(reml=False)
    pval = re.pvalues.iloc[1]

    return 1 - pval

class MapSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float):        
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.ftrs = []
        self.cols = X.columns

        for col in self.cols:
            ftr = X[col]
            ftr = add_constant(ftr)
            groups = ftr.index.get_level_values(1)
            model = MixedLM(y,ftr,groups).fit(reml=False)
            est = model.params.iloc[1]
            pval = model.pvalues.iloc[1]
            if (pval < self.threshold) & (est > 0):
                self.ftrs.append(col)

        return self

    def transform(self, X: pd.DataFrame):
        if self.ftrs == []:
            return pd.DataFrame(index=X.index, columns=["no_signal"], data=0,dtype=np.float16)
        
        return X[self.ftrs]
    
# Setup pipeline 
    
# Define models and grids for optimization
mods_fsz = {
    "MAP_Z": Pipeline(
        [
            ("selector", MapSelector(threshold=0.05)),
            ("zscore", msl.FeatureAverager()),
            ("predictor", msl.NaivePredictor()),
        ]
    ),
}

grids_fsz = {
    "MAP_Z": {
        "selector__threshold": [0.01, 0.05, 0.1, 0.2],
    },
}

score_fsz = make_scorer(panel_significance_probability)
splitter_fsz = msl.RollingKFoldPanelSplit(n_splits=4)

# Signal optimization
print("Start optimisation")
so_fsz = msl.SignalOptimizer(inner_splitter=splitter_fsz, X=X, y=y, blacklist=fxblack)
so_fsz.calculate_predictions(
    name="MACRO_OPTSELZ",
    models=mods_fsz,
    hparam_grid=grids_fsz,
    metric=score_fsz,
    min_cids=4,
    min_periods=36,
    n_jobs = -1
)