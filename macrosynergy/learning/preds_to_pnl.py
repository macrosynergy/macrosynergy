"""
Collection of functions to convert machine learning model predictions into custom PnLs.
There are two cases that we consider:
1) Simple cross-validation where hyper-parameters are chosen over an initial training set
   and then fixed for all hold-out sets (despite retraining).
2) Nested cross-validation allowing for adaptive hyper-parameter selection at each test
   time.
TODO: write adaptive_preds_to_pnl function
"""

import numpy as np
import pandas as pd

from macrosynergy.learning import PanelTimeSeriesSplit
import macrosynergy.pnl as msn

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from typing import List, Union, Dict, Optional


def static_preds_to_pnl(
    models: Dict[str, Union[BaseEstimator, Pipeline]],
    X: pd.DataFrame,
    y: pd.Series,
    splitter: PanelTimeSeriesSplit,
    daily_return_series: pd.DataFrame,
    sig_mode: str = "binary",
    rebal_freq: str = "monthly",
    min_obs: int = 12,
    additional_X: Optional[List[pd.DataFrame]] = None,
    additional_y: Optional[List[pd.Series]] = None,
):
    """
    Function to create a naive PnL from the predictions of a hyperparameter-static model.

    :param <Dict[str, Union[BaseEstimator,Pipeline]]> model: dictionary of sklearn predictors.
        PnLs are generated from predictions for each of the models in the dictionary.
    :param <pd.DataFrame> X: Wide-format pandas dataframe of features over the time period for which
        the PnL is to be calculated. These should lag behind returns by a single frequency
        unit. 
    :param <pd.Series> y: Pandas series of targets corresponding with the features in X.
        This should be the same length as X.
    :param <PanelTimeSeriesSplit> splitter: Panel walk-forward validation splitter.
        Predictions are made for each test sample, created by the splitter,
        and subsequently stored in a signal dataframe. 
    :param <pd.DataFrame> daily_return_series: Pandas dataframe of daily returns
        for each cross-section in X. This spans the minimum and maximum dates in X.
        The dataframe should have the following columns: cid, real_date, xcat, value.
        Since each signal is propagated forward to the next rebalancing date, a daily
        return series is required to add to the internal signal dataframe for the 
        cumulative PnL.
    :param <str> sig_mode: Signal transformation option for PnL construction. Either "binary" or "zn_score_pan".
        Default is "binary". 
    :param <str> rebal_freq: Rebalancing frequency for the PnL. Either "daily", "weekly" or "monthly".
        Default is "monthly".
    :param <int> min_obs: Minimum number of observations required to calculate 
        make_zn_scores. This is only relevant if sig_mode == "zn_score_pan".
        Default is 12 (one year), since the default rebalancing frequency is monthly.
        If rebal_freq is "daily", 252 days is the recommended value. If rebal_freq is "weekly", 
        52 weeks is the recommended value.
    :param <List[pd.DataFrame]> additional_X: List of Pandas dataframes of additional, "seen" features
        prior to the PnL calculation over (X, y). These should be listed sequentially
        and will be appended to each training dataframe created by the splitter.
    :param <List[pd.Series]> additional_y: List of Pandas series of additional, "seen" targets
        prior to the PnL calculation over (X, y). These should be listed sequentially
        and will be appended to each training target series created by the splitter. These
        should be ordered accordingly with the dataframes in additional_X.
    """
    # (1) Create a dataframe to store the signals induced by each model and cross-section. 
    signal_xs_levels: List[str] = sorted(X.index.get_level_values(0).unique())
    original_date_levels: List[pd.Timestamp] = sorted(X.index.get_level_values(1).unique())
    min_date: pd.Timestamp = min(original_date_levels)
    max_date: pd.Timestamp = max(original_date_levels)
    signal_date_levels: pd.DatetimeIndex = pd.bdate_range(start=min_date, end=max_date, freq="B")

    sig_idxs = pd.MultiIndex.from_product(
        [signal_xs_levels, signal_date_levels], names=["cid", "real_date"]
    )
    signal_df: pd.MultiIndex = pd.DataFrame(
        index=sig_idxs, columns=models.keys(), data=np.nan, dtype="float64"
    )
    # (2) For each model, create the signal that it induces.
    # (2a) First concatenate the additional dataframes and series, if they exist, for
    #      efficiency in the later loop. 
    if additional_X is not None:
        X_old: pd.DataFrame = pd.concat(additional_X, axis=0)
        y_old: pd.Series = pd.concat(additional_y, axis=0)

    # (2b) For each model and each (training, test) pair, fit the model (with additional data if necessary)
    #      and store the test set predictions in the signal dataframe.
    for train_idx, test_idx in splitter.split(X, y):
        # Set up training and test sets
        X_train_i: pd.DataFrame = X.iloc[train_idx]
        y_train_i: pd.Series = y.iloc[train_idx]
        X_test_i: pd.DataFrame = X.iloc[test_idx]
        # Append additional data if necessary
        if additional_X is not None:
            X_train_i = pd.concat([X_train_i, X_old], axis=0)
            y_train_i = pd.concat([y_train_i, y_old], axis=0)
        # Get correct index to match with
        test_xs_levels: List[str] = X_test_i.index.get_level_values(0).unique()
        test_date_levels: List[pd.Timestamp] = sorted(X_test_i.index.get_level_values(1).unique())
        # Since the features lag behind the targets, the dates need to be adjusted 
        # by a single frequency unit
        locs: np.ndarray = np.searchsorted(original_date_levels, test_date_levels, side="left") - 1
        test_date_levels: pd.DatetimeIndex = pd.DatetimeIndex([original_date_levels[i] if i >= 0 else pd.NaT for i in locs])
        # Fit and predict for each model
        for name, model in models.items():
            # Train model
            model.fit(X_train_i, y_train_i)
            preds: np.ndarray = model.predict(X_test_i)
            # Store the predictions.
            sig_idxs: pd.MultiIndex = pd.MultiIndex.from_product([test_xs_levels, test_date_levels])
            signal_df[name].loc[sig_idxs] = preds

    # (3) Now transform the signal dataframe into a format that can be used by the 
    #     Naive PnL class. 
    signal_df = signal_df.groupby(level=0).ffill()
    signal_df_long: pd.DataFrame = pd.melt(
        frame=signal_df.reset_index(), id_vars=["cid", "real_date"], var_name="xcat"
    )
    pnl_start = signal_df_long.dropna().real_date.min()
    returns = daily_return_series[daily_return_series.xcat.isin([y.name])][
        ["cid", "real_date", "xcat", "value"]
    ]
    signal_df_long = pd.concat((signal_df_long, returns), axis=0)

    # (4) Finally, create the PnLs.
    pnl = msn.NaivePnL(
        df=signal_df_long,
        ret=y.name,
        sigs=list(models.keys()),
        cids=signal_xs_levels,
        start=pnl_start,
    )

    pnl.make_long_pnl(
        vol_scale=10,
        label="Long_Only",
    )

    for sig in list(models.keys()):
        pnl.make_pnl(
            sig,
            sig_neg=False,
            sig_op=sig_mode,
            rebal_freq=rebal_freq, 
            vol_scale=10,
            rebal_slip=1,
            pnl_name=(sig + "_BINARY" if sig_mode == "binary" else sig + "_ZN"),
            thresh=3,
            min_obs=min_obs,
        )

    title = "Cumulative naive PnLs, binary signal" if sig_mode == "binary" else "Cumulative naive PnLs, z-score signal"
    pnl.plot_pnls(title=title)

if __name__ == "__main__":
    from macrosynergy.management.simulate import make_qdf
    import macrosynergy.management as msm
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    """Example 1: Unbalanced panel """

    df_cids2 = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids2.loc["AUD"] = ["2002-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["CAD"] = ["2003-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["GBP"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["USD"] = ["2000-01-01", "2020-12-31", 0, 1]

    df_xcats2 = pd.DataFrame(index=xcats, columns=cols)
    df_xcats2.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats2.loc["CRY"] = ["2000-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats2.loc["GROWTH"] = ["2000-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats2.loc["INFL"] = ["2000-01-01", "2020-12-31", 1, 2, 0.8, 0.5]

    dfd2 = make_qdf(df_cids2, df_xcats2, back_ar=0.75)
    dfd2["grading"] = np.ones(dfd2.shape[0])
    black = {"GBP": ["2009-01-01", "2012-06-30"], "CAD": ["2018-01-01", "2100-01-01"]}
    dfd2 = msm.reduce_df(df=dfd2, cids=cids, xcats=xcats, blacklist=black)

    dfd2 = dfd2.pivot(index=["cid", "real_date"], columns="xcat", values="value")
    X = dfd2.drop(columns=["XR"])
    y = dfd2["XR"]
    y_long = pd.melt(
        frame=y.reset_index(), id_vars=["cid", "real_date"], var_name="xcat"
    )
    splitter = PanelTimeSeriesSplit(
        train_intervals=1, test_size=1, min_cids=4, min_periods=2
    )
    models = {"ols": LinearRegression(), "knn": KNeighborsRegressor()}
    #static_preds_to_pnl(models=models, splitter=splitter, X=X, y=y, daily_return_series=y_long, sig_mode="binary")
    static_preds_to_pnl(models=models, splitter=splitter, X=X, y=y, daily_return_series=y_long, sig_mode="zn_score_pan")
