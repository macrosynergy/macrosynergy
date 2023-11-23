"""example/macrosynergy/learning/cv_tools.py"""


    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

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
X2 = dfd2.drop(columns=["XR"])
y2 = dfd2["XR"]

# 1) Demonstration of panel_cv_scores
splitex = ExpandingKFoldPanelSplit(n_splits=100)
models = {"OLS": LinearRegression(), "Lasso": Lasso()}
metrics = {
    "rmse": make_scorer(
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
    ),
    "mae": make_scorer(mean_absolute_error),
    "mape": make_scorer(mean_absolute_percentage_error),
    "acc": make_scorer(msl.regression_accuracy),
    "bac": make_scorer(msl.regression_balanced_accuracy),
    "map": make_scorer(msl.panel_significance_probability),
    "sharpe": make_scorer(msl.sharpe_ratio),
    "sortino": make_scorer(msl.sortino_ratio),
}
df_ev = panel_cv_scores(
    X2,
    y2,
    splitter=splitex,
    estimators=models,
    scoring=metrics,
    show_longbias=True,
    show_std=False,
    n_jobs=-1,
    verbose=1,
)
print(df_ev)
