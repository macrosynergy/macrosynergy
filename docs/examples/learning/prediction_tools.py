"""example/macrosynergy/learning/prediction_tools.py"""

    regression_balanced_accuracy,
    MapSelectorTransformer,
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
# dfd2 = msm.reduce_df(df=dfd2, cids=cids, xcats=xcats, blacklist=black)
dfd2 = msm.categories_df(
    df=dfd2, xcats=xcats, cids=cids, val="value", blacklist=black, freq="M", lag=1
).dropna()
X = dfd2.drop(columns=["XR"])
y = dfd2["XR"]
y_long = pd.melt(
    frame=y.reset_index(), id_vars=["cid", "real_date"], var_name="xcat"
)

# (1) Example AdaptiveSignalHandler usage.
#     We get adaptive signals for a linear regression and a KNN regressor, with the
#     hyperparameters for the latter optimised across regression balanced accuracy.

models = {
    "OLS": LinearRegression(),
    "KNN": KNeighborsRegressor(),
}
metric = make_scorer(regression_balanced_accuracy, greater_is_better=True)

inner_splitter = RollingKFoldPanelSplit(n_splits=4)

hparam_grid = {
    "OLS": {},
    "KNN": {"n_neighbors": [1, 2, 5]},
}

ash = AdaptiveSignalHandler(
    inner_splitter=inner_splitter,
    X=X,
    y=y,
)

preds, models = ash.calculate_predictions(
    name="test",
    models=models,
    metric=metric,
    hparam_grid=hparam_grid,
    hparam_type="grid",
)

print(preds, models)

# (2) Example AdaptiveSignalHandler usage.
#     Visualise the model selection heatmap for the two most frequently selected models.
ash.models_heatmap(name="test", cap=2)

# (3) Example AdaptiveSignalHandler usage.
#     We get adaptive signals for two KNN regressors. 
#     All chosen models are visualised in a heatmap.
models2 = {
    "KNN1": KNeighborsRegressor(),
    "KNN2": KNeighborsRegressor(),
}
hparam_grid2 = {
    "KNN1": {"n_neighbors": [5, 10]},
    "KNN2": {"n_neighbors": [1, 2]},
}

preds2, models2 = ash.calculate_predictions(
    name="test2",
    models=models2,
    metric=metric,
    hparam_grid=hparam_grid2,
    hparam_type="grid",
)

print(preds2, models2)
ash.models_heatmap(name="test2", cap=4)

# (4) Example AdaptiveSignalHandler usage.
#     Print the predictions and model choices for all pipelines.
print(ash.get_all_preds())
print(ash.get_all_models())
