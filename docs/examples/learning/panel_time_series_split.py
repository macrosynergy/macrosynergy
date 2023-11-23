"""example/macrosynergy/learning/panel_time_series_split.py"""


cids = ["AUD", "CAD", "GBP", "USD"]
xcats = ["XR", "CRY", "GROWTH", "INFL"]
cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

# """Example 1: Unbalanced panel """

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

# 1) Demonstration of basic functionality

# a) n_splits = 4, n_split_method = expanding
splitter = ExpandingKFoldPanelSplit(n_splits=4)
splitter.split(X2, y2)
cv_results = cross_validate(
    LinearRegression(), X2, y2, cv=splitter, scoring="neg_root_mean_squared_error"
)
splitter.visualise_splits(X2, y2)

# b) n_splits = 4, n_split_method = rolling
splitter = RollingKFoldPanelSplit(n_splits=4)
splitter.split(X2, y2)
cv_results = cross_validate(
    LinearRegression(), X2, y2, cv=splitter, scoring="neg_root_mean_squared_error"
)
splitter.visualise_splits(X2, y2)

# c) train_intervals = 21*12, test_size = 21*12, min_periods = 21 , min_cids = 4
splitter = ExpandingIncrementPanelSplit(
    train_intervals=21 * 12, test_size=1, min_periods=21, min_cids=4
)
splitter.split(X2, y2)
cv_results = cross_validate(
    LinearRegression(), X2, y2, cv=splitter, scoring="neg_root_mean_squared_error"
)
splitter.visualise_splits(X2, y2)

# d) train_intervals = 21*12, test_size = 21*12, min_periods = 21 , min_cids = 4, max_periods=12*21
splitter = ExpandingIncrementPanelSplit(
    train_intervals=21 * 12,
    test_size=21 * 12,
    min_periods=21,
    min_cids=4,
    max_periods=12 * 21,
)
splitter.split(X2, y2)
cv_results = cross_validate(
    LinearRegression(), X2, y2, cv=splitter, scoring="neg_root_mean_squared_error"
)
splitter.visualise_splits(X2, y2)

# 2) Grid search capabilities
lasso = Lasso()
parameters = {"alpha": [0.1, 1, 10]}
splitter = ExpandingIncrementPanelSplit(
    train_intervals=21 * 12, test_size=1, min_periods=21, min_cids=4
)
gs = GridSearchCV(
    lasso,
    parameters,
    cv=splitter,
    scoring="neg_root_mean_squared_error",
    refit=False,
    verbose=3,
)
gs.fit(X2, y2)
print(gs.best_params_)

# Define the cids
cids = ["cid1", "cid2"]

# Define the dates
dates_cid1 = pd.date_range(start="2023-01-01", periods=10, freq="D")
dates_cid2 = pd.date_range(
    start="2023-01-02", periods=9, freq="D"
)  # only 9 dates for cid2

# Create a MultiIndex for each cid with the respective dates
multiindex_cid1 = pd.MultiIndex.from_product(
    [["cid1"], dates_cid1], names=["cid", "real_date"]
)
multiindex_cid2 = pd.MultiIndex.from_product(
    [["cid2"], dates_cid2], names=["cid", "real_date"]
)

# Concatenate the MultiIndexes
multiindex = multiindex_cid1.append(multiindex_cid2)

# Initialize a DataFrame with the MultiIndex and columns A and B.
# Fill in some example data using random numbers.
df = pd.DataFrame(
    np.random.rand(len(multiindex), 2), index=multiindex, columns=["A", "B"]
)

X2 = df.drop(columns=["B"])
y2 = df["B"]

# splitter = ExpandingIncrementPanelSplit(
#     train_intervals=1,
#     test_size=2,
#     min_periods=1,
#     min_cids=2,
#     max_periods=12 * 21,
# )
splitter = ExpandingKFoldPanelSplit(n_splits=4)
splits = splitter.split(X2, y2)
print(splits)
splitter.visualise_splits(X2, y2)
