"""example/macrosynergy/learning/transformers.py"""


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

selector = MapSelectorTransformer(0.05)
selector.fit(X, y)
print(selector.transform(X).columns)

selector = LassoSelectorTransformer(0.00001)
selector.fit(X, y)
print(selector.transform(X).columns)

# Split X and y into training and test sets
X_train, X_test = X[X.index.get_level_values(1) < pd.Timestamp(day=1,month=1,year=2018)], X[X.index.get_level_values(1) >= pd.Timestamp(day=1,month=1,year=2018)]
y_train, y_test = y[y.index.get_level_values(1) < pd.Timestamp(day=1,month=1,year=2018)], y[y.index.get_level_values(1) >= pd.Timestamp(day=1,month=1,year=2018)]

selector = BenchmarkTransformer(neutral="mean", use_signs=True)
selector.fit(X_train, y_train)
print(selector.transform(X_test))

selector = BenchmarkTransformer(neutral="zero")
selector.fit(X_train, y_train)
print(selector.transform(X_test))
