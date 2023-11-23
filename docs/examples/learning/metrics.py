"""example/macrosynergy/learning/metrics.py"""


cids = ["AUD", "CAD", "GBP", "USD"]
xcats = ["XR", "CPI", "GROWTH", "RIR"]

df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31"]
df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31"]
df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31"]
df_cids.loc["USD"] = ["2000-01-01", "2020-12-31"]

tuples = []

for cid in cids:
    # get list of all elidgible dates
    sdate = df_cids.loc[cid]["earliest"]
    edate = df_cids.loc[cid]["latest"]
    all_days = pd.date_range(sdate, edate)
    work_days = all_days[all_days.weekday < 5]
    for work_day in work_days:
        tuples.append((cid, work_day))

n_samples = len(tuples)
ftrs = np.random.normal(loc=0, scale=1, size=(n_samples, 3))
labels = np.random.normal(loc=0, scale=1, size=n_samples)
df = pd.DataFrame(
    data=np.concatenate((np.reshape(labels, (-1, 1)), ftrs), axis=1),
    index=pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"]),
    columns=xcats,
    dtype=np.float32,
)

X = df.drop(columns="XR")
y = df["XR"]

splitter = ExpandingKFoldPanelSplit(n_splits=4)
scorer1 = make_scorer(panel_significance_probability, greater_is_better=True)
scorer2 = make_scorer(sharpe_ratio, greater_is_better=True)
scorer3 = make_scorer(sortino_ratio, greater_is_better=True)
scorer4 = make_scorer(regression_accuracy, greater_is_better=True)
scorer5 = make_scorer(regression_balanced_accuracy, greater_is_better=True)
cv_results1 = cross_val_score(
    LinearRegression(), X, y, cv=splitter, scoring=scorer1
)
cv_results2 = cross_val_score(
    LinearRegression(), X, y, cv=splitter, scoring=scorer2
)
cv_results3 = cross_val_score(
    LinearRegression(), X, y, cv=splitter, scoring=scorer3
)
cv_results4 = cross_val_score(
    LinearRegression(), X, y, cv=splitter, scoring=scorer4
)
cv_results5 = cross_val_score(
    LinearRegression(), X, y, cv=splitter, scoring=scorer5
)
print("Probabilities of significances, per split:", cv_results1)
print("Sharpe ratios, per split:", cv_results2)
print("Sortino ratios, per split:", cv_results3)
print("Regression accuracies, per split:", cv_results4)
print("Regression balanced accuracies, per split:", cv_results5)
print("Done")
