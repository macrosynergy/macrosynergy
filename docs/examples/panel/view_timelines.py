"""example/macrosynergy/panel/view_timelines.py"""

cids = ["AUD", "CAD", "GBP", "NZD"]
xcats = ["XR", "CRY", "INFL", "FXXR"]
df_cids = pd.DataFrame(
    index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
)
df_cids.loc["AUD"] = ["2010-01-01", "2020-12-31", 0.2, 0.2]
df_cids.loc["CAD"] = ["2011-01-01", "2020-11-30", 0, 1]
df_cids.loc["GBP"] = ["2012-01-01", "2020-11-30", 0, 2]
df_cids.loc["NZD"] = ["2012-01-01", "2020-09-30", -0.1, 3]

df_xcats = pd.DataFrame(
    index=xcats,
    columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
)

df_xcats.loc["XR"] = ["2010-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
df_xcats.loc["INFL"] = ["2015-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
df_xcats.loc["CRY"] = ["2013-01-01", "2020-10-30", 1, 2, 0.95, 0.5]
df_xcats.loc["FXXR"] = ["2013-01-01", "2020-10-30", 1, 2, 0.95, 0.5]

dfd: pd.DataFrame = make_qdf(df_cids, df_xcats, back_ar=0.75)
# sort by cid, xcat, and real_date
dfd = dfd.sort_values(["cid", "xcat", "real_date"])
ctr: int = -1
for xcat in xcats[:2]:
    for cid in cids[:2]:
        ctr *= -1
        mask = (dfd["cid"] == cid) & (dfd["xcat"] == xcat)
        dfd.loc[mask, "value"] = (
            10
            * ctr
            * np.arange(dfd.loc[mask, "value"].shape[0])
            / (dfd.loc[mask, "value"].shape[0] - 1)
        )

dfdx = dfd[~((dfd["cid"] == "AUD") & (dfd["xcat"] == "XR"))]

view_timelines(
    dfd,
    xcats=["XR", "CRY"],
    cids=cids[0],
    size=(10, 5),
    title="AUD Return and Carry",
)

view_timelines(
    dfd,
    xcats=["XR", "CRY", "INFL"],
    cids=cids[0],
    xcat_grid=True,
    title_adj=0.8,
    xcat_labels=["Return", "Carry", "Inflation"],
    title="AUD Return, Carry & Inflation",
)

view_timelines(dfd, xcats=["CRY"], cids=cids, ncol=2, title="Carry", cs_mean=True)

view_timelines(
    dfd, xcats=["XR"], cids=cids[:2], ncol=2, cumsum=True, same_y=False, aspect=2
)

dfd = dfd.set_index("real_date")
view_timelines(
    dfd,
    xcats=["XR"],
    cids=cids[:2],
    ncol=2,
    cumsum=True,
    same_y=False,
    aspect=2,
    single_chart=True,
)
