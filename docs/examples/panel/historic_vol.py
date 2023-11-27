"""example/macrosynergy/panel/historic_vol.py"""


from macrosynergy.management.simulate import make_qdf
from macrosynergy.panel.historic_vol import historic_vol
import pandas as pd

## Set the currency areas (cross-sectional identifiers) and categories

cids = ["AUD", "CAD", "GBP", "USD"]
xcats = ["XR", "CRY", "GROWTH", "INFL"]

## Creating the mock data

cols = ["earliest", "latest", "mean_add", "sd_mult"]

df_cids = pd.DataFrame(index=cids, columns=cols)

df_cids.loc["AUD"] = ["2010-01-01", "2020-12-31", 0.5, 2]
df_cids.loc["CAD"] = ["2011-01-01", "2020-11-30", 0, 1]
df_cids.loc["GBP"] = ["2012-01-01", "2020-10-30", -0.2, 0.5]
df_cids.loc["USD"] = ["2013-01-01", "2020-09-30", -0.2, 0.5]

cols += ["ar_coef", "back_coef"]

df_xcats = pd.DataFrame(index=xcats, columns=cols)

df_xcats.loc["XR"] = ["2010-01-01", "2020-12-31", 0, 1, 0, 0.3]
df_xcats.loc["CRY"] = ["2011-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
df_xcats.loc["GROWTH"] = ["2012-01-01", "2020-10-30", 1, 2, 0.9, 1]
df_xcats.loc["INFL"] = ["2013-01-01", "2020-10-30", 1, 2, 0.8, 0.5]

dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
dfd["grading"] = 1


## Example 1 - Calculate historic volatility with the moving average method
df = historic_vol(
    dfd,
    cids=cids,
    xcat="XR",
    lback_periods=7,
    lback_meth="ma",
    est_freq="w",
    half_life=3,
    remove_zeros=True,
)

print(df.head(10))

## Example 2 - Calculate historic volatility with the exponential moving average method

df = historic_vol(
    dfd,
    cids=cids,
    xcat="XR",
    lback_periods=7,
    lback_meth="xma",
    est_freq="w",
    half_life=3,
    remove_zeros=True,
)


print(df.head(10))
