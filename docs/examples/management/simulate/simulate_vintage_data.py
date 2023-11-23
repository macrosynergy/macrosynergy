"""example/macrosynergy/management/simulate/simulate_vintage_data.py"""
from macrosynergy.management.simulate import VintageData

vins_m = VintageData(
    "USD_INDX_SA",
    cutoff="2019-06-30",
    release_lags=[3, 20, 25],
    number_firsts=12,
    shortest=12,
    sd_ar=5,
    trend_ar=20,
    seasonal=10,
    added_dates=6,
)

dfm1 = vins_m.make_grade1()

dfm1.groupby("release_date").agg(["mean", "count"])
dfm2 = vins_m.make_grade2()
dfmg = vins_m.make_graded(grading=[3, 2.1, 1], upgrades=[12, 24])

vins_q = VintageData(
    "USD_INDX_SA",
    release_lags=[3, 20, 25],
    number_firsts=2,
    shortest=8,
    freq="Q",
    seasonal=10,
    added_dates=4,
)
dfq1 = vins_q.make_grade1()
dfq1.groupby("release_date").agg(["mean", "count"])
dfq2 = vins_q.make_grade2()

vins_w = VintageData(
    "USD_INDX_SA",
    cutoff="2019-06-30",
    release_lags=[3, 20, 25],
    number_firsts=3 * 52,
    shortest=26,
    freq="W",
    seasonal=10,
    added_dates=52,
)
dfw1 = vins_w.make_grade1()
dfw1.groupby("release_date").agg(["mean", "count"])
dfw2 = vins_w.make_grade2()

dfm1.info()
