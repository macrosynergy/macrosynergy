"""Output-parity guard for the T2c targets (runs in the default gate; not marked perf)."""

import pandas as pd

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.types.qdf.methods import _get_tickers_series
from tests.perf.data import qdf_for_tier
from tests.perf.parity import assert_categorical_equal, assert_qdf_equal


def test_get_tickers_series_categorical_contract():
    qdf = qdf_for_tier("tiny", categorical=True)
    series = _get_tickers_series(qdf)
    # categories may be ordered by first appearance; rebuild the same way the function does
    labels = [f"{c}_{x}" for c, x in zip(
        qdf["cid"].cat.categories[qdf["cid"].cat.codes],
        qdf["xcat"].cat.categories[qdf["xcat"].cat.codes],
    )]
    cats = pd.unique(pd.Series(labels))
    expected = pd.Categorical(labels, categories=cats, ordered=True)
    assert_categorical_equal(pd.Categorical(series), expected)


def test_add_ticker_column_parity_object_vs_categorical():
    obj = qdf_for_tier("tiny", categorical=False)
    cat = QuantamentalDataFrame(obj.copy(), categorical=True)
    out_cat = cat.add_ticker_column()
    tickers_cat = [str(t) for t in out_cat["ticker"]]
    tickers_obj = [f"{c}_{x}" for c, x in zip(obj["cid"], obj["xcat"])]
    assert sorted(tickers_cat) == sorted(tickers_obj)


def test_reduce_df_by_ticker_parity():
    cat = qdf_for_tier("tiny", categorical=True)
    tickers = sorted({f"{c}_{x}" for c, x in zip(
        cat["cid"].astype(str), cat["xcat"].astype(str))})[:5]
    out = cat.reduce_df_by_ticker(tickers=tickers)
    got = sorted({f"{c}_{x}" for c, x in zip(out["cid"].astype(str), out["xcat"].astype(str))})
    assert got == sorted(tickers)
