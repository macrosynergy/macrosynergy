import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from macrosynergy.learning.preprocessing.imputers.imputers import ConstantImputer


@pytest.fixture
def dates():
    return pd.to_datetime(["2020-01-01", "2020-01-02"])


def _panel_df(values_by_cid, dates, columns):
    """
    Build a MultiIndex (cid, real_date) DataFrame from a dict:
      values_by_cid = {"A": [[...row per date...], [...]], "B": ...}
    """
    tuples, rows = [], []
    for cid, rows_for_cid in values_by_cid.items():
        assert len(rows_for_cid) == len(dates)
        for d, row in zip(dates, rows_for_cid):
            tuples.append((cid, d))
            rows.append(row)

    idx = pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"])
    return pd.DataFrame(rows, index=idx, columns=list(columns))


def test_transform_raises_if_not_fitted(dates):
    X = _panel_df(values_by_cid={"A": [[1.0], [2.0]]}, dates=dates, columns=("xcat1",))
    imp = ConstantImputer(fill_value=0)
    with pytest.raises(NotFittedError):
        imp.transform(X)


def test_replaces_nans_with_fill_value_float(dates):
    X = _panel_df(
        values_by_cid={"A": [[np.nan], [2.0]], "B": [[3.0], [np.nan]]},
        dates=dates,
        columns=("xcat1",),
    )

    imp = ConstantImputer(fill_value=-1.5)
    Xt = imp.fit_transform(X)

    assert Xt.loc[("A", dates[0]), "xcat1"] == pytest.approx(-1.5)
    assert Xt.loc[("A", dates[1]), "xcat1"] == pytest.approx(2.0)
    assert Xt.loc[("B", dates[0]), "xcat1"] == pytest.approx(3.0)
    assert Xt.loc[("B", dates[1]), "xcat1"] == pytest.approx(-1.5)


def test_does_not_overwrite_existing_values(dates):
    X = _panel_df(
        values_by_cid={"A": [[1.0], [2.0]], "B": [[3.0], [4.0]]},
        dates=dates,
        columns=("xcat1",),
    )
    imp = ConstantImputer(fill_value=999)
    Xt = imp.fit_transform(X)

    pd.testing.assert_frame_equal(Xt, X)


def test_nan_threshold_drops_all_nan_columns(dates):
    X = _panel_df(
        values_by_cid={
            "A": [[1.0, np.nan], [2.0, np.nan]],
            "B": [[3.0, np.nan], [4.0, np.nan]],
        },
        dates=dates,
        columns=("xcat1", "xcat_all_nan"),
    )
    imp = ConstantImputer(fill_value=0, nan_threshold=1.0)
    Xt = imp.fit_transform(X)

    assert Xt.columns.tolist() == ["xcat1"]
    assert imp.get_feature_names_out().tolist() == ["xcat1"]


def test_nan_threshold_drops_columns_at_or_above_threshold(dates):
    # 4 rows total per column (2 cids x 2 dates)
    # x has 2 NaNs -> missing fraction 0.5
    # y has 1 NaN  -> missing fraction 0.25
    X = _panel_df(
        values_by_cid={
            "A": [[np.nan, 10.0], [2.0, np.nan]],
            "B": [[np.nan, 30.0], [4.0, 40.0]],
        },
        dates=dates,
        columns=("xcat1", "xcat2"),
    )

    imp = ConstantImputer(fill_value=0, nan_threshold=0.5)
    Xt = imp.fit_transform(X)

    # because violations are computed with ">="
    assert Xt.columns.tolist() == ["xcat2"]

    # remaining y NaN replaced
    assert Xt.loc[("A", dates[1]), "xcat2"] == pytest.approx(0.0)


def test_fit_stores_diagnostics_attributes(dates):
    X = _panel_df(
        values_by_cid={
            "A": [[np.nan, 1.0], [2.0, np.nan]],
            "B": [[3.0, 4.0], [np.nan, 6.0]],
        },
        dates=dates,
        columns=("xcat1", "xcat2"),
    )
    imp = ConstantImputer(fill_value=0, nan_threshold=1.0)
    imp.fit(X)

    # These are explicitly set during fit :contentReference[oaicite:5]{index=5}
    assert hasattr(imp, "feature_names_in_")
    assert hasattr(imp, "missing_fraction_by_col_")
    assert hasattr(imp, "missing_fraction_by_cid_and_col_")
    assert hasattr(imp, "dropped_features_")
    assert hasattr(imp, "kept_features_")
    assert hasattr(imp, "n_features_out_")

    # sanity checks
    assert list(imp.feature_names_in_) == ["xcat1", "xcat2"]
    assert list(imp.kept_features_) == ["xcat1", "xcat2"]
    assert imp.n_features_out_ == 2


def test_transform_requires_same_column_order_as_fit(dates):
    X_fit = _panel_df(
        {"A": [[1.0, 10.0], [2.0, 20.0]], "B": [[3.0, 30.0], [4.0, 40.0]]},
        dates,
        columns=("xcat1", "xcat2"),
    )
    X_bad = X_fit[["xcat2", "xcat1"]]  # swapped order

    imp = ConstantImputer(fill_value=0)
    imp.fit(X_fit)

    with pytest.raises(ValueError, match="Input columns differ from fit-time columns"):
        imp.transform(X_bad)


def test_validate_input_requires_dataframe(dates):
    imp = ConstantImputer()
    with pytest.raises(TypeError):
        imp.fit([1, 2, 3])  # not a DataFrame


def test_fit_transform_returns_only_kept_features(dates):
    # Column z is all NaN -> will be dropped, so output should not include it
    X = _panel_df(
        {"A": [[np.nan, np.nan], [2.0, np.nan]], "B": [[3.0, np.nan], [4.0, np.nan]]},
        dates,
        columns=("x", "z"),
    )
    imp = ConstantImputer(fill_value=0, nan_threshold=1.0)
    Xt = imp.fit_transform(X)

    assert list(Xt.columns) == ["x"]
    assert (
        Xt.isna().sum().sum() == 0
    )  # all remaining NaNs in kept cols should be filled
