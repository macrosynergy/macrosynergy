import pandas as pd
import numpy as np
import pytest

from macrosynergy.learning.preprocessing.imputers.imputers import (
    BaseImputer,
    DATE_INDEX_NAME,
    CIDS_INDEX_NAME,
)


@pytest.fixture
def data() -> pd.DataFrame:
    date_range = pd.date_range("2023-01-01", "2025-01-01", freq="ME")
    cids = ["CAD", "USD", "GBP", "AUS"]
    index = pd.MultiIndex.from_product(
        iterables=[cids, date_range], names=[CIDS_INDEX_NAME, DATE_INDEX_NAME]
    )

    data = pd.DataFrame(
        index=index,
        data=np.random.rand(len(index), 4),
        columns=["GROWTH", "CPI", "RIR", "XR"],
    )

    return data


@pytest.fixture
def nan_data(data: pd.DataFrame) -> pd.DataFrame:
    data[["GROWTH", "RIR"]] = np.nan
    return data


class DummyImputer(BaseImputer):
    def _fit_fill_values(self, X, y=None):
        self.seen_cols_ = list(X.columns)

    def _transform_with_fill_values(self, X):
        return X  # does nothing


def test_drops_threshold_violating_columns(nan_data):
    imputer = DummyImputer(nan_threshold=0.7)
    Xt = imputer.fit_transform(nan_data)

    assert imputer.dropped_features_ == ["GROWTH", "RIR"]
    assert imputer.kept_features_ == ["CPI", "XR"]
    assert list(imputer.feature_names_in_) == ["GROWTH", "CPI", "RIR", "XR"]
    assert imputer.n_features_out_ == 2
    assert imputer.seen_cols_ == ["CPI", "XR"]
    assert Xt.columns.tolist() == ["CPI", "XR"]
    assert Xt.shape == (nan_data.shape[0], 2)

    pd.testing.assert_frame_equal(Xt, nan_data[["CPI", "XR"]])

    pd.testing.assert_series_equal(
        left=pd.Series([1.0, 0.0, 1.0, 0.0], index=["GROWTH", "CPI", "RIR", "XR"]),
        right=imputer.missing_fraction_by_col_,
    )

    pd.testing.assert_frame_equal(
        left=pd.DataFrame(
            data={
                "GROWTH": [1.0, 1.0, 1.0, 1.0],
                "CPI": [0.0, 0.0, 0.0, 0.0],
                "RIR": [1.0, 1.0, 1.0, 1.0],
                "XR": [0.0, 0.0, 0.0, 0.0],
            },
            index=["AUS", "CAD", "GBP", "USD"],
        ),
        right=imputer.missing_fraction_by_cid_and_col_,
        check_names=False,
    )


def test_nothing_to_drop(data):
    imputer = DummyImputer()
    Xt = imputer.fit_transform(data)

    assert imputer.dropped_features_ == []
    assert imputer.kept_features_ == ["GROWTH", "CPI", "RIR", "XR"]
    assert list(imputer.feature_names_in_) == ["GROWTH", "CPI", "RIR", "XR"]
    assert imputer.n_features_out_ == 4
    assert imputer.seen_cols_ == ["GROWTH", "CPI", "RIR", "XR"]
    assert Xt.columns.tolist() == ["GROWTH", "CPI", "RIR", "XR"]
    assert Xt.shape == (data.shape[0], 4)

    pd.testing.assert_frame_equal(Xt, data)

    pd.testing.assert_series_equal(
        left=pd.Series([0.0] * 4, index=["GROWTH", "CPI", "RIR", "XR"]),
        right=imputer.missing_fraction_by_col_,
    )

    pd.testing.assert_frame_equal(
        left=pd.DataFrame(
            data={"GROWTH": [0] * 4, "CPI": [0] * 4, "RIR": [0] * 4, "XR": [0] * 4},
            index=["AUS", "CAD", "GBP", "USD"],
            dtype=float,
        ),
        right=imputer.missing_fraction_by_cid_and_col_,
        check_names=False,
    )


def test_unordered_columns(data):
    imputer = DummyImputer()
    imputer.fit(data)

    data = data[["XR", "CPI", "RIR", "GROWTH"]]
    with pytest.raises(ValueError):
        imputer.transform(data)


def test_invalid_index():
    X = pd.DataFrame({"a": [1.0], "b": [2.0]})
    imputer = DummyImputer()

    with pytest.raises(ValueError):
        imputer.fit(X)
