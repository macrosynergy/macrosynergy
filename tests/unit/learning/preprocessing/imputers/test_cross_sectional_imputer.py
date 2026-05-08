import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from macrosynergy.learning.preprocessing.imputers.imputers import CrossSectionalImputer


@pytest.fixture
def data():
    """
    Panel data with deliberate NaNs to test:
      - peer mean imputation
      - fallback to all-cids mean
      - interpolation fallback within cid
      - all-NaN column dropped by BaseImputer fit/transform (if applicable)
    """
    cids = ["CAD", "USD", "GBP", "EUR"]
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    index = pd.MultiIndex.from_product(
        [cids, dates], names=["cid", "real_date"]
    )
    data = pd.DataFrame(
        index=index,
        data={
            "feature_A": np.nan,
            "feature_B": np.nan,
            "feature_all_nan": np.nan,  # should be dropped by BaseImputer
        },
        dtype="float64",
    )

    # Convenience: (cid, date) -> value helper
    def set_val(cid, date, col, val):
        data.loc[(cid, pd.to_datetime(date)), col] = val

    # --- feature_A values ---
    # 2020-01-01: CAD is missing, USD=2, GBP=4, EUR=10
    # -> peer mean for CAD could be mean([2,4])=3
    set_val("USD", "2020-01-01", "feature_A", 2.0)
    set_val("GBP", "2020-01-01", "feature_A", 4.0)
    set_val("EUR", "2020-01-01", "feature_A", 10.0)

    # 2020-01-02: make peers missing but EUR present, so fallback all-cids mean works.
    # CAD missing, USD missing, GBP missing, EUR=8 -> all-cids mean = 8
    set_val("EUR", "2020-01-02", "feature_A", 8.0)

    # 2020-01-03: set CAD endpoints for interpolation test, and set some others
    set_val("CAD", "2020-01-01", "feature_A", 1.0)
    # CAD on 2020-01-02 will remain missing after cross-sectional steps
    # (if we also make all missing there), and should interpolate between
    # 1 and 3 if fallback="fill".
    set_val("CAD", "2020-01-03", "feature_A", 3.0)

    # Also populate some values for other cids on 2020-01-03 (not strictly required)
    set_val("USD", "2020-01-03", "feature_A", 6.0)
    set_val("GBP", "2020-01-03", "feature_A", 6.0)
    set_val("EUR", "2020-01-03", "feature_A", 6.0)

    # --- feature_B values ---
    # Set up a leading-missing pattern for CAD so interpolation limit_direction="both"
    # can bfill/fill
    # CAD missing on 2020-01-01, but has 5 on 2020-01-02 -> with interpolation both,
    # 2020-01-01 becomes 5
    set_val("CAD", "2020-01-02", "feature_B", 5.0)

    # For peers, add some values so peer mean has something to use
    set_val("USD", "2020-01-01", "feature_B", 1.0)
    set_val("GBP", "2020-01-01", "feature_B", 3.0)
    set_val("EUR", "2020-01-01", "feature_B", 5.0)

    return data


def test_peer_mean_imputation_basic(data):
    """
    If CAD feature_A is missing on 2020-01-01 and peers USD=2, GBP=4,
    fill with mean = 3 (EUR=10 should be ignored since not a peer).
    """
    # Make CAD missing on that date for feature_A (ensure missing)
    data.loc[("CAD", "2020-01-01"), "feature_A"] = np.nan

    peer_map = {"CAD": ["USD", "GBP"]}

    imputer = CrossSectionalImputer(peer_map=peer_map, fallback=None)
    out = imputer.fit_transform(data).sort_index()

    result_feature_A = out.loc[("CAD", "2020-01-01"), "feature_A"]
    expected_feature_A = pytest.approx(3.0)

    result_feature_B = out.loc[("CAD", "2020-01-01"), "feature_B"]
    expected_feature_B = pytest.approx(2.0)

    assert result_feature_A.item() == expected_feature_A
    assert result_feature_B.item() == expected_feature_B
    assert out.columns.tolist() == ["feature_A", "feature_B"]


def test_peer_map_filters_missing_peers(data):
    """
    If peer_map contains cids not present in the data, they should be ignored.
    """
    data.loc[("CAD", "2020-01-01"), "feature_A"] = np.nan

    imputer = CrossSectionalImputer(
        peer_map={"CAD": ["USD", "XXX", "GBP"]}, fallback=None
    )
    out = imputer.fit_transform(data).sort_index()

    result_feature_A = out.loc[("CAD", "2020-01-01"), "feature_A"]
    expected_feature_A = pytest.approx(3.0)

    result_feature_B = out.loc[("CAD", "2020-01-01"), "feature_B"]
    expected_feature_B = pytest.approx(2.0)

    assert result_feature_A.item() == expected_feature_A
    assert result_feature_B.item() == expected_feature_B
    assert out.columns.tolist() == ["feature_A", "feature_B"]


def test_default_peers_all_when_not_in_map(data):
    """
    If a cid isn't in peer_map and default_peers='all', it should use all other cids.
    We'll test EUR missing on 2020-01-01 for feature_A: use mean of CAD, USD, GBP on
    that date.
    """
    # Ensure values exist for CAD/USD/GBP on 2020-01-01
    data.loc[("CAD", "2020-01-01"), "feature_A"] = 1.0
    data.loc[("USD", "2020-01-01"), "feature_A"] = 2.0
    data.loc[("GBP", "2020-01-01"), "feature_A"] = 4.0

    # Make EUR missing at that date
    data.loc[("EUR", "2020-01-01"), "feature_A"] = np.nan

    imputer = CrossSectionalImputer(
        peer_map={"CAD": ["USD", "GBP"]},
        default_peers="all",
        fallback=None,
    )
    out = imputer.fit_transform(data).sort_index()

    result = out.loc[("EUR", "2020-01-01"), "feature_A"].item()
    expected = pytest.approx((1.0 + 2.0 + 4.0) / 3.0)

    assert result == expected


def test_default_peers_none_leaves_missing_when_no_fallback(data):
    """
    If cid not in peer_map and default_peers='none', it should not be imputed by peers
    at all and fallback='none' should leave it NaN.
    """
    data.loc[("EUR", pd.to_datetime("2020-01-01")), "feature_A"] = np.nan

    imputer = CrossSectionalImputer(
        peer_map={"CAD": ["USD", "GBP"]},
        default_peers=None,
        fallback=None,
    )
    out = imputer.fit_transform(data).sort_index()

    result = out.loc[("EUR", "2020-01-01"), "feature_A"].item()

    assert np.isnan(result)


def test_fallback_fill_uses_mean_when_peers_unavailable(data):
    """
    For feature_A on 2020-01-02:
      USD and GBP are missing, CAD is missing, EUR=8.
    If CAD peers are USD/GBP, peer mean is unavailable -> fallback should use
    column mean.
    """
    # Ensure CAD is missing on that date
    data.loc[("CAD", "2020-01-02"), "feature_A"] = np.nan

    # Ensure peers missing
    data.loc[("USD", "2020-01-02"), "feature_A"] = np.nan
    data.loc[("GBP", "2020-01-02"), "feature_A"] = np.nan

    # EUR is already set to 8 in fixture

    imputer = CrossSectionalImputer(
        peer_map={"CAD": ["USD", "GBP"]}, fallback="mean"
    )
    out = imputer.fit_transform(data).sort_index()

    result = out.loc[("CAD", "2020-01-02"), "feature_A"].item()
    expected = pytest.approx(data["feature_A"].mean())

    assert result == expected

def test_fallback_fill_uses_all_peer_mean_when_desired_peers_unavailable(data):
    """
    For feature_A on 2020-01-02:
      USD and GBP are missing, CAD is missing, EUR=8, CHF=12.
    If CAD peers are USD/GBP, peer mean is unavailable -> fallback should use
    EUR + CHF mean.

    Also, since default_peers=None, USD, GBP, should also be filled in with
    EUR + CHF mean at fallback time
    """
    # Ensure CAD is missing on that date
    data.loc[("CAD", "2020-01-02"), "feature_A"] = np.nan

    # Ensure peers missing
    data.loc[("USD", "2020-01-02"), "feature_A"] = np.nan
    data.loc[("GBP", "2020-01-02"), "feature_A"] = np.nan

    # EUR is already set to 8 in fixture
    data.loc[("CHF", "2020-01-02"), "feature_A"] = 12

    imputer = CrossSectionalImputer(
        peer_map={"CAD": ["USD", "GBP"]},
        fallback="all_peer_mean",
        default_peers=None,
    )
    out = imputer.fit_transform(data).sort_index()

    assert out.loc[("CAD", "2020-01-02"), "feature_A"].item() == pytest.approx(10)
    assert out.loc[("USD", "2020-01-02"), "feature_A"].item() == pytest.approx(10)
    assert out.loc[("GBP", "2020-01-02"), "feature_A"].item() == pytest.approx(10)
    assert out.loc[("EUR", "2020-01-02"), "feature_A"].item() == pytest.approx(8)
    assert out.loc[("CHF", "2020-01-02"), "feature_A"].item() == pytest.approx(12)

def test_invalid_args():
    with pytest.raises(ValueError):
        CrossSectionalImputer(default_peers="something_wrong")

    with pytest.raises(ValueError):
        CrossSectionalImputer(fallback="something_wrong")


def test_transform_raises_if_not_fitted(data):
    imputer = CrossSectionalImputer()
    with pytest.raises(NotFittedError):
        imputer.transform(data)
