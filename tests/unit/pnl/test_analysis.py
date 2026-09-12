import itertools
import logging
from typing import Dict, List
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from macrosynergy.management.simulate import SignalsAndReturnsGenerator
from macrosynergy.pnl.analysis import (
    _long_cov_to_dict,
    _resolve_config,
    cov_estimators_bias_variance,
    realized_to_forecast_vol_ratios,
)

FIDS = ["AUD_FX", "CAD_FX"]
EQUAL_WEIGHTS = np.array([0.5, 0.5])

MA_10D = {"est_freqs": ["D"], "lback_meth": "ma", "lback_periods": [10]}
MA_60D = {"est_freqs": ["D"], "lback_meth": "ma", "lback_periods": [60]}
MA_MONTHLY_16 = {"est_freqs": ["M"], "lback_meth": "ma", "lback_periods": [16]}
MA_DAILY_WEEKLY_MONTHLY = {
    "est_freqs": ["D", "W", "M"],
    "est_weights": [1 / 3, 1 / 3, 1 / 3],
    "lback_meth": "ma",
    "lback_periods": [60, 24, 12],
}


@pytest.fixture
def truth_2x2() -> np.ndarray:
    return np.array([[4.0, 1.0], [1.0, 9.0]])

@pytest.fixture
def cov_2x2() -> np.ndarray:
    return np.array([[4.0, 1.0], [1.0, 9.0]])

def make_long_cov(matrix: np.ndarray, fids: List[str], date: str) -> pd.DataFrame:
    """Long-format upper triangle of a covariance matrix."""
    rows = [
        {
            "fid1": fids[i],
            "fid2": fids[j],
            "real_date": pd.Timestamp(date),
            "value": matrix[i, j],
        }
        for i in range(len(fids))
        for j in range(i, len(fids))
    ]
    return pd.DataFrame(rows)


def cov_dict(*dated_matrices) -> Dict[pd.Timestamp, np.ndarray]:
    """Date-keyed covariance matrices, the form `realized_to_forecast_vol_ratios` takes."""
    return {
        pd.Timestamp(date): np.asarray(matrix, dtype=float)
        for date, matrix in dated_matrices
    }

def run_bias_variance(**overrides):
    params = dict(
        corr=np.array([[1.0, 0.3], [0.3, 1.0]]),
        base_vol=np.array([0.010, 0.020]),
        vol_persistence=0.94,
        vol_of_vol=0.15,
        configs=[MA_10D],
        n_periods=280,
        fid_names=["FID1", "FID2"],
        n_iter=2,
        seed=7,
    )
    params.update(overrides)
    return cov_estimators_bias_variance(**params)



class TestMatrixConstruction:
    def test_single_date_returns_dict_keyed_by_date(self, cov_2x2):
        cov_long = make_long_cov(cov_2x2, ["AUD_FX", "CAD_FX"], "2020-01-01")

        result = _long_cov_to_dict(cov_long)

        assert list(result.keys()) == [pd.Timestamp("2020-01-01")]

    def test_matrix_values_match_input(self, cov_2x2):
        cov_long = make_long_cov(cov_2x2, ["AUD_FX", "CAD_FX"], "2020-01-01")

        result = _long_cov_to_dict(cov_long)

        np.testing.assert_allclose(result[pd.Timestamp("2020-01-01")], cov_2x2)

    def test_lower_triangle_mirrored_from_upper(self):
        cov = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
        cov_long = make_long_cov(cov, ["A_FX", "B_FX", "C_FX"], "2020-01-01")

        result = _long_cov_to_dict(cov_long)

        matrix = result[pd.Timestamp("2020-01-01")]
        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(matrix, cov)

    def test_single_fid_gives_1x1_matrix(self):
        cov_long = pd.DataFrame(
            {
                "fid1": ["AUD_FX"],
                "fid2": ["AUD_FX"],
                "real_date": [pd.Timestamp("2020-01-01")],
                "value": [4.0],
            }
        )

        result = _long_cov_to_dict(cov_long)

        np.testing.assert_allclose(
            result[pd.Timestamp("2020-01-01")], np.array([[4.0]])
        )

    def test_multiple_dates_give_one_matrix_each(self, cov_2x2):
        fids = ["AUD_FX", "CAD_FX"]
        cov_later = 2 * cov_2x2
        cov_long = pd.concat(
            [
                make_long_cov(cov_2x2, fids, "2020-01-01"),
                make_long_cov(cov_later, fids, "2020-01-02"),
            ]
        )

        result = _long_cov_to_dict(cov_long)

        assert len(result) == 2
        np.testing.assert_allclose(result[pd.Timestamp("2020-01-01")], cov_2x2)
        np.testing.assert_allclose(result[pd.Timestamp("2020-01-02")], cov_later)


class TestFidOrdering:
    def test_fids_inferred_in_sorted_order(self, cov_2x2):
        # data lists CAD before AUD; sorted order puts AUD at position 0
        cov_long = make_long_cov(cov_2x2, ["CAD_FX", "AUD_FX"], "2020-01-01")

        result = _long_cov_to_dict(cov_long)

        matrix = result[pd.Timestamp("2020-01-01")]
        assert matrix[0, 0] == pytest.approx(9.0)  # AUD_FX variance
        assert matrix[1, 1] == pytest.approx(4.0)  # CAD_FX variance

    def test_explicit_fids_control_matrix_positions(self, cov_2x2):
        cov_long = make_long_cov(cov_2x2, ["AUD_FX", "CAD_FX"], "2020-01-01")

        result = _long_cov_to_dict(cov_long, fids=["CAD_FX", "AUD_FX"])

        matrix = result[pd.Timestamp("2020-01-01")]
        assert matrix[0, 0] == pytest.approx(9.0)  # CAD_FX variance
        assert matrix[1, 1] == pytest.approx(4.0)  # AUD_FX variance


class TestValidation:
    def test_missing_columns_raise_value_error(self):
        cov_long = pd.DataFrame({"fid1": ["A_FX"], "value": [1.0]})

        with pytest.raises(ValueError, match=r"missing columns.*fid2.*real_date"):
            _long_cov_to_dict(cov_long)

    def test_missing_pair_entries_raise_value_error(self):
        # diagonal only: the off-diagonal pair is never provided
        cov_long = pd.DataFrame(
            {
                "fid1": ["AUD_FX", "CAD_FX"],
                "fid2": ["AUD_FX", "CAD_FX"],
                "real_date": pd.Timestamp("2020-01-01"),
                "value": [4.0, 9.0],
            }
        )

        with pytest.raises(ValueError, match="2 unfilled entries"):
            _long_cov_to_dict(cov_long)


class TestRatioComputation:

    def test_perfect_estimate_gives_ratio_one(self, truth_2x2):
        cov_true = cov_dict(("2020-01-01", truth_2x2))
        cov_est = cov_dict(("2020-01-01", truth_2x2))

        ratios = realized_to_forecast_vol_ratios(
            cov_true, [cov_est], weights=EQUAL_WEIGHTS
        )

        assert ratios[0, 0] == pytest.approx(1.0)

    def test_underestimated_vol_gives_ratio_above_one(self, truth_2x2):
        # estimate has a quarter of the true covariance: positions are scaled
        # 2x too large, so realized vol overshoots the target by 2x
        cov_true = cov_dict(("2020-01-01", truth_2x2))
        cov_est = cov_dict(("2020-01-01", truth_2x2 / 4.0))

        ratios = realized_to_forecast_vol_ratios(
            cov_true, [cov_est], weights=EQUAL_WEIGHTS
        )

        assert ratios[0, 0] == pytest.approx(2.0)

    def test_each_estimator_gets_its_own_column(self, truth_2x2):
        cov_true = cov_dict(("2020-01-01", truth_2x2))
        perfect_est = cov_dict(("2020-01-01", truth_2x2))
        quarter_est = cov_dict(("2020-01-01", truth_2x2 / 4.0))

        ratios = realized_to_forecast_vol_ratios(
            cov_true, [perfect_est, quarter_est], weights=EQUAL_WEIGHTS
        )

        np.testing.assert_allclose(ratios[0], [1.0, 2.0])

    def test_multiple_dates_give_one_ratio_per_date(self, truth_2x2):
        cov_true = cov_dict(("2020-01-01", truth_2x2), ("2020-01-02", truth_2x2))
        cov_est = cov_dict(("2020-01-01", truth_2x2), ("2020-01-02", truth_2x2 / 4.0))

        ratios = realized_to_forecast_vol_ratios(
            cov_true, [cov_est], weights=EQUAL_WEIGHTS
        )

        # rows follow sorted true dates: 2020-01-01 -> row 0, 2020-01-02 -> row 1
        np.testing.assert_allclose(ratios[:, 0], [1.0, 2.0])

    def test_rows_are_aligned_by_date_across_estimators(self, truth_2x2):
        # est covering only the second date must land in row 1, leaving row 0 NaN
        cov_true = cov_dict(("2020-01-01", truth_2x2), ("2020-01-02", truth_2x2))
        second_date_only = cov_dict(("2020-01-02", truth_2x2 / 4.0))

        ratios = realized_to_forecast_vol_ratios(
            cov_true, [second_date_only], weights=EQUAL_WEIGHTS
        )

        assert np.isnan(ratios[0, 0])
        assert ratios[1, 0] == pytest.approx(2.0)


class TestWeights:
    def test_default_weights_are_min_variance_from_truth(self):
        # truth diag(1, 4) -> min-var weights [0.8, 0.2], so the ratio is
        # sqrt(0.8 / 2.6). Weights derived from the estimate (diag(4, 1) ->
        # [0.2, 0.8]) would give sqrt(2.6 / 0.8) instead: the optimizer picks
        # what looks cheapest under its own noise, biasing the denominator low.
        est = np.diag([4.0, 1.0])
        truth = np.diag([1.0, 4.0])
        cov_true = cov_dict(("2020-01-01", truth))
        cov_est = cov_dict(("2020-01-01", est))

        ratios = realized_to_forecast_vol_ratios(cov_true, [cov_est])

        assert ratios[0, 0] == pytest.approx(np.sqrt(0.8 / 2.6))

    def test_default_weights_shared_across_estimators(self):
        # truth diag(4, 1) -> min-var weights [0.2, 0.8] for every estimator:
        # est_a diag(1, 4) gives sqrt(0.8 / 2.6), a perfect est_b gives 1.0.
        # Estimate-derived weights would give sqrt(2.6 / 0.8) for est_a.
        truth = np.diag([4.0, 1.0])
        cov_true = cov_dict(("2020-01-01", truth))
        est_a = cov_dict(("2020-01-01", np.diag([1.0, 4.0])))
        est_b = cov_dict(("2020-01-01", truth))

        ratios = realized_to_forecast_vol_ratios(cov_true, [est_a, est_b])

        np.testing.assert_allclose(ratios[0], [np.sqrt(0.8 / 2.6), 1.0])


class TestUnestimableConfig:
    def test_lookback_longer_than_the_panel_is_reported_clearly(self):
        too_long = {"est_freqs": ["D"], "lback_meth": "ma", "lback_periods": [3000]}

        with pytest.raises(ValueError, match=r"No fid has enough history"):
            run_bias_variance(configs=[too_long])

    def test_the_failing_config_is_named(self):
        too_long = {"est_freqs": ["D"], "lback_meth": "ma", "lback_periods": [3000]}

        with pytest.raises(ValueError, match=r"config at index 1"):
            run_bias_variance(configs=[MA_10D, too_long])


class TestCommonSample:
    """Configs warm up at different rates, so they are scored on shared dates."""

    def test_slowest_config_is_unaffected_by_the_flag(self):
        common, _ = run_bias_variance(configs=[MA_10D, MA_60D], n_periods=400, n_iter=4)
        per_config, _ = run_bias_variance(
            configs=[MA_10D, MA_60D], n_periods=400, n_iter=4, common_sample=False
        )

        assert common[1] == pytest.approx(per_config[1])

    def test_faster_config_loses_its_extra_dates(self):
        # the 10-day estimator is live on dates the 60-day one is not, and those are
        # exactly the dates the common sample drops
        common, _ = run_bias_variance(configs=[MA_10D, MA_60D], n_periods=400, n_iter=4)
        per_config, _ = run_bias_variance(
            configs=[MA_10D, MA_60D], n_periods=400, n_iter=4, common_sample=False
        )

        assert common[0] != pytest.approx(per_config[0])


    def test_evenly_covered_configs_do_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="macrosynergy.pnl.analysis"):
            run_bias_variance(configs=[MA_10D, MA_10D], n_periods=400, n_iter=2)

        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


class TestConfigResolution:

    def test_lback_meth_is_case_insensitive(self):
        upper, _ = run_bias_variance(configs=[{**MA_10D, "lback_meth": "MA"}])
        lower, _ = run_bias_variance(configs=[MA_10D])

        assert upper[0] == pytest.approx(lower[0])

    def test_unknown_lback_meth_is_rejected(self):
        with pytest.raises(ValueError, match=r"must be 'ma' or 'xma'"):
            run_bias_variance(configs=[{**MA_10D, "lback_meth": "ewma"}])

    def test_single_lookback_expands_across_frequencies(self):
        resolved = _resolve_config(
            {"est_freqs": ["D", "W", "M"], "lback_meth": "ma", "lback_periods": [60]}
        )

        assert resolved["lback_periods"] == [60, 60, 60]
        assert resolved["est_weights"] == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    def test_missing_lookback_names_the_key(self):
        with pytest.raises(ValueError, match=r"missing \['lback_periods'\]"):
            run_bias_variance(configs=[{"est_freqs": ["D"], "lback_meth": "ma"}])

    def test_exponential_weights_require_an_explicit_half_life(self):
        with pytest.raises(ValueError, match=r"needs an explicit `half_life`"):
            run_bias_variance(
                configs=[
                    {"est_freqs": ["D"], "lback_meth": "xma", "lback_periods": [60]}
                ]
            )

    def test_invalid_frequency_is_rejected(self):
        with pytest.raises(ValueError, match=r"est_freq\[0\]"):
            run_bias_variance(configs=[{**MA_10D, "est_freqs": ["Daily"]}])

    def test_configs_are_checked_before_any_simulation(self):
        with mock.patch.object(
            SignalsAndReturnsGenerator, "simulate_signals_and_returns"
        ) as simulate:
            with pytest.raises(ValueError, match=r"unknown key"):
                run_bias_variance(configs=[MA_10D, {**MA_10D, "lbak_periods": [5]}])

        simulate.assert_not_called()


class TestCovEstimatorsBiasVariance:
    def test_returns_one_bias_and_std_per_config(self):
        bias, std = run_bias_variance(configs=[MA_10D, MA_60D])

        assert bias.shape == (2,)
        assert std.shape == (2,)

    def test_same_seed_reproduces_identical_results(self):
        bias_a, std_a = run_bias_variance(seed=11)
        bias_b, std_b = run_bias_variance(seed=11)

        np.testing.assert_array_equal(bias_a, bias_b)
        np.testing.assert_array_equal(std_a, std_b)

    def test_different_seeds_produce_different_results(self):
        bias_a, _ = run_bias_variance(seed=11)
        bias_b, _ = run_bias_variance(seed=12)

        assert not np.array_equal(bias_a, bias_b)

    def test_identical_configs_get_identical_columns(self):
        bias, std = run_bias_variance(configs=[MA_10D, MA_10D])

        assert bias[0] == pytest.approx(bias[1])
        assert std[0] == pytest.approx(std[1])

    def test_long_lookback_is_nearly_unbiased_when_vol_is_constant(self):
        bias, _ = run_bias_variance(
            vol_of_vol=0.0, configs=[MA_60D], n_periods=800, n_iter=4
        )

        assert abs(bias[0]) < 0.06

    def test_dispersion_shrinks_with_longer_lookback(self):
        _, std = run_bias_variance(
            vol_of_vol=0.0, configs=[MA_10D, MA_60D], n_periods=400, n_iter=4
        )

        assert std[0] > std[1]

    def test_combined_frequency_estimator_is_nearly_unbiased(self):
        bias, std = run_bias_variance(
            vol_of_vol=0.0,
            configs=[MA_DAILY_WEEKLY_MONTHLY],
            n_periods=800,
            n_iter=4,
        )

        assert np.isfinite(bias).all() and np.isfinite(std).all()
        assert abs(bias[0]) < 0.1

    def test_short_lookback_underforecasts_vol_more(self):
        bias, _ = run_bias_variance(
            vol_of_vol=0.0, configs=[MA_10D, MA_60D], n_periods=400, n_iter=4
        )

        assert bias[0] < bias[1]
