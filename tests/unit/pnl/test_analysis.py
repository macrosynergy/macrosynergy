from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from macrosynergy.management.simulate import SignalsAndReturnsGenerator
from macrosynergy.pnl.analysis import (
    _long_cov_to_dict,
    _resolve_config,
    cov_estimators_cost_accuracy,
    realized_to_forecast_vol_ratios,
    scaling_factor_bias_variance,
)
from macrosynergy.pnl.historic_portfolio_volatility import _cov_matrix_history
from macrosynergy.pnl.notional_positions import notional_positions
from macrosynergy.pnl.transaction_costs import TransactionCostsDictAdapter

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

TCOST_DICT = {
    "AUD_FX": {
        "bid_offer": {
            "size": {"median": 58.5, "pct90": 234.0},
            "cost": {"median": 0.0023, "pct90": 0.0048},
        },
        "rollcost": {
            "size": {"median": 58.5, "pct90": 234.0},
            "cost": {"median": 0.000864, "pct90": 0.00154},
        },
    },
    "CAD_FX": {
        "bid_offer": {
            "size": {"median": 29.3, "pct90": 117.0},
            "cost": {"median": 0.00594, "pct90": 0.0128},
        },
        "rollcost": {
            "size": {"median": 29.3, "pct90": 117.0},
            "cost": {"median": 0.00183, "pct90": 0.00343},
        },
    },
    "GBP_FX": {
        "bid_offer": {
            "size": {"median": 29.3, "pct90": 117.0},
            "cost": {"median": 0.00594, "pct90": 0.0128},
        },
        "rollcost": {
            "size": {"median": 29.3, "pct90": 117.0},
            "cost": {"median": 0.00183, "pct90": 0.00343},
        },
    },
}
TCOST = TransactionCostsDictAdapter(TCOST_DICT)


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
    params.setdefault("report_floor", False)
    params.update(overrides)
    return scaling_factor_bias_variance(**params)


COST_FIDS = ["AUD_FX", "CAD_FX", "GBP_FX"]
COST_CORR = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]])
COST_BASE_VOL = np.array([0.010, 0.020, 0.012])


def cost_accuracy_params(**overrides):
    params = dict(
        corr=COST_CORR,
        base_vol=COST_BASE_VOL,
        vol_persistence=0.94,
        vol_of_vol=0.15,
        configs=[MA_10D],
        n_periods=800,
        fid_names=COST_FIDS,
        n_iter=1,
        seed=7,
        signal_half_life=21,
        tcost_obj=TCOST,
    )
    params.update(overrides)
    return params


def run_cost_accuracy(**overrides):
    return cov_estimators_cost_accuracy(**cost_accuracy_params(**overrides))


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


class TestRatioComputation:
    def test_perfect_estimate_gives_ratio_one(self, truth_2x2):
        cov_true = cov_dict(("2020-01-01", truth_2x2))
        cov_est = cov_dict(("2020-01-01", truth_2x2))

        ratios = realized_to_forecast_vol_ratios(
            cov_true, [cov_est], weights=EQUAL_WEIGHTS
        )

        assert ratios[0, 0] == pytest.approx(1.0)

    def test_underestimated_vol_gives_ratio_above_one(self, truth_2x2):
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

    def test_per_date_eights(self, truth_2x2):
        cov_true = cov_dict(("2020-01-01", truth_2x2), ("2020-01-02", truth_2x2))
        cov_est = cov_dict(("2020-01-01", truth_2x2), ("2020-01-02", truth_2x2 / 4.0))
        repeated = {date: EQUAL_WEIGHTS for date in cov_true}

        as_array = realized_to_forecast_vol_ratios(
            cov_true, [cov_est], weights=EQUAL_WEIGHTS
        )
        as_mapping = realized_to_forecast_vol_ratios(
            cov_true, [cov_est], weights=repeated
        )

        np.testing.assert_allclose(as_array, as_mapping)

    def test_each_date_uses_its_own_weights(self):
        truth = np.diag([4.0, 1.0])
        est = np.diag([1.0, 4.0])
        cov_true = cov_dict(("2020-01-01", truth), ("2020-01-02", truth))
        cov_est = cov_dict(("2020-01-01", est), ("2020-01-02", est))
        per_date = {
            pd.Timestamp("2020-01-01"): np.array([1.0, 0.0]),
            pd.Timestamp("2020-01-02"): np.array([0.0, 1.0]),
        }

        ratios = realized_to_forecast_vol_ratios(cov_true, [cov_est], weights=per_date)

        np.testing.assert_allclose(ratios[:, 0], [2.0, 0.5])

    def test_date_missing_from_the_mapping_is_named(self, truth_2x2):
        cov_true = cov_dict(("2020-01-01", truth_2x2), ("2020-01-02", truth_2x2))
        cov_est = cov_dict(("2020-01-01", truth_2x2), ("2020-01-02", truth_2x2))
        incomplete = {pd.Timestamp("2020-01-01"): EQUAL_WEIGHTS}

        with pytest.raises(
            ValueError, match=r"missing 1 of the 2 dates .*first being 2020-01-02"
        ):
            realized_to_forecast_vol_ratios(cov_true, [cov_est], weights=incomplete)

    def test_wrong_length_vector_is_rejected(self, truth_2x2):
        cov_true = cov_dict(("2020-01-01", truth_2x2))
        cov_est = cov_dict(("2020-01-01", truth_2x2))
        wrong = {pd.Timestamp("2020-01-01"): np.array([1.0, 0.0, 0.0])}

        with pytest.raises(ValueError, match=r"2020-01-01.*3.*2"):
            realized_to_forecast_vol_ratios(cov_true, [cov_est], weights=wrong)

    def test_wrong_length_array_is_rejected(self, truth_2x2):
        cov_true = cov_dict(("2020-01-01", truth_2x2))
        cov_est = cov_dict(("2020-01-01", truth_2x2))

        with pytest.raises(ValueError, match=r"3.*2"):
            realized_to_forecast_vol_ratios(cov_true, [cov_est], weights=np.ones(3) / 3)


class TestScalingFactorBiasVariance:
    def test_lookback_longer_than_the_panel_errors(self):
        too_long = {"est_freqs": ["D"], "lback_meth": "ma", "lback_periods": [3000]}

        with pytest.raises(ValueError, match=r"No fid has enough history"):
            run_bias_variance(configs=[too_long])

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
        # note it will never be perfectly unbiased because of Jensen's inequality
        bias, _ = run_bias_variance(
            vol_of_vol=0.0, configs=[MA_60D], n_periods=800, n_iter=10
        )

        assert abs(bias[0]) < 0.06

    def test_std_shrinks_with_longer_lookback_when_data_stationary(self):
        _, std = run_bias_variance(
            vol_of_vol=0.0, configs=[MA_10D, MA_60D], n_periods=400, n_iter=4
        )

        assert std[0] > std[1]

    def test_combined_frequency_estimator_is_nearly_unbiased(self):
        # will never be completely unbiased due to Jensen's inequality
        bias, std = run_bias_variance(
            vol_of_vol=0.0,
            configs=[MA_DAILY_WEEKLY_MONTHLY],
            n_periods=800,
            n_iter=4,
        )

        assert np.isfinite(bias).all() and np.isfinite(std).all()
        assert abs(bias[0]) < 0.1

    # Test the common sample argument
    def test_slowest_config_is_unaffected_by_common_sample_flag(self):
        common, _ = run_bias_variance(configs=[MA_10D, MA_60D], n_periods=400, n_iter=4)
        per_config, _ = run_bias_variance(
            configs=[MA_10D, MA_60D], n_periods=400, n_iter=4, common_sample=False
        )

        assert common[1] == pytest.approx(per_config[1])

    def test_faster_config_loses_its_extra_dates_under_comon_sample(self):
        common, _ = run_bias_variance(configs=[MA_10D, MA_60D], n_periods=400, n_iter=4)
        per_config, _ = run_bias_variance(
            configs=[MA_10D, MA_60D], n_periods=400, n_iter=4, common_sample=False
        )

        assert common[0] != pytest.approx(per_config[0])

    # Test config resolution
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


class TestCostAccuracySimulation:
    def test_same_seed_reproduces_identical_results(self):
        costs1, vols1 = run_cost_accuracy(seed=11)
        costs2, vols2 = run_cost_accuracy(seed=11)

        np.testing.assert_array_almost_equal(costs1, costs2)
        np.testing.assert_array_almost_equal(vols1, vols2)

    def test_diff_seed_gives_different_results(self):
        costs1, vols1 = run_cost_accuracy(seed=11)
        costs2, vols2 = run_cost_accuracy(seed=22)

        assert not np.array_equal(costs1, costs2)
        assert not np.array_equal(vols1, vols2)

    def test_estimator_matches_cov_matrix_history(self):
        generator = SignalsAndReturnsGenerator(
            n_fids=3, corr=COST_CORR, base_vol=COST_BASE_VOL, half_life=21
        )
        generator.simulate_signals_and_returns(
            n_periods=800,
            signal_names=[f"{fid}_CSIG_STRAT" for fid in COST_FIDS],
            return_names=[f"{fid}XR" for fid in COST_FIDS],
            seed=5,
        )
        df = generator.quantamental_returns_and_signals()
        resolved = _resolve_config(MA_60D)

        _, vcv = notional_positions(
            df=df,
            sname="STRAT",
            fids=COST_FIDS,
            aum=100,
            slip=0,
            vol_target=10,
            rebal_freq="M",
            rstring="XR",
            nan_tolerance=0.0,
            remove_zeros=False,
            lback_meth=MA_60D["lback_meth"],
            est_freqs=resolved["est_freqs"],
            est_weights=resolved["est_weights"],
            lback_periods=resolved["lback_periods"],
            half_life=resolved["half_life"],
            return_vcv=True,
        )
        from_positions = _long_cov_to_dict(vcv, fids=COST_FIDS, check_psd=False)

        dates = pd.Series(sorted(from_positions.keys()))
        analytical = _cov_matrix_history(
            pivot_returns=100 * generator.returns,
            estimation_dates=dates,
            nan_tolerance=0,
            remove_zeros=False,
            **resolved,
        )

        assert len(from_positions) > 12
        for date, matrix in zip(dates, analytical):
            np.testing.assert_allclose(from_positions[date], matrix, rtol=1e-10)
