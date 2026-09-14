import itertools
import logging
from typing import Dict, List
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from macrosynergy.pnl import analysis
from macrosynergy.management.simulate import SignalsAndReturnsGenerator
from macrosynergy.pnl.analysis import (
    _long_cov_to_dict,
    _min_var_weights,
    _resolve_config,
    cov_estimators_cost_accuracy,
    scaling_factor_bias_variance,
    realized_to_forecast_vol_ratios,
    uniform_cost_object,
)
from macrosynergy.pnl.historic_portfolio_volatility import _cov_matrix_history
from macrosynergy.pnl.notional_positions import notional_positions

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
    params.setdefault("report_floor", False)
    params.update(overrides)
    result = scaling_factor_bias_variance(**params)
    return result["bias"].to_numpy(), result["std"].to_numpy()


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
    )
    params.update(overrides)
    return params


def run_cost_accuracy(**overrides):
    return cov_estimators_cost_accuracy(**cost_accuracy_params(**overrides))


def pooled_vol_bias(**params):
    """
    `1 - sqrt(E[r^2])` per config, the moment a full-sample realized vol measures.

    `scaling_factor_bias_variance` reports `1 - E[r]`, the mean of the per-interval
    ratios. A PnL's annualized standard deviation pools every day into one number, which
    is a quadratic mean, so it converges on `sqrt(E[r^2])` instead. The two differ by
    Jensen and the gap widens with estimator noise - 0.06 at a 10-day lookback against
    0.001 at 60 - so the traded harness has to be read against this moment, not the one
    its companion returns.
    """
    captured = []
    real = analysis.realized_to_forecast_vol_ratios

    def recorder(**kwargs):
        out = real(**kwargs)
        captured.append(out)
        return out

    with mock.patch.object(analysis, "realized_to_forecast_vol_ratios", recorder):
        analysis.scaling_factor_bias_variance(report_floor=False, **params)

    ratios = np.vstack(captured)
    ratios = ratios[np.isfinite(ratios).all(axis=1)]  # the common sample
    return 1 - np.sqrt((ratios**2).mean(axis=0))


def captured_panels(fn, **params):
    """Every simulated return panel a harness draws, in order."""
    panels = []
    real = SignalsAndReturnsGenerator.simulate_signals_and_returns

    def recorder(self, *args, **kwargs):
        out = real(self, *args, **kwargs)
        panels.append(self.returns.to_numpy().copy())
        return out

    with mock.patch.object(
        SignalsAndReturnsGenerator, "simulate_signals_and_returns", recorder
    ):
        fn(**params)
    return panels



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
    """Weights are the caller's to choose: the function never derives them from truth."""

    def test_weights_are_required(self, truth_2x2):
        # deriving weights from `cov_true` is lookahead - `w` minimising the
        # numerator of the ratio biases it by an amount that depends on how noisy
        # the estimator in the denominator is, which inverts the ranking
        cov_true = cov_dict(("2020-01-01", truth_2x2))
        cov_est = cov_dict(("2020-01-01", truth_2x2))

        with pytest.raises(TypeError, match=r"weights"):
            realized_to_forecast_vol_ratios(cov_true, [cov_est])

    def test_min_variance_weights_are_shared_across_estimators(self):
        # truth diag(4, 1) -> min-var weights [0.2, 0.8] for every estimator:
        # est_a diag(1, 4) gives sqrt(0.8 / 2.6), a perfect est_b gives 1.0.
        # Estimate-derived weights would give sqrt(2.6 / 0.8) for est_a.
        truth = np.diag([4.0, 1.0])
        cov_true = cov_dict(("2020-01-01", truth))
        est_a = cov_dict(("2020-01-01", np.diag([1.0, 4.0])))
        est_b = cov_dict(("2020-01-01", truth))

        ratios = realized_to_forecast_vol_ratios(
            cov_true, [est_a, est_b], weights=_min_var_weights(truth)
        )

        np.testing.assert_allclose(ratios[0], [np.sqrt(0.8 / 2.6), 1.0])


class TestPerDateWeights:
    """`weights` also takes one vector per date, the form the signal path needs."""

    def test_mapping_repeating_one_vector_matches_the_array_form(self, truth_2x2):
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
        # identical covariances on both dates, so any difference between the rows
        # can only come from the weights
        truth = np.diag([4.0, 1.0])
        est = np.diag([1.0, 4.0])
        cov_true = cov_dict(("2020-01-01", truth), ("2020-01-02", truth))
        cov_est = cov_dict(("2020-01-01", est), ("2020-01-02", est))
        per_date = {
            pd.Timestamp("2020-01-01"): np.array([1.0, 0.0]),
            pd.Timestamp("2020-01-02"): np.array([0.0, 1.0]),
        }

        ratios = realized_to_forecast_vol_ratios(cov_true, [cov_est], weights=per_date)

        # date 1 holds only the first contract: sqrt(4 / 1); date 2 only the second
        np.testing.assert_allclose(ratios[:, 0], [2.0, 0.5])

    def test_date_missing_from_the_mapping_is_named(self, truth_2x2):
        cov_true = cov_dict(("2020-01-01", truth_2x2), ("2020-01-02", truth_2x2))
        cov_est = cov_dict(("2020-01-01", truth_2x2), ("2020-01-02", truth_2x2))
        incomplete = {pd.Timestamp("2020-01-01"): EQUAL_WEIGHTS}

        # the counts have to read as what they are: one date short of two, not
        # "covers none of 1 of the 2"
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
            realized_to_forecast_vol_ratios(
                cov_true, [cov_est], weights=np.ones(3) / 3
            )


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


class TestSignalWeighting:
    """The portfolio scored is the one the DGP's own signals imply."""

    def _first_panel_signal_weights(self, **overrides):
        """Rebuild iteration 0's signal weights from the documented seeding protocol."""
        params = dict(
            corr=np.array([[1.0, 0.3], [0.3, 1.0]]),
            base_vol=np.array([0.010, 0.020]),
            vol_persistence=0.94,
            vol_of_vol=0.15,
            n_periods=280,
            fid_names=["FID1", "FID2"],
            seed=7,
            signal_half_life=21,
        )
        params.update(overrides)

        first_seed = analysis._iteration_seeds(seed=params["seed"], n_iter=2)[0]
        generator = SignalsAndReturnsGenerator(
            n_fids=2,
            corr=params["corr"],
            base_vol=params["base_vol"],
            vol_persistence=params["vol_persistence"],
            vol_of_vol=params["vol_of_vol"],
            half_life=params["signal_half_life"],
        )
        generator.simulate_signals_and_returns(
            n_periods=params["n_periods"],
            signal_names=[f"{fid}SIG" for fid in params["fid_names"]],
            return_names=[f"{fid}XR" for fid in params["fid_names"]],
            seed=first_seed,
            end_date=analysis.DEFAULT_END_DATE,
        )
        cov_true = generator.realized_cov(long=False)
        return {date: generator.signals.loc[date].to_numpy() for date in cov_true}

    def test_default_weights_are_the_simulated_signals(self):
        expected = self._first_panel_signal_weights()

        with mock.patch.object(
            analysis,
            "realized_to_forecast_vol_ratios",
            wraps=analysis.realized_to_forecast_vol_ratios,
        ) as ratios:
            run_bias_variance(signal_half_life=21)

        passed = ratios.call_args_list[0].kwargs["weights"]
        assert list(passed.keys()) == list(expected.keys())
        for date, weight in expected.items():
            np.testing.assert_allclose(passed[date], weight)

    def test_explicit_weights_override_the_signals(self):
        with mock.patch.object(
            analysis,
            "realized_to_forecast_vol_ratios",
            wraps=analysis.realized_to_forecast_vol_ratios,
        ) as ratios:
            run_bias_variance(weights=EQUAL_WEIGHTS)

        np.testing.assert_allclose(
            ratios.call_args_list[0].kwargs["weights"], EQUAL_WEIGHTS
        )

    def test_signal_half_life_moves_the_weighting_but_not_the_panel(self):
        # returns and volatilities are drawn before the signal is built, so a change
        # of half life must move the signal-weighted result and nothing else
        signal_short, _ = run_bias_variance(signal_half_life=5)
        signal_long, _ = run_bias_variance(signal_half_life=63)
        fixed_short, _ = run_bias_variance(signal_half_life=5, weights=EQUAL_WEIGHTS)
        fixed_long, _ = run_bias_variance(signal_half_life=63, weights=EQUAL_WEIGHTS)

        assert signal_short[0] != pytest.approx(signal_long[0])
        assert fixed_short[0] == pytest.approx(fixed_long[0])

    def test_short_lookback_is_the_most_biased_config(self):
        # the ranking test. Weights derived from the forward truth reverse this:
        # they score MA_10D as the best calibrated of the set
        bias, _ = run_bias_variance(
            configs=[MA_10D, MA_60D, MA_DAILY_WEEKLY_MONTHLY],
            n_periods=800,
            n_iter=4,
        )

        assert abs(bias[0]) > abs(bias[1])
        assert abs(bias[0]) > abs(bias[2])


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


class TestCostAccuracySimulation:
    """The cost harness generates its own panel, from the same DGP as its companion."""

    def test_returns_one_labelled_row_per_config(self):
        result = run_cost_accuracy(configs=[MA_10D, MA_60D])

        assert list(result.index) == [0, 1]
        assert list(result.columns) == [
            "cost_pct_aum_yr",
            "cost_per_unit_risk",
            "realized_vol_pct",
        ]

    def test_same_seed_reproduces_identical_results(self):
        pd.testing.assert_frame_equal(
            run_cost_accuracy(seed=11), run_cost_accuracy(seed=11)
        )

    def test_panels_match_the_analytical_harness_seed_for_seed(self):
        # the property the whole comparison rests on: iteration i is the same panel in
        # both harnesses, so a config's forecast error and its cost describe one world
        params = cost_accuracy_params(n_iter=2, n_periods=400)
        analytical = {k: v for k, v in params.items() if k != "aum"}

        from_cost = captured_panels(cov_estimators_cost_accuracy, **params)
        from_analytical = captured_panels(
            scaling_factor_bias_variance, report_floor=False, **analytical
        )

        assert len(from_cost) == len(from_analytical) == 2
        for cost_panel, analytical_panel in zip(from_cost, from_analytical):
            np.testing.assert_allclose(cost_panel, analytical_panel)

    def test_configs_are_checked_before_any_simulation(self):
        with mock.patch.object(
            SignalsAndReturnsGenerator, "simulate_signals_and_returns"
        ) as simulate:
            with pytest.raises(ValueError, match=r"unknown key"):
                run_cost_accuracy(configs=[MA_10D, {**MA_10D, "lbak_periods": [5]}])

        simulate.assert_not_called()


class TestCostAccuracyCosts:
    def test_no_cost_object_charges_nothing(self):
        result = run_cost_accuracy(configs=[MA_10D, MA_60D])

        np.testing.assert_array_equal(result["cost_pct_aum_yr"], [0.0, 0.0])
        np.testing.assert_array_equal(result["cost_per_unit_risk"], [0.0, 0.0])
        # the book still ran, so the diagnostic volatility is unaffected
        assert (result["realized_vol_pct"] > 0).all()

    def test_doubling_the_cost_spec_doubles_the_cost(self):
        single = run_cost_accuracy(
            tcost_obj=uniform_cost_object(COST_FIDS, bidoffer=0.02, rollcost=0.002)
        )
        double = run_cost_accuracy(
            tcost_obj=uniform_cost_object(COST_FIDS, bidoffer=0.04, rollcost=0.004)
        )

        assert single["cost_pct_aum_yr"].iat[0] > 0
        # the raw figure is exactly linear in the cost spec
        np.testing.assert_allclose(
            double["cost_pct_aum_yr"], 2 * single["cost_pct_aum_yr"], rtol=1e-10
        )
        # the deflated one only nearly so, because charging more cost drags the PnL and
        # so moves the realized volatility it is divided by
        np.testing.assert_allclose(
            double["cost_per_unit_risk"], 2 * single["cost_per_unit_risk"], rtol=1e-3
        )
        # costs are charged against the PnL, not the positions, so the risk the book ran
        # is barely touched by the cost spec
        assert double["realized_vol_pct"].iat[0] == pytest.approx(
            single["realized_vol_pct"].iat[0], abs=5e-2
        )

    def test_short_lookback_trades_more_and_costs_more(self):
        result = run_cost_accuracy(
            configs=[MA_10D, MA_60D],
            tcost_obj=uniform_cost_object(COST_FIDS),
            n_periods=1200,
        )

        # true of the raw figure, and still true once the risk it ran is divided out -
        # the short lookback genuinely trades more, it is not just running hotter
        assert result["cost_pct_aum_yr"].iat[0] > result["cost_pct_aum_yr"].iat[1]
        assert result["cost_per_unit_risk"].iat[0] > result["cost_per_unit_risk"].iat[1]

    def test_cost_is_expressed_per_year_not_per_sample(self):
        # the raw total `evaluate_pnl` reports scales with panel length; the reported
        # figure must not
        short = run_cost_accuracy(
            n_periods=252 * 4, tcost_obj=uniform_cost_object(COST_FIDS)
        )
        long = run_cost_accuracy(
            n_periods=252 * 8, tcost_obj=uniform_cost_object(COST_FIDS)
        )

        assert short["cost_pct_aum_yr"].iat[0] == pytest.approx(
            long["cost_pct_aum_yr"].iat[0], rel=0.25
        )

    def test_per_unit_risk_divides_out_the_risk_actually_run(self):
        # configs are all asked for the same vol target but realize different vols, and
        # cost is charged on the notional actually run, so the raw column reports part of
        # the leverage difference as a turnover difference. Note this does not simply
        # shrink the spread - a slow monthly estimator can run hot and still be cheap per
        # unit of risk, in which case dividing the risk out separates the configs further
        # and can reorder them
        result = run_cost_accuracy(
            configs=[MA_10D, MA_60D, MA_MONTHLY_16],
            tcost_obj=uniform_cost_object(COST_FIDS),
            n_periods=1200,
        )

        expected = result["cost_pct_aum_yr"] / (result["realized_vol_pct"] / 10)
        np.testing.assert_allclose(result["cost_per_unit_risk"], expected, rtol=1e-12)

        # and the deflator is doing real work rather than rescaling everything equally
        assert result["realized_vol_pct"].max() / result["realized_vol_pct"].min() > 1.05
        factors = result["cost_per_unit_risk"] / result["cost_pct_aum_yr"]
        assert factors.max() / factors.min() > 1.05


class TestCostAccuracyAgreesWithTheAnalyticalHarness:
    """The two harnesses must read the same estimator the same way."""

    def test_estimator_matches_cov_matrix_history(self):
        # same configs, same data, two code paths into the covariance: the positions
        # path through `notional_positions`, and the analytical one
        generator = SignalsAndReturnsGenerator(
            n_fids=3, corr=COST_CORR, base_vol=COST_BASE_VOL, half_life=21
        )
        generator.simulate_signals_and_returns(
            n_periods=800,
            signal_names=[f"{fid}_CSIG_STRAT" for fid in COST_FIDS],
            return_names=[f"{fid}XR" for fid in COST_FIDS],
            seed=5,
        )
        df = pd.concat(
            (generator.quantamental_signals(), generator.quantamental_returns()),
            ignore_index=True,
        )
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

    def test_bias_agrees_with_the_analytical_harness(self):
        # the traded path is the only check that `notional_positions` actually sizes the
        # book the way the quadratic form assumes. It has to be read against the pooled
        # moment - see `pooled_vol_bias` - because a PnL standard deviation is a
        # quadratic mean over days, not a mean of per-interval ratios
        params = cost_accuracy_params(
            configs=[MA_10D, MA_60D], n_periods=252 * 5, n_iter=2
        )
        analytical = {k: v for k, v in params.items() if k != "aum"}

        traded = cov_estimators_cost_accuracy(vol_target=10, **params)
        pnl_bias = 1 - traded["realized_vol_pct"].to_numpy() / 10
        pooled = pooled_vol_bias(**analytical)

        np.testing.assert_allclose(pnl_bias, pooled, atol=0.02)
        assert abs(pnl_bias[0]) > abs(pnl_bias[1])

    def test_the_two_harnesses_report_different_moments(self):
        # guards the distinction above: reading the traded path against `1 - E[r]`, the
        # moment its companion returns, is wrong by more than the tolerance at a short
        # lookback, and the error grows with estimator noise
        params = cost_accuracy_params(
            configs=[MA_10D, MA_60D], n_periods=252 * 5, n_iter=2
        )
        analytical = {k: v for k, v in params.items() if k != "aum"}

        mean_bias = scaling_factor_bias_variance(
            report_floor=False, **analytical
        )["bias"].to_numpy()
        pooled = pooled_vol_bias(**analytical)

        # sqrt(E[r^2]) >= E[r], so the pooled reading is always the more negative
        assert (pooled <= mean_bias + 1e-12).all()
        assert abs(pooled[0] - mean_bias[0]) > abs(pooled[1] - mean_bias[1])


class TestDofCorrection:
    """
    `dof_correct` travels from a config through to the estimator, and lands where theory
    says it should.
    """

    @staticmethod
    def _configs(dof_correct):
        return [
            {
                "est_freqs": ["D"],
                "lback_meth": "ma",
                "lback_periods": [n],
                "dof_correct": dof_correct,
            }
            for n in (15, 60)
        ]

    @staticmethod
    def _jensen_floor(n):
        """
        What remains once the variance estimate is unbiased: Jensen on `1 / sqrt`.

        Positions scale with `1 / sqrt(variance)`, which is convex, so a noisy but
        correctly centred estimate still oversizes on average. `E[sqrt(k / chi2_k)]` with
        `k = n - 1` degrees of freedom. Irreducible - it is a property of targeting
        volatility off a finite sample, not of this estimator.
        """
        k = n - 1
        from scipy.special import gammaln

        return 1 - np.sqrt(k / 2) * np.exp(gammaln((k - 1) / 2) - gammaln(k / 2))

    def test_config_key_is_accepted_and_forwarded(self):
        resolved = _resolve_config({**MA_10D, "dof_correct": True})
        assert resolved["dof_correct"] is True
        assert _resolve_config(MA_10D)["dof_correct"] is False

    def test_correction_lands_on_the_jensen_floor(self):
        # constant volatility, so there is nothing to forecast and every reading is
        # artifact. Uncorrected, a 15-day lookback reports a large negative "bias" purely
        # from the lost degree of freedom; corrected, only the Jensen floor is left
        dgp = dict(
            corr=COST_CORR,
            base_vol=COST_BASE_VOL,
            vol_persistence=0.94,
            vol_of_vol=0.0,
            fid_names=COST_FIDS,
            n_periods=252 * 12,
            n_iter=12,
            seed=7,
        )
        uncorrected = scaling_factor_bias_variance(
            configs=self._configs(False), report_floor=False, **dgp
        )["bias"].to_numpy()
        corrected = scaling_factor_bias_variance(
            configs=self._configs(True), report_floor=False, **dgp
        )["bias"].to_numpy()
        floor = np.array([self._jensen_floor(n) for n in (15, 60)])

        # the correction removes the degrees-of-freedom component and only that
        np.testing.assert_allclose(corrected, floor, atol=0.015)
        assert (corrected > uncorrected).all()

        # and it is worth far more at the short lookback than the long one
        moved = corrected - uncorrected
        assert moved[0] > 3 * moved[1]

    def test_corrected_estimator_sizes_smaller_positions(self):
        # the whole point: a larger risk estimate means a smaller book
        on = cov_estimators_cost_accuracy(
            **cost_accuracy_params(configs=self._configs(True), n_periods=252 * 5)
        )
        off = cov_estimators_cost_accuracy(
            **cost_accuracy_params(configs=self._configs(False), n_periods=252 * 5)
        )

        # a larger risk estimate sizes a smaller book, so the overshoot shrinks
        assert on["realized_vol_pct"].iat[0] < off["realized_vol_pct"].iat[0]


class TestNoiseFloor:
    """
    `bias` carries a floor that depends only on estimator noise, so it falls with the
    lookback whether or not there is anything to forecast. `noise_floor` measures it and
    `excess_bias` nets it out.
    """

    FLOOR_DGP = dict(
        corr=COST_CORR,
        base_vol=COST_BASE_VOL,
        vol_persistence=0.94,
        fid_names=COST_FIDS,
        n_periods=252 * 8,
        n_iter=4,
        seed=7,
    )

    def test_frame_is_labelled_and_one_row_per_config(self):
        result = scaling_factor_bias_variance(
            configs=[MA_10D, MA_60D], vol_of_vol=0.15, **self.FLOOR_DGP
        )

        assert list(result.index) == [0, 1]
        assert list(result.columns) == ["bias", "std", "noise_floor", "excess_bias"]
        np.testing.assert_allclose(
            result["excess_bias"], result["bias"] - result["noise_floor"]
        )

    def test_floor_is_nan_when_not_requested(self):
        result = scaling_factor_bias_variance(
            configs=[MA_10D], vol_of_vol=0.15, report_floor=False, **self.FLOOR_DGP
        )

        assert np.isfinite(result["bias"]).all()
        assert result["noise_floor"].isna().all()
        assert result["excess_bias"].isna().all()

    def test_floor_falls_with_the_lookback(self):
        # the whole problem in one assertion: with nothing to forecast, a short lookback
        # still scores far worse than a long one
        result = scaling_factor_bias_variance(
            configs=[MA_10D, MA_60D], vol_of_vol=0.15, **self.FLOOR_DGP
        )

        floor = result["noise_floor"]
        assert floor.iat[0] < floor.iat[1] < 0
        assert abs(floor.iat[0]) > 3 * abs(floor.iat[1])

    def test_raw_bias_ranks_lookbacks_even_with_nothing_to_forecast(self):
        # this is why the ranking cannot be read off `bias`. Under constant volatility
        # every config is correctly specified, yet the short lookback still looks worst -
        # and `excess_bias` correctly collapses to about zero for both
        result = scaling_factor_bias_variance(
            configs=[MA_10D, MA_60D], vol_of_vol=0.0, **self.FLOOR_DGP
        )

        assert result["bias"].iat[0] < result["bias"].iat[1]  # short "looks worse"
        np.testing.assert_allclose(result["excess_bias"], 0.0, atol=1e-12)

    def test_excess_bias_is_what_responds_to_forecastable_vol(self):
        # `excess_bias` must be near zero when there is nothing to forecast and clearly
        # negative when there is, which is the property `bias` itself does not have
        quiet = scaling_factor_bias_variance(
            configs=[MA_10D], vol_of_vol=0.0, **self.FLOOR_DGP
        )
        stormy = scaling_factor_bias_variance(
            configs=[MA_10D], vol_of_vol=0.35, **self.FLOOR_DGP
        )

        assert quiet["excess_bias"].iat[0] == pytest.approx(0.0, abs=1e-12)
        assert stormy["excess_bias"].iat[0] < -0.01

        # `bias` moves far less between the two regimes than the floor explains, which is
        # exactly the confound: most of the short-lookback reading is noise either way
        assert quiet["bias"].iat[0] < -0.03


class TestCalendarIsPinned:
    """
    `SignalsAndReturnsGenerator` dates its panel back from today when `end_date` is None.
    Left at that, the business-month-start grid - and so every reported number - would
    change from one day to the next for the same seed.
    """

    @staticmethod
    def _end_dates_seen(fn, **params):
        seen = []
        real = SignalsAndReturnsGenerator.simulate_signals_and_returns

        def recorder(self, *args, **kwargs):
            seen.append(kwargs.get("end_date"))
            return real(self, *args, **kwargs)

        with mock.patch.object(
            SignalsAndReturnsGenerator, "simulate_signals_and_returns", recorder
        ):
            fn(**params)
        return seen

    def test_analytical_harness_never_uses_today(self):
        # `cost_accuracy_params` carries no `aum`, so it is already the analytical set
        seen = self._end_dates_seen(
            scaling_factor_bias_variance,
            report_floor=False,
            **cost_accuracy_params(n_iter=2, n_periods=400),
        )

        assert seen, "the harness did not simulate anything"
        assert all(date is not None for date in seen)
        assert set(seen) == {analysis.DEFAULT_END_DATE}

    def test_cost_harness_uses_the_same_calendar(self):
        seen = self._end_dates_seen(
            cov_estimators_cost_accuracy,
            **cost_accuracy_params(n_iter=2, n_periods=400),
        )

        assert set(seen) == {analysis.DEFAULT_END_DATE}

    def test_default_end_date_is_not_a_rebalance_date(self):
        # the DGP zeroes its final signal row; if that row is also a rebalance date the
        # last positions are all zero and `proxy_pnl_calc` rejects the frame
        end = pd.Timestamp(analysis.DEFAULT_END_DATE)
        month_starts = pd.date_range(end - pd.offsets.BDay(400), end, freq="BMS")

        assert end.weekday() < 5
        assert end not in month_starts

    def test_explicit_end_date_still_changes_the_panel(self):
        # the pin is a default, not a hard-coding
        base = run_bias_variance(configs=[MA_10D])
        moved = run_bias_variance(configs=[MA_10D], end_date="2025-03-19")

        assert base[0][0] != pytest.approx(moved[0][0])


class TestIterationSeeds:
    """
    Iterations are averaged as though independent, so their panels must be distinct.
    Drawing with replacement from a narrow range duplicated one with probability ~1.9%
    at `n_iter=20` and ~11.5% at 50.
    """

    def test_seeds_are_distinct_at_scale(self):
        seeds = analysis._iteration_seeds(seed=42, n_iter=500)

        assert len(set(seeds)) == 500

    def test_seeds_are_deterministic_in_the_seed(self):
        assert analysis._iteration_seeds(seed=42, n_iter=10) == analysis._iteration_seeds(
            seed=42, n_iter=10
        )
        assert analysis._iteration_seeds(seed=42, n_iter=10) != analysis._iteration_seeds(
            seed=43, n_iter=10
        )

    def test_seeds_are_prefix_stable(self):
        # what lets the analytical harness run at a high `n_iter` and the cost harness at
        # a low one over the same panels
        assert (
            analysis._iteration_seeds(seed=42, n_iter=20)[:5]
            == analysis._iteration_seeds(seed=42, n_iter=5)
        )

    def test_streams_do_not_overlap(self):
        # distinct seeds are necessary but not sufficient; the streams they open must
        # differ too
        first, second = analysis._iteration_seeds(seed=42, n_iter=2)
        draws_a = np.random.default_rng(first).standard_normal(50)
        draws_b = np.random.default_rng(second).standard_normal(50)

        assert not np.allclose(draws_a, draws_b)

    def test_harness_panels_are_all_different(self):
        panels = captured_panels(
            scaling_factor_bias_variance,
            report_floor=False,
            **cost_accuracy_params(n_iter=6, n_periods=400),
        )

        assert len(panels) == 6
        for i in range(len(panels)):
            for j in range(i + 1, len(panels)):
                assert not np.allclose(panels[i], panels[j])

    def test_a_shorter_cost_run_reuses_the_analytical_panels(self):
        params = cost_accuracy_params(n_periods=400)
        analytical = captured_panels(
            scaling_factor_bias_variance,
            report_floor=False,
            **{**params, "n_iter": 4},
        )
        traded = captured_panels(
            cov_estimators_cost_accuracy, **{**params, "n_iter": 2}
        )

        assert len(analytical) == 4 and len(traded) == 2
        for cheap, expensive in zip(analytical, traded):
            np.testing.assert_allclose(cheap, expensive)


class TestHalfLivesAreDistinct:
    """
    Two unrelated half lives meet in one call: the simulated signal's forecasting decay
    and an exponentially weighted estimator's. Only the second belongs in a config.
    """

    def test_both_half_lives_can_be_set_in_one_call(self):
        xma = {
            "est_freqs": ["D"],
            "lback_meth": "xma",
            "half_life": [12],
            "lback_periods": [-1],
        }
        result = scaling_factor_bias_variance(
            configs=[xma],
            signal_half_life=63,
            report_floor=False,
            vol_of_vol=0.15,
            corr=COST_CORR,
            base_vol=COST_BASE_VOL,
            vol_persistence=0.94,
            fid_names=COST_FIDS,
            n_periods=252 * 4,
            n_iter=2,
            seed=7,
        )

        assert np.isfinite(result["bias"]).all()

    def test_signal_half_life_is_not_an_estimator_config_key(self):
        with pytest.raises(ValueError, match=r"unknown key"):
            _resolve_config({**MA_10D, "signal_half_life": 21})

    def test_estimator_half_life_does_not_reach_the_generator(self):
        # changing only the estimator's decay must leave the simulated panel identical
        def panels(half_life):
            return captured_panels(
                scaling_factor_bias_variance,
                report_floor=False,
                **cost_accuracy_params(
                    n_iter=2,
                    n_periods=400,
                    configs=[
                        {
                            "est_freqs": ["D"],
                            "lback_meth": "xma",
                            "half_life": [half_life],
                            "lback_periods": [-1],
                        }
                    ],
                ),
            )

        for fast, slow in zip(panels(6), panels(40)):
            np.testing.assert_allclose(fast, slow)
