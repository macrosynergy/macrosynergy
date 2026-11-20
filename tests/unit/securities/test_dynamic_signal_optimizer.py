"""Compare the wrapper with ordinary SignalOptimizer runs."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from joblib import cpu_count
from pydantic import ValidationError
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_squared_error

from macrosynergy.learning import (
    ExpandingKFoldPanelSplit,
    SignalOptimizer,
)

from macrosynergy.securities import DynamicSignalOptimizer

MODULE = "macrosynergy.securities.dynamic_signal_optimizer"


def make_panel(freq="D"):
    dates = (
        pd.bdate_range("2018-01-01", periods=40)
        if freq == "D"
        else pd.bdate_range("2018-01-01", "2019-12-31")
    )
    rng = np.random.RandomState(17)
    frames = []
    for cid in ["A", "B", "C"]:
        f1, f2 = rng.normal(size=(2, len(dates)))
        wide = pd.DataFrame(
            {
                "F1": f1,
                "F2": f2,
                "Y": 2 * f1 - f2 + rng.normal(size=len(dates)),
                "Z": f1 + 3 * f2 + rng.normal(size=len(dates)),
            },
            index=dates,
        )
        wide.index.name = "real_date"
        long = wide.reset_index().melt(
            id_vars="real_date", var_name="xcat", value_name="value"
        )
        long["cid"] = cid
        frames.append(long)
    return pd.concat(frames, ignore_index=True), dates


def calculate(optimizer, name="signal", **kwargs):
    options = dict(
        models={"lr": LinearRegression()},
        min_cids=1,
        min_periods=5,
        show_tqdm=False,
    )
    if not isinstance(optimizer, DynamicSignalOptimizer):
        options["n_jobs_outer"] = 1
    options.update(kwargs)
    optimizer.calculate_predictions(name, **options)
    return optimizer


def assert_signals(left, right):
    def clean(frame):
        frame = frame.copy()
        for key in ["cid", "xcat"]:
            frame[key] = frame[key].astype(object)
        return frame.sort_values(["real_date", "cid", "xcat"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(clean(left), clean(right), check_dtype=False)


@pytest.mark.parametrize("freq", ["D", "M"])
@pytest.mark.parametrize("lag", [0, 1, 2])
@pytest.mark.parametrize("max_periods", [None, 8])
def test_fixed_specification_matches_signal_optimizer(freq, lag, max_periods):
    df, _ = make_panel(freq)
    init = dict(df=df, xcats=["F1", "F2", "Y"], freq=freq, lag=lag)
    options = dict(test_size=3, max_periods=max_periods, min_cids=2)
    dynamic = calculate(DynamicSignalOptimizer(**init), **options)
    ordinary = calculate(SignalOptimizer(**init), **options)
    assert_signals(dynamic.preds, ordinary.preds)
    for table in [
        "feature_importances",
        "selected_ftrs",
        "chosen_models",
        "intercepts",
    ]:
        pd.testing.assert_frame_equal(
            getattr(dynamic, table).reset_index(drop=True),
            getattr(ordinary, table).reset_index(drop=True),
            check_dtype=False,
            check_categorical=False,
        )
    assert type(dynamic.optimizers["signal"][0]) is SignalOptimizer


def test_cv_controls_are_delegated_and_inputs_are_not_mutated():
    df, _ = make_panel()
    model = Ridge()
    cv = ExpandingKFoldPanelSplit(n_splits=2)
    kwargs = dict(
        models={"ridge": model},
        inner_splitters={"cv": cv},
        hyperparameters={"ridge": {"alpha": [1, 2]}},
        scorers={"mse": make_scorer(mean_squared_error, greater_is_better=False)},
        split_functions={"cv": lambda n: n / 20},
        test_size=10,
        min_periods=10,
    )
    dynamic = calculate(DynamicSignalOptimizer(df, ["F1", "Y"], freq="D"), **kwargs)
    assert cv.n_splits == 2
    assert not hasattr(model, "coef_")
    ordinary = calculate(SignalOptimizer(df, ["F1", "Y"], freq="D"), **kwargs)
    assert_signals(dynamic.preds, ordinary.preds)
    pd.testing.assert_frame_equal(dynamic.chosen_models, ordinary.chosen_models)


def test_start_date_retains_underlying_meaning():
    df, dates = make_panel()
    opts = dict(start_date=str(dates[20].date()))
    dynamic = calculate(DynamicSignalOptimizer(df, ["F1", "Y"], freq="D"), **opts)
    ordinary = calculate(SignalOptimizer(df, ["F1", "Y"], freq="D"), **opts)
    assert_signals(dynamic.preds, ordinary.preds)


@pytest.mark.parametrize("options,expected", [({}, 36), ({"min_periods": 6}, 6)])
def test_min_periods_default_and_override_reach_child(options, expected):
    df, _ = make_panel()
    optimizer = DynamicSignalOptimizer(df, ["F1", "Y"], freq="D")
    with patch.object(SignalOptimizer, "calculate_predictions") as predict:
        optimizer.calculate_predictions(
            "signal", models={"lr": LinearRegression()}, show_tqdm=False, **options
        )
    assert predict.call_args.kwargs["min_periods"] == expected


@pytest.mark.parametrize("value", [None, 0])
def test_invalid_min_periods_fails_before_workers_start(value):
    df, _ = make_panel()
    optimizer = DynamicSignalOptimizer(df, ["F1", "Y"], freq="D")
    with patch(f"{MODULE}._fit_vintage") as fit:
        with pytest.raises(ValidationError):
            calculate(optimizer, min_periods=value)
        fit.assert_not_called()


@pytest.mark.parametrize("value", [0, -2, True, 1.5])
def test_invalid_worker_counts_fail_before_fitting(value):
    df, _ = make_panel()
    optimizer = DynamicSignalOptimizer(df, ["F1", "Y"], freq="D")
    with patch(f"{MODULE}._fit_vintage") as fit:
        with pytest.raises(ValidationError):
            calculate(optimizer, n_jobs_vintages=value)
        fit.assert_not_called()


def test_vintage_workers_cannot_exceed_available_cores():
    df, _ = make_panel()
    optimizer = DynamicSignalOptimizer(df, ["F1", "Y"], freq="D")
    with patch(f"{MODULE}.cpu_count", return_value=8), patch(
        f"{MODULE}._fit_vintage"
    ) as fit:
        with pytest.raises(ValidationError, match="exceeding 8 available cores"):
            calculate(optimizer, n_jobs_vintages=9)
        fit.assert_not_called()


@pytest.mark.parametrize("jobs", [8, -1])
def test_core_budget_accepts_equality_and_resolves_all_cores(jobs):
    from macrosynergy.securities.dynamic_signal_optimizer import (
        _ExecutionConfig,
    )

    with patch(f"{MODULE}.cpu_count", return_value=8):
        config = _ExecutionConfig(n_jobs_vintages=jobs)
    assert config.n_jobs_vintages == 8


@pytest.mark.parametrize("field", ["n_jobs_outer", "n_jobs_inner"])
def test_nested_worker_options_are_not_exposed(field):
    df, _ = make_panel()
    optimizer = DynamicSignalOptimizer(df, ["F1", "Y"], freq="D")
    with patch(f"{MODULE}._fit_vintage") as fit:
        with pytest.raises(TypeError, match="fixed at 1"):
            calculate(optimizer, **{field: 2})
        fit.assert_not_called()


def test_child_parallelism_and_progress_are_hard_coded():
    from macrosynergy.securities.dynamic_signal_optimizer import _fit_vintage

    with patch.object(SignalOptimizer, "calculate_predictions") as predict:
        df, _ = make_panel()
        _fit_vintage(
            df=df,
            xcats=["F1", "Y"],
            cids=None,
            n_targets=1,
            init_options={"freq": "D"},
            name="signal",
            models={"lr": LinearRegression()},
            run_options={"test_size": 3},
        )
    assert predict.call_args.kwargs == {
        "n_jobs_outer": 1,
        "n_jobs_inner": 1,
        "show_tqdm": False,
        "test_size": 3,
    }


def test_rerunning_one_process_preserves_other_process_results():
    df, _ = make_panel()
    optimizer = DynamicSignalOptimizer(df, ["F1", "Y", "Z"], n_targets=2, freq="D")
    calculate(optimizer, name="first", test_size=20)
    calculate(optimizer, name="second", test_size=20)
    second_signals = optimizer.get_optimized_signals("second").copy(deep=True)
    second_models = optimizer.chosen_models.loc[
        optimizer.chosen_models.name == "second"
    ].copy()
    calculate(optimizer, name="first", models={"ridge": Ridge()}, test_size=20)
    assert_signals(optimizer.get_optimized_signals("second"), second_signals)
    pd.testing.assert_frame_equal(
        optimizer.chosen_models.loc[
            optimizer.chosen_models.name == "second"
        ].reset_index(drop=True),
        second_models.reset_index(drop=True),
        # Other processes can extend shared category vocabularies.
        check_categorical=False,
    )
    assert not optimizer.preds.duplicated(["cid", "xcat", "real_date"]).any()


@pytest.mark.parametrize("failure", ["name_collision", "merge"])
def test_storage_failure_preserves_all_published_state(failure):
    from macrosynergy.management.utils import concat_categorical

    df, _ = make_panel()
    optimizer = DynamicSignalOptimizer(df, ["F1", "Y", "Z"], n_targets=2, freq="D")
    calculate(optimizer, test_size=20)
    original_tables = {
        key: getattr(optimizer, key).copy(deep=True) for key in optimizer._RESULT_TABLES
    }
    original_children = optimizer.optimizers.copy()
    original_outputs = optimizer._process_outputs.copy()
    original_regimes = optimizer.regimes.copy(deep=True)

    if failure == "name_collision":
        with pytest.raises(ValueError, match="collide"):
            calculate(optimizer, name="Y_signal", test_size=20)
    else:
        # Fail after staging one table, without publishing a partial update.
        calls = 0

        def fail_second_merge(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("Result merge failed")
            return concat_categorical(*args, **kwargs)

        with patch(f"{MODULE}.concat_categorical", side_effect=fail_second_merge):
            with pytest.raises(ValueError, match="Result merge failed"):
                calculate(optimizer, test_size=20)

    for key, frame in original_tables.items():
        pd.testing.assert_frame_equal(getattr(optimizer, key), frame)
    assert optimizer.optimizers == original_children
    assert optimizer._process_outputs == original_outputs
    pd.testing.assert_frame_equal(optimizer.regimes, original_regimes)


@pytest.mark.parametrize("name", [123, ["signal", 1], {"signal": None}])
def test_signal_getter_rejects_invalid_name_types(name):
    df, _ = make_panel()
    optimizer = DynamicSignalOptimizer(df, ["F1", "Y"], freq="D")
    with pytest.raises(TypeError, match="string or a list of strings"):
        optimizer.get_optimized_signals(name)


@pytest.mark.parametrize(
    "method",
    [
        "get_optimized_signals",
        "get_optimal_models",
        "get_selected_features",
        "get_feature_importances",
        "get_intercepts",
        "get_feature_correlations",
    ],
)
def test_combined_result_getters_match_native_optimizer(method):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df, _ = make_panel()
    init = dict(df=df, xcats=["F1", "F2", "Y"], freq="D")
    options = dict(
        models={"pipeline": make_pipeline(StandardScaler(), LinearRegression())},
        store_correlations=True,
        test_size=20,
    )
    dynamic = calculate(DynamicSignalOptimizer(**init), **options)
    ordinary = calculate(SignalOptimizer(**init), **options)
    for name in (None, "signal", ["signal"]):
        actual = getattr(dynamic, method)(name)
        expected = getattr(ordinary, method)(name)
        pd.testing.assert_frame_equal(actual, expected)
    with pytest.raises(ValueError):
        getattr(dynamic, method)("missing")
    with pytest.raises(TypeError):
        getattr(dynamic, method)(123)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cids": ["unknown"]},
        {"xcats": ["F1", "unknown"]},
        {"start": "2099-01-01"},
    ],
)
def test_invalid_input_selection_is_rejected_without_a_learner(kwargs):
    df, _ = make_panel()
    options = dict(xcats=["F1", "Y"], freq="D")
    options.update(kwargs)
    with patch(
        f"{MODULE}.SignalOptimizer", side_effect=AssertionError("Unexpected learner")
    ):
        with pytest.raises(ValueError):
            DynamicSignalOptimizer(df, **options)


def make_regimes():
    return {
        "summer": {
            "start": "2018-06-30",
            "end": "2018-09-30",
            "end_return": "2018-09-28",
            "targets": ["Y", "Z", "MISSING"],
        },
        17: {
            "start": "2018-09-30",
            "end": "2018-12-31",
            "end_return": "2018-12-28",
            "targets": ["Z"],
        },
    }


def make_dynamic(df, regimes=None, **kwargs):
    return DynamicSignalOptimizer(
        df,
        ["F1", "F2", "Y", "Z", "MISSING"],
        n_targets=3,
        cids=["A"],
        freq="M",
        regimes=make_regimes() if regimes is None else regimes,
        **kwargs,
    )


def manual_notebook_run(df, details, *, lag=1, test_size=3):
    """Reproduce the notebook's preparation and fitting independently of the wrapper."""
    regime_data = df.loc[(df.cid == "A") & (df.real_date <= details["end"])]
    returns = regime_data.loc[regime_data.xcat.isin(details["targets"])]
    if returns.real_date.dt.to_period("M").nunique() < 5:
        return None
    targets = [target for target in details["targets"] if target in set(returns.xcat)]
    features = regime_data.loc[regime_data.xcat.isin(["F1", "F2"])]
    inputs = pd.concat([features, returns], ignore_index=True)
    inputs = inputs.loc[inputs.real_date <= details["end_return"]]
    child = SignalOptimizer(
        inputs,
        ["F1", "F2"] + targets,
        cids=["A"],
        n_targets=len(targets),
        freq="M",
        lag=lag,
    )
    dates = child.y.index.get_level_values(1).unique().sort_values()
    first_prediction = dates[dates <= pd.Timestamp(details["start"])][-1]
    if first_prediction >= dates.max():
        return None
    calculate(
        child, start_date=first_prediction.strftime("%Y-%m-%d"), test_size=test_size
    )
    return child


@pytest.mark.parametrize("lag", [0, 1, 2])
@pytest.mark.parametrize("test_size", [1, 3])
def test_explicit_regimes_match_notebook(lag, test_size):
    from copy import deepcopy

    df, _ = make_panel("M")
    regimes = make_regimes()
    original_regimes = deepcopy(regimes)
    original_df = df.copy(deep=True)
    dynamic = calculate(make_dynamic(df, regimes, lag=lag), test_size=test_size)
    expected_tables = {table: [] for table in dynamic._RESULT_TABLES}
    for key, details in regimes.items():
        ordinary = manual_notebook_run(df, details, lag=lag, test_size=test_size)
        child = dynamic.optimizers["signal"][key]
        pd.testing.assert_frame_equal(child.X, ordinary.X)
        if isinstance(child.y, pd.Series):
            pd.testing.assert_series_equal(child.y, ordinary.y)
        else:
            pd.testing.assert_frame_equal(child.y, ordinary.y)
        assert child.df.real_date.max() <= pd.Timestamp(details["end_return"])
        assert child.df.real_date.min() < pd.Timestamp(details["start"])
        for table in dynamic._RESULT_TABLES:
            pd.testing.assert_frame_equal(
                getattr(child, table), getattr(ordinary, table)
            )
            frame = getattr(ordinary, table).copy()
            if table == "preds" and ordinary.n_targets == 1:
                frame["xcat"] = ordinary.xcats[-1] + "_signal"
            bday = pd.offsets.BDay()
            frame = frame.loc[
                (frame.real_date >= bday.rollback(pd.Timestamp(details["start"])))
                & (frame.real_date < bday.rollback(pd.Timestamp(details["end"])))
            ]
            expected_tables[table].append(frame)
    for table, frames in expected_tables.items():
        expected = pd.concat(frames, ignore_index=True)
        actual = getattr(dynamic, table)
        if table == "preds":
            assert_signals(actual, expected)
        else:
            pd.testing.assert_frame_equal(actual, expected, check_categorical=False)
    report = dynamic.get_regimes("signal")
    assert list(report.index) == ["summer", 17]
    assert report.loc["summer", "missing_targets"] == ("MISSING",)
    assert report.loc[17, "available_targets"] == ("Z",)
    assert list(report.first_prediction) == list(
        pd.to_datetime(["2018-06-29", "2018-09-28"])
    )
    if lag > 0:
        assert dynamic.preds.real_date.min() == pd.Timestamp("2018-06-29")
        assert dynamic.preds.loc[
            dynamic.preds.real_date == "2018-09-28", "xcat"
        ].tolist() == ["Z_signal"]
    assert not dynamic.preds.duplicated(["cid", "xcat", "real_date"]).any()
    assert regimes == original_regimes
    pd.testing.assert_frame_equal(df, original_df)


def test_plan_preserves_ids_target_order_and_separate_identical_regimes():
    df, _ = make_panel("M")
    regimes = make_regimes()
    regimes[17]["targets"] = regimes["summer"]["targets"] = ["Z", "Y"]
    regimes = {17: regimes[17], "summer": regimes["summer"]}
    with patch(
        f"{MODULE}.SignalOptimizer", side_effect=AssertionError("Unexpected learner")
    ):
        optimizer = make_dynamic(df, regimes)
        plan = optimizer.get_regimes()
    assert list(plan.index) == ["summer", 17]
    assert list(plan.targets) == [("Z", "Y"), ("Z", "Y")]
    assert plan.loc["summer", "start"] == pd.Timestamp("2018-06-30")
    calculate(optimizer, test_size=3)
    assert len(optimizer.optimizers["signal"]) == 2
    assert list(optimizer.get_regimes("signal").first_prediction) == list(
        pd.to_datetime(["2018-06-29", "2018-09-28"])
    )
    assert all(
        child.xcats[-2:] == ["Z", "Y"]
        for child in optimizer.optimizers["signal"].values()
    )


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"a": {"start": "2018-06-30", "end": "2018-09-30", "targets": ["Y"]}},
        {
            "a": {
                "start": "bad",
                "end": "2018-09-30",
                "end_return": "2018-09-28",
                "targets": ["Y"],
            }
        },
        {
            "a": {
                "start": "2018-09-30",
                "end": "2018-06-30",
                "end_return": "2018-09-28",
                "targets": ["Y"],
            }
        },
        {
            "a": {
                "start": "2018-06-30",
                "end": "2018-09-30",
                "end_return": "2018-09-28",
                "targets": ["F1"],
            }
        },
        {
            "a": {
                "start": "2018-06-30",
                "end": "2018-09-30",
                "end_return": "2018-09-28",
                "targets": ["Y", "Y"],
            }
        },
        {
            "a": {
                "start": "2018-06-30",
                "end": "2018-09-30",
                "end_return": "2018-09-28",
                "targets": "Y",
            }
        },
        {
            "a": {
                "start": "2018-06-30",
                "end": "2018-09-30",
                "end_return": "2018-09-28",
                "targets": ["Y"],
                "extra": 1,
            }
        },
    ],
)
def test_invalid_regimes_fail_before_a_learner_is_created(bad):
    df, _ = make_panel("M")
    with patch(
        f"{MODULE}.SignalOptimizer", side_effect=AssertionError("Unexpected learner")
    ):
        with pytest.raises(ValidationError):
            make_dynamic(df, bad)


def test_overlapping_regimes_are_rejected():
    df, _ = make_panel("M")
    regimes = make_regimes()
    regimes[17]["start"] = "2018-09-29"
    with pytest.raises(ValidationError, match="overlap"):
        make_dynamic(df, regimes)


def test_regime_start_date_cannot_be_overridden():
    df, _ = make_panel("M")
    with patch(f"{MODULE}._fit_vintage") as fit:
        with pytest.raises(TypeError, match="derived"):
            calculate(make_dynamic(df), start_date="2018-01-31")
        fit.assert_not_called()


@pytest.mark.parametrize(
    "targets,minimum,reason",
    [
        ([], 5, "no_targets"),
        (["MISSING"], 5, "no_targets"),
        (["Y"], 100, "insufficient_history"),
    ],
)
def test_skipped_regimes_produce_empty_results_and_a_report(targets, minimum, reason):
    df, _ = make_panel("M")
    regimes = {"skip": dict(make_regimes()["summer"], targets=targets)}
    optimizer = make_dynamic(df, regimes)
    with patch(
        f"{MODULE}.SignalOptimizer", side_effect=AssertionError("Unexpected learner")
    ):
        calculate(optimizer, min_periods=minimum)
    assert optimizer.optimizers["signal"] == {}
    assert optimizer.get_optimized_signals("signal").empty
    report = optimizer.get_regimes("signal")
    assert report.loc["skip", "status"] == "skipped"
    assert report.loc["skip", "skip_reason"] == reason
    assert all(getattr(optimizer, table).empty for table in optimizer._RESULT_TABLES)


@pytest.mark.parametrize(
    "start,end_return",
    [
        ("2017-12-31", "2018-06-29"),
        ("2018-12-31", "2018-06-29"),
    ],
)
def test_regimes_without_a_prediction_period_are_skipped(start, end_return):
    df, _ = make_panel("M")
    regimes = {
        "skip": {
            "start": start,
            "end": "2019-03-31",
            "end_return": end_return,
            "targets": ["Y"],
        }
    }
    with patch.object(
        SignalOptimizer,
        "calculate_predictions",
        side_effect=AssertionError("Unexpected fit"),
    ):
        optimizer = calculate(make_dynamic(df, regimes))
    assert (
        optimizer.get_regimes("signal").loc["skip", "skip_reason"]
        == "no_prediction_periods"
    )


def test_history_check_precedes_end_return_cutoff():
    df, _ = make_panel("M")
    regimes = {
        "early_cutoff": {
            "start": "2018-04-30",
            "end": "2018-06-30",
            "end_return": "2018-05-31",
            "targets": ["Y"],
        }
    }
    optimizer = calculate(make_dynamic(df, regimes), min_periods=6)
    report = optimizer.get_regimes("signal")
    assert report.loc["early_cutoff", "history_months"] == 6
    assert report.loc["early_cutoff", "status"] == "fitted"
    child = optimizer.optimizers["signal"]["early_cutoff"]
    assert child.df.real_date.dt.to_period("M").nunique() == 5
    assert child.df.real_date.max() == pd.Timestamp("2018-05-31")


def test_history_counts_union_months_including_nan_records():
    df, _ = make_panel("M")
    df = df.loc[
        ~((df.xcat == "Y") & (df.real_date >= "2018-04-01"))
        & ~((df.xcat == "Z") & (df.real_date < "2018-04-01"))
    ].copy()
    df.loc[df.xcat.isin(["Y", "Z"]), "value"] = np.nan
    regimes = {
        "split_history": {
            "start": "2018-04-30",
            "end": "2018-06-30",
            "end_return": "2018-06-29",
            "targets": ["Y", "Z"],
        }
    }
    with patch.object(SignalOptimizer, "calculate_predictions") as fit:
        optimizer = calculate(make_dynamic(df, regimes, drop_nas="X"), min_periods=6)
    fit.assert_called_once()
    report = optimizer.get_regimes("signal")
    assert report.loc["split_history", "history_months"] == 6
    assert report.loc["split_history", "available_targets"] == ("Y", "Z")


def test_future_target_records_do_not_make_it_available_early():
    df, _ = make_panel("M")
    df = df.loc[~((df.xcat == "Z") & (df.real_date <= "2018-09-30"))]
    optimizer = calculate(make_dynamic(df))
    report = optimizer.get_regimes("signal")
    assert report.loc["summer", "available_targets"] == ("Y",)
    assert report.loc["summer", "missing_targets"] == ("Z", "MISSING")
    assert report.loc[17, "available_targets"] == ("Z",)


def test_data_after_end_return_does_not_change_that_regime():
    df, _ = make_panel("M")
    original = calculate(make_dynamic(df), test_size=3)
    changed = df.copy()
    changed.loc[changed.real_date > "2018-09-28", "value"] *= 100
    future = calculate(make_dynamic(changed), test_size=3)
    assert_signals(
        original.optimizers["signal"]["summer"].preds,
        future.optimizers["signal"]["summer"].preds,
    )
    assert_signals(
        original.preds.loc[original.preds.real_date < "2018-09-28"],
        future.preds.loc[future.preds.real_date < "2018-09-28"],
    )


def test_global_input_bounds_apply_without_truncating_training_at_regime_start():
    df, _ = make_panel("M")
    optimizer = calculate(make_dynamic(df, start="2018-02-01", end="2018-10-31"))
    report = optimizer.get_regimes("signal")
    assert report.loc["summer", "history_months"] == 8
    assert report.loc[17, "history_months"] == 9
    for child in optimizer.optimizers["signal"].values():
        assert child.df.real_date.min() == pd.Timestamp("2018-02-01")
        assert child.df.real_date.max() <= pd.Timestamp("2018-10-31")


def test_all_skipped_rerun_replaces_old_results_but_preserves_other_processes():
    df, _ = make_panel("M")
    optimizer = calculate(make_dynamic(df))
    calculate(optimizer, name="second")
    second = optimizer.get_optimized_signals("second")
    calculate(optimizer, min_periods=100)
    assert optimizer.get_optimized_signals("signal").empty
    assert optimizer.optimizers["signal"] == {}
    assert set(optimizer.get_regimes("signal").status) == {"skipped"}
    assert_signals(optimizer.get_optimized_signals("second"), second)


@pytest.mark.parametrize("show", [True, False])
def test_one_progress_bar_covers_fitted_and_skipped_regimes(show):
    df, _ = make_panel("M")
    regimes = make_regimes()
    regimes["skip"] = {
        "start": "2018-12-31",
        "end": "2019-03-31",
        "end_return": "2019-03-29",
        "targets": [],
    }
    with patch(f"{MODULE}.tqdm") as progress, patch(
        "macrosynergy.learning.sequential.base_panel_learner.tqdm",
        side_effect=lambda iterable, **kwargs: iterable,
    ) as child_progress:
        calculate(make_dynamic(df, regimes), show_tqdm=show)
    progress.assert_called_once_with(
        total=3, desc="Vintages", unit="vintage", disable=not show
    )
    assert progress.return_value.__enter__.return_value.update.call_count == 3
    assert child_progress.call_count == 2
    assert all(call.kwargs["disable"] for call in child_progress.call_args_list)


def test_parallel_regimes_match_serial_with_cv_and_preserve_ids():
    if cpu_count() < 2:
        pytest.skip("Requires two cores")
    df, _ = make_panel("M")
    kwargs = dict(
        models={"ridge": Ridge()},
        hyperparameters={"ridge": {"alpha": [1, 2]}},
        scorers={"mse": make_scorer(mean_squared_error, greater_is_better=False)},
        inner_splitters={"cv": ExpandingKFoldPanelSplit(n_splits=2)},
        test_size=3,
    )
    serial = calculate(make_dynamic(df), **kwargs)
    parallel = calculate(make_dynamic(df), n_jobs_vintages=2, **kwargs)
    for table in serial._RESULT_TABLES:
        pd.testing.assert_frame_equal(getattr(serial, table), getattr(parallel, table))
    pd.testing.assert_frame_equal(
        serial.get_regimes("signal"), parallel.get_regimes("signal")
    )
    assert list(parallel.optimizers["signal"]) == ["summer", 17]
    for key, child in parallel.optimizers["signal"].items():
        pd.testing.assert_frame_equal(
            child.get_optimized_signals(),
            serial.optimizers["signal"][key].get_optimized_signals(),
        )


class RejectMultipleTargets(LinearRegression):
    def fit(self, X, y, **kwargs):
        if y.ndim == 2 and y.shape[1] > 1:
            raise ValueError("Multiple targets rejected for this test")
        return super().fit(X, y, **kwargs)


def test_parallel_failure_preserves_results_children_and_reports():
    if cpu_count() < 2:
        pytest.skip("Requires two cores")
    df, _ = make_panel("M")
    optimizer = calculate(make_dynamic(df))
    original = {
        table: getattr(optimizer, table).copy(deep=True)
        for table in optimizer._RESULT_TABLES
    }
    report = optimizer.get_regimes("signal")
    children = optimizer.optimizers["signal"]
    with pytest.raises(ValueError, match="Multiple targets rejected"):
        calculate(
            optimizer,
            models={"bad": RejectMultipleTargets()},
            n_jobs_vintages=2,
            skip_validation=True,
        )
    for table, frame in original.items():
        pd.testing.assert_frame_equal(getattr(optimizer, table), frame)
    pd.testing.assert_frame_equal(optimizer.get_regimes("signal"), report)
    assert optimizer.optimizers["signal"] is children


@pytest.mark.parametrize("n_jobs_vintages", [1, 2])
def test_portfolio_network_with_changing_target_counts(n_jobs_vintages):
    if cpu_count() < n_jobs_vintages:
        pytest.skip("Insufficient cores")
    torch = pytest.importorskip("torch")
    from macrosynergy.learning import MLPRegressor, NegMeanPortfolioReturn

    df, _ = make_panel("M")
    third = df.loc[df.xcat == "Y"].copy()
    third["xcat"] = "W"
    third["value"] = -third.value
    df = pd.concat([df, third], ignore_index=True)
    regimes = make_regimes()
    regimes[17]["targets"] = ["Y", "Z", "W"]
    network = MLPRegressor(
        n_latent=4,
        epochs=1,
        patience=None,
        batch_size=8,
        min_samples=3,
        signal_modifier=torch.nn.Softmax(dim=1),
        loss_func=NegMeanPortfolioReturn(),
        random_state=42,
    )
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        optimizer = calculate(
            DynamicSignalOptimizer(
                df,
                ["F1", "F2", "Y", "Z", "W", "MISSING"],
                n_targets=4,
                cids=["A"],
                freq="M",
                drop_nas="X",
                regimes=regimes,
            ),
            models={"network": network},
            n_jobs_vintages=n_jobs_vintages,
            test_size=3,
            skip_validation=True,
            store_additional_data=["targets"],
        )
    finally:
        torch.set_num_threads(threads)
    weights = optimizer.get_optimized_signals("signal")
    assert np.isfinite(weights.value).all()
    np.testing.assert_allclose(weights.groupby("real_date").value.sum(), 1, atol=1e-6)
    assert set(weights.loc[weights.real_date >= "2018-09-28", "xcat"]) == {
        "Y_signal",
        "Z_signal",
        "W_signal",
    }
    assert not hasattr(network, "targets")


def test_collection_preserves_categories_and_child_results():
    from macrosynergy.securities.dynamic_signal_optimizer import _RegimeResult

    df, _ = make_panel("M")
    optimizer = calculate(make_dynamic(df))
    children = optimizer.optimizers["signal"]
    before = {
        key: {
            table: getattr(child, table).copy(deep=True)
            for table in optimizer._RESULT_TABLES
        }
        for key, child in children.items()
    }
    results = {
        key: _RegimeResult(
            targets=tuple(child.xcats[-child.n_targets :]), optimizer=child
        )
        for key, child in children.items()
    }
    collected = optimizer._collect_results(results, "signal")
    for table, frame in collected.items():
        stored = getattr(optimizer, table)
        assert list(frame.select_dtypes("category")) == list(
            stored.select_dtypes("category")
        )
        pd.testing.assert_frame_equal(frame, stored, check_categorical=False)
    for key, child in children.items():
        for table, frame in before[key].items():
            pd.testing.assert_frame_equal(getattr(child, table), frame)


def test_gaps_between_regimes_have_no_signals():
    df, _ = make_panel("M")
    regimes = make_regimes()
    regimes[17]["start"] = "2018-10-31"
    optimizer = calculate(make_dynamic(df, regimes))
    assert optimizer.preds.loc[
        optimizer.preds.real_date.between("2018-09-28", "2018-10-30")
    ].empty
    assert not optimizer.preds.loc[optimizer.preds.real_date >= "2018-10-31"].empty


def test_end_return_later_than_end_does_not_extend_input_history():
    df, _ = make_panel("M")
    regimes = make_regimes()
    regimes["summer"]["end_return"] = "2019-12-31"
    optimizer = calculate(make_dynamic(df, regimes))
    child = optimizer.optimizers["signal"]["summer"]
    assert child.df.real_date.max() == pd.Timestamp("2018-09-28")
    assert optimizer.get_regimes("signal").loc["summer", "history_months"] == 9


def test_report_is_a_copy_and_unrun_process_is_rejected():
    df, _ = make_panel("M")
    optimizer = make_dynamic(df)
    with pytest.raises(ValueError, match="No regime results"):
        optimizer.get_regimes("signal")
    calculate(optimizer)
    report = optimizer.get_regimes("signal")
    report.loc["summer", "status"] = "changed"
    assert optimizer.get_regimes("signal").loc["summer", "status"] == "fitted"
