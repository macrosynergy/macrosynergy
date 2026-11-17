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
    DynamicSignalOptimizer,
    ExpandingKFoldPanelSplit,
    SignalOptimizer,
)

MODULE = "macrosynergy.learning.sequential.dynamic_signal_optimizer"


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


def test_scheduled_targets_match_manual_runs():
    df, dates = make_panel()
    boundary = dates[22]
    dynamic = calculate(
        DynamicSignalOptimizer(
            df,
            ["F1", "F2", "Y", "Z"],
            n_targets=2,
            freq="D",
            cids=["A"],
            target_schedule={dates[0]: ["Y", "Z"], boundary: ["Z"]},
        ),
        test_size=3,
    )
    old = calculate(
        SignalOptimizer(df, ["F1", "F2", "Y", "Z"], n_targets=2, cids=["A"], freq="D"),
        test_size=3,
    )
    new = calculate(
        SignalOptimizer(df, ["F1", "F2", "Z"], cids=["A"], freq="D"), test_size=3
    )
    new.preds["xcat"] = "Z_signal"
    assert_signals(
        dynamic.get_optimized_signals("signal"),
        pd.concat(
            [
                old.preds.loc[old.preds.real_date < boundary],
                new.preds.loc[new.preds.real_date >= boundary],
            ]
        ),
    )
    assert_signals(
        dynamic.get_optimized_signals("Z_signal"),
        dynamic.preds.loc[dynamic.preds.xcat == "Z_signal"],
    )
    with patch("matplotlib.pyplot.show"):
        child = dynamic.optimizers["signal"][0]
        child.feature_selection_heatmap("signal")
        child.models_heatmap("signal")


def test_empty_selection_suspends_signals_and_reentry_uses_native_clock():
    df, dates = make_panel()
    dynamic = calculate(
        DynamicSignalOptimizer(
            df,
            ["F1", "Y"],
            freq="D",
            target_schedule={dates[8]: ["Y"], dates[20]: [], dates[30]: ["Y"]},
        ),
        test_size=3,
    )
    ordinary = calculate(SignalOptimizer(df, ["F1", "Y"], freq="D"), test_size=3)
    expected = ordinary.preds.loc[
        ordinary.preds.real_date.between(dates[8], dates[19])
        | ordinary.preds.real_date.ge(dates[30])
    ]
    assert_signals(dynamic.preds, expected)


def test_boolean_and_dict_schedules_agree_without_rounding():
    df, _ = make_panel("M")
    mapping = {"2018-01-01": ["Y"], "2019-06-30": ["Z"]}
    boolean = pd.DataFrame(
        [[True, False], [False, True]], index=list(mapping), columns=["Y", "Z"]
    )
    opts = dict(df=df, xcats=["F1", "Y", "Z"], n_targets=2, freq="M")
    one = DynamicSignalOptimizer(**opts, target_schedule=mapping).get_regimes()
    two = DynamicSignalOptimizer(**opts, target_schedule=boolean).get_regimes()
    pd.testing.assert_frame_equal(one, two)
    assert one.iloc[1].start == pd.Timestamp("2019-06-30")


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


def test_rerun_replaces_results_and_failure_preserves_previous_results():
    df, dates = make_panel()
    dynamic = DynamicSignalOptimizer(
        df,
        ["F1", "Y", "Z"],
        n_targets=2,
        freq="D",
        target_schedule={dates[0]: ["Y"], dates[20]: ["Y", "Z"]},
    )
    calculate(dynamic, test_size=10)
    original = {
        key: getattr(dynamic, key).copy(deep=True) for key in dynamic._RESULT_TABLES
    }
    calculate(dynamic, test_size=10)
    for key, frame in original.items():
        pd.testing.assert_frame_equal(getattr(dynamic, key), frame)
    with patch.object(
        SignalOptimizer, "calculate_predictions", side_effect=ValueError("fit failed")
    ):
        with pytest.raises(ValueError, match="fit failed"):
            calculate(dynamic)
    for key, frame in original.items():
        pd.testing.assert_frame_equal(getattr(dynamic, key), frame)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_schedule": {"2018-01-01": ["F1"]}},
        {"target_schedule": {"2018-01-01": "Y"}},
        {"target_schedule": {"not-a-date": ["Y"]}},
        {"target_schedule": {1: ["Y"]}},
        {"target_schedule": {}},
        {"n_targets": 2},
        {"n_targets": True},
    ],
)
def test_schedule_validation_uses_pydantic(kwargs):
    df, _ = make_panel()
    with pytest.raises(ValidationError):
        DynamicSignalOptimizer(df, ["F1", "Y"], freq="D", **kwargs)


def test_no_active_targets():
    df, dates = make_panel()
    dynamic = DynamicSignalOptimizer(
        df, ["F1", "Y"], freq="D", target_schedule={dates[0]: []}
    )
    with pytest.raises(ValueError, match="No active targets"):
        calculate(dynamic)


def test_start_date_retains_underlying_meaning():
    df, dates = make_panel()
    opts = dict(start_date=str(dates[20].date()))
    dynamic = calculate(DynamicSignalOptimizer(df, ["F1", "Y"], freq="D"), **opts)
    ordinary = calculate(SignalOptimizer(df, ["F1", "Y"], freq="D"), **opts)
    assert_signals(dynamic.preds, ordinary.preds)


@pytest.mark.parametrize("n_jobs_vintages", [1, 2])
def test_portfolio_network_accepts_changing_output_dimension(n_jobs_vintages):
    if cpu_count() < n_jobs_vintages:
        pytest.skip("Insufficient cores for parallel vintages")
    torch = pytest.importorskip("torch")
    from macrosynergy.learning import MLPRegressor, NegMeanPortfolioReturn

    df, dates = make_panel()
    third = df.loc[df.xcat == "Y"].copy()
    third["xcat"] = "W"
    third["value"] = -third.value
    df = pd.concat([df, third], ignore_index=True)
    df.loc[(df.xcat == "W") & (df.real_date < dates[6]), "value"] = np.nan
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
        dynamic = calculate(
            DynamicSignalOptimizer(
                df,
                ["F1", "F2", "Y", "Z", "W"],
                n_targets=3,
                cids=["A"],
                freq="D",
                drop_nas="X",
                target_schedule={dates[0]: ["Y", "W"], dates[30]: ["Y", "Z", "W"]},
            ),
            models={"network": network},
            n_jobs_vintages=n_jobs_vintages,
            test_size=15,
            skip_validation=True,
            store_additional_data=["targets"],
        )
    finally:
        torch.set_num_threads(threads)
    weights = dynamic.get_optimized_signals("signal")
    assert np.isfinite(weights.value).all()
    np.testing.assert_allclose(weights.groupby("real_date").value.sum(), 1, atol=1e-6)
    assert weights.loc[weights.xcat == "Z_signal", "real_date"].min() == dates[30]
    assert not hasattr(network, "targets")
    assert all("targets" in entry for entry in dynamic.chosen_models.additional_data)


def test_future_returns_do_not_change_earlier_signals():
    df, dates = make_panel()
    opts = dict(
        xcats=["F1", "Y", "Z"],
        n_targets=2,
        freq="D",
        target_schedule={dates[0]: ["Y"], dates[20]: ["Y", "Z"]},
    )
    original = calculate(DynamicSignalOptimizer(df, **opts))
    changed = df.copy()
    changed.loc[changed.real_date > dates[30], "value"] *= 100
    future = calculate(DynamicSignalOptimizer(changed, **opts))
    assert_signals(
        original.preds.loc[original.preds.real_date < dates[30]],
        future.preds.loc[future.preds.real_date < dates[30]],
    )


def test_targets_with_nonoverlapping_histories():
    df, dates = make_panel()
    df.loc[(df.xcat == "Y") & (df.real_date >= dates[20]), "value"] = np.nan
    df.loc[(df.xcat == "Z") & (df.real_date < dates[20]), "value"] = np.nan
    dynamic = calculate(
        DynamicSignalOptimizer(
            df,
            ["F1", "Y", "Z"],
            n_targets=2,
            freq="D",
            target_schedule={dates[0]: ["Y"], dates[20]: ["Z"]},
        )
    )
    assert set(dynamic.preds.xcat) == {"Y_signal", "Z_signal"}
    assert (
        dynamic.preds.loc[dynamic.preds.xcat == "Y_signal", "real_date"].max()
        < dates[20]
    )
    assert (
        dynamic.preds.loc[dynamic.preds.xcat == "Z_signal", "real_date"].min()
        >= dates[20]
    )


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
    from macrosynergy.learning.sequential.dynamic_signal_optimizer import (
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


@pytest.mark.parametrize("show", [True, False])
def test_one_vintage_progress_bar_and_no_child_bars(show):
    df, dates = make_panel()
    optimizer = DynamicSignalOptimizer(
        df,
        ["F1", "Y", "Z"],
        n_targets=2,
        freq="D",
        target_schedule={dates[0]: ["Y"], dates[15]: [], dates[25]: ["Y", "Z"]},
    )
    # Real child fitting, with spies on only the progress displays.
    with patch(f"{MODULE}.tqdm") as progress, patch(
        "macrosynergy.learning.sequential.base_panel_learner.tqdm",
        side_effect=lambda iterable, **kwargs: iterable,
    ) as child_progress:
        calculate(optimizer, show_tqdm=show, test_size=20)
    progress.assert_called_once_with(
        total=2, desc="Vintages", unit="vintage", disable=not show
    )
    assert progress.return_value.__enter__.return_value.update.call_count == 2
    assert child_progress.call_count == 2
    assert all(call.kwargs["disable"] for call in child_progress.call_args_list)


def test_parallel_vintages_match_serial_with_cv():
    if cpu_count() < 2:
        pytest.skip("Requires two cores")
    df, dates = make_panel()
    init = dict(
        df=df,
        xcats=["F1", "Y", "Z"],
        n_targets=2,
        freq="D",
        target_schedule={dates[0]: ["Y"], dates[20]: ["Y", "Z"]},
    )
    kwargs = dict(
        models={"ridge": Ridge()},
        hyperparameters={"ridge": {"alpha": [1, 2]}},
        scorers={"mse": make_scorer(mean_squared_error, greater_is_better=False)},
        inner_splitters={"cv": ExpandingKFoldPanelSplit(n_splits=2)},
        min_periods=10,
        test_size=20,
    )
    serial = calculate(DynamicSignalOptimizer(**init), **kwargs)
    parallel = calculate(
        DynamicSignalOptimizer(**init),
        **kwargs,
        n_jobs_vintages=2,
    )
    for table in serial._RESULT_TABLES:
        pd.testing.assert_frame_equal(getattr(serial, table), getattr(parallel, table))
    assert [so.n_targets for so in parallel.optimizers["signal"]] == [1, 2]
    for serial_child, parallel_child in zip(
        serial.optimizers["signal"], parallel.optimizers["signal"]
    ):
        pd.testing.assert_frame_equal(
            serial_child.get_optimized_signals(), parallel_child.get_optimized_signals()
        )
    assert not hasattr(kwargs["models"]["ridge"], "coef_")
    assert kwargs["inner_splitters"]["cv"].n_splits == 2


class RejectMultipleTargets(LinearRegression):
    def fit(self, X, y, **kwargs):
        if y.ndim == 2 and y.shape[1] > 1:
            raise ValueError("Multiple targets rejected for this test")
        return super().fit(X, y, **kwargs)


def test_parallel_failure_preserves_stored_results():
    if cpu_count() < 2:
        pytest.skip("Requires two cores")
    df, dates = make_panel()
    optimizer = DynamicSignalOptimizer(
        df,
        ["F1", "Y", "Z"],
        n_targets=2,
        freq="D",
        target_schedule={dates[0]: ["Y"], dates[20]: ["Y", "Z"]},
    )
    calculate(optimizer, test_size=20)
    original = {
        key: getattr(optimizer, key).copy(deep=True) for key in optimizer._RESULT_TABLES
    }
    original_children = optimizer.optimizers["signal"]
    with pytest.raises(ValueError, match="Multiple targets rejected"):
        calculate(
            optimizer,
            models={"bad": RejectMultipleTargets()},
            n_jobs_vintages=2,
            test_size=20,
            skip_validation=True,
        )
    for key, frame in original.items():
        pd.testing.assert_frame_equal(getattr(optimizer, key), frame)
    assert optimizer.optimizers["signal"] is original_children


def test_child_parallelism_and_progress_are_hard_coded():
    from macrosynergy.learning.sequential.dynamic_signal_optimizer import _fit_vintage

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


def test_schedule_boundaries_preserve_target_order_and_merge_repeated_selections():
    df, dates = make_panel()
    schedule = {
        dates[20]: ["Z"],
        dates[15]: [],
        dates[10]: ["Z", "Y"],
        dates[0] - pd.Timedelta(days=10): ["Y", "Z"],
        dates[21]: ["Z"],
        dates[-1] + pd.Timedelta(days=10): ["Y"],
    }
    optimizer = DynamicSignalOptimizer(
        df, ["F1", "Y", "Z"], n_targets=2, freq="D", target_schedule=schedule
    )
    expected = pd.DataFrame(
        {
            "start": [dates[0], dates[15], dates[20]],
            "end": [dates[15], dates[20], pd.NaT],
            "targets": [("Y", "Z"), (), ("Z",)],
        }
    )
    pd.testing.assert_frame_equal(optimizer.get_regimes(), expected)
    assert list(schedule)[0] == dates[20]  # Input was not sorted in place.


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


def test_construction_and_schedule_inspection_do_not_construct_a_learner():
    df, dates = make_panel()
    with patch(
        f"{MODULE}.SignalOptimizer", side_effect=AssertionError("Unexpected learner")
    ):
        optimizer = DynamicSignalOptimizer(
            df,
            ["F1", "Y", "Z"],
            n_targets=2,
            freq="D",
            target_schedule={dates[0]: ["Y"], dates[20]: ["Z"]},
        )
        assert len(optimizer.get_regimes()) == 2
        assert optimizer.get_optimized_signals().empty
        assert optimizer.get_optimal_models().empty
    assert not isinstance(optimizer, SignalOptimizer)
    for attribute in ("run", "X", "y", "forecast_idxs", "store_split_data"):
        assert not hasattr(optimizer, attribute)


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


def test_schedule_window_uses_selected_raw_data_and_explicit_bounds():
    df, dates = make_panel()
    # An unrelated category must not extend the scheduling window.
    extra = df.iloc[:1].copy()
    extra["xcat"] = "UNUSED"
    extra["real_date"] = dates[-1] + pd.Timedelta(days=365)
    df = pd.concat([df, extra], ignore_index=True)
    optimizer = DynamicSignalOptimizer(
        df,
        ["F1", "Y", "Z"],
        n_targets=2,
        cids=["A"],
        freq="M",
        start=str(dates[5].date()),
        end=str(dates[30].date()),
        target_schedule={dates[0]: ["Y"], dates[20]: ["Z"], dates[35]: ["Y"]},
    )
    plan = optimizer.get_regimes()
    assert list(plan.start) == [dates[5], dates[20]]
    assert list(plan.targets) == [("Y",), ("Z",)]


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
