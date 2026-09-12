from typing import List

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from macrosynergy.learning import ExpandingIncrementPanelSplit, RollingKFoldPanelSplit
from macrosynergy.learning.sequential.base_panel_learner import BasePanelLearner
from tests.simulate import make_qdf


@pytest.fixture
def outer_splitter():
    return ExpandingIncrementPanelSplit(
        train_intervals=1,
        test_size=1,
        start_date="2019-01-31",
    )


@pytest.fixture
def scorers():
    return {"R2": make_scorer(r2_score, greater_is_better=True)}


@pytest.fixture
def models_and_hyperparameters():
    models = {"lr": LinearRegression(), "rr": Ridge()}

    hparams = {
        "lr": {"positive": [True, False], "fit_intercept": [True, False]},
        "rr": {"positive": [True, False], "fit_intercept": [True, False]},
    }

    return models, hparams


@pytest.fixture
def pipeline_models_and_hyperparameters():
    models = {
        "ridge_pipe": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge())]
        ),
    }

    hparams = {
        "ridge_pipe": {"model__alpha": [0.1, 1.0, 10.0]},
    }

    return models, hparams


@pytest.fixture
def inner_splitter():
    return {"rolling": RollingKFoldPanelSplit(n_splits=5)}


@pytest.fixture
def df():
    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["factor1", "factor2", "return"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2014-01-01", "2025-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2015-01-01", "2025-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2015-01-01", "2025-12-31", 0, 1]
    df_cids.loc["USD"] = ["2015-01-01", "2025-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["factor1"] = ["2014-01-01", "2025-12-31", 0, 1, 0.9, 0.5]
    df_xcats.loc["factor2"] = ["2015-01-01", "2025-12-31", 1, 2, 0.95, 1]
    df_xcats.loc["return"] = ["2014-01-01", "2025-12-31", 0.1, 1, 0, 0.3]

    return make_qdf(df_cids, df_xcats, back_ar=0.75)


class PanelLearner(BasePanelLearner):
    def __init__(self, df: pd.DataFrame, xcats: List[str]):
        super().__init__(df, xcats)


def _run(pl, models, hparams, inner_splitter, scorers, outer_splitter, **kwargs):
    return pl.run(
        name="test",
        models=models,
        inner_splitters=inner_splitter,
        hyperparameters=hparams,
        scorers=scorers,
        outer_splitter=outer_splitter,
        **kwargs,
    )


class TestRun:
    def test_default_selects_every_period(
        self,
        df,
        models_and_hyperparameters,
        inner_splitter,
        scorers,
        outer_splitter,
    ):
        pl = PanelLearner(df=df, xcats=["factor1", "factor2", "return"])

        results = pl.run(
            name="test",
            models=models_and_hyperparameters[0],
            hyperparameters=models_and_hyperparameters[1],
            inner_splitters=inner_splitter,
            scorers=scorers,
            outer_splitter=outer_splitter,
        )

        # With no selection_freq, model selection (CV) runs every period, so every
        # period records a non-zero CV score.
        assert all(result["model_choice"][2] != 0 for result in results)

    def test_selection_freq_none_matches_freq_one(
        self,
        df,
        models_and_hyperparameters,
        inner_splitter,
        scorers,
        outer_splitter,
    ):
        pl = PanelLearner(df=df, xcats=["factor1", "factor2", "return"])

        results_one = pl.run(
            name="test",
            models=models_and_hyperparameters[0],
            hyperparameters=models_and_hyperparameters[1],
            inner_splitters=inner_splitter,
            scorers=scorers,
            outer_splitter=outer_splitter,
            selection_freq=1,
        )

        results_none = pl.run(
            name="test",
            models=models_and_hyperparameters[0],
            hyperparameters=models_and_hyperparameters[1],
            inner_splitters=inner_splitter,
            scorers=scorers,
            outer_splitter=outer_splitter,
            selection_freq=None,
        )
        results_none = list(results_none)

        assert results_one == results_none

    def test_model_selection_frequency(
        self,
        df,
        models_and_hyperparameters,
        inner_splitter,
        scorers,
        outer_splitter,
    ):
        pl = PanelLearner(df=df, xcats=["factor1", "factor2", "return"])

        freq = 12
        results = pl.run(
            name="test",
            models=models_and_hyperparameters[0],
            hyperparameters=models_and_hyperparameters[1],
            inner_splitters=inner_splitter,
            scorers=scorers,
            outer_splitter=outer_splitter,
            selection_freq=freq,
        )

        # Model selection (CV) only happens every 12 periods. Selection periods
        # record a non-zero score; reuse periods record a zero score.
        for i, result in enumerate(results):
            cv_score = result["model_choice"][2]
            if (i % freq) == 0:
                assert cv_score != 0
            else:
                assert cv_score == 0

        # Each reuse period must report the same model name and hyperparameters as
        # its block's selection period.
        for i, result in enumerate(results):
            block_start = (i // freq) * freq
            assert result["model_choice"][1] == results[block_start]["model_choice"][1]
            assert result["model_choice"][3] == results[block_start]["model_choice"][3]

    def test_pipeline_model_reuse(
        self,
        df,
        pipeline_models_and_hyperparameters,
        inner_splitter,
        scorers,
        outer_splitter,
    ):
        pl = PanelLearner(df=df, xcats=["factor1", "factor2", "return"])

        results = pl.run(
            name="test",
            models=pipeline_models_and_hyperparameters[0],
            inner_splitters=inner_splitter,
            hyperparameters=pipeline_models_and_hyperparameters[1],
            scorers=scorers,
            outer_splitter=outer_splitter,
            selection_freq=12,
        )

        # Reuse periods must rebuild the pipeline from nested (set_params-style)
        # hyperparameters without error and report those hyperparameters.
        for i, result in enumerate(results):
            block_start = (i // 12) * 12
            assert result["model_choice"][1] == "ridge_pipe"
            assert set(result["model_choice"][3]) == {"model__alpha"}
            assert result["model_choice"][3] == results[block_start]["model_choice"][3]

    def test_reuse_periods_refit_with_selected_hyperparameters(
        self,
        df,
        inner_splitter,
        scorers,
        outer_splitter,
    ):
        pl = PanelLearner(df=df, xcats=["factor1", "factor2", "return"])

        # A single-value grid forces the selected alpha to 1e6, which differs from
        # Ridge's default of 1.0. Reuse periods must refit the model with the
        # selected alpha rather than the constructor default. The actually-fit alpha
        # is captured via store_additional_data, so this checks the model itself
        # rather than the recorded metadata.
        freq = 4
        results = pl.run(
            name="test",
            models={"rr": Ridge()},
            hyperparameters={"rr": {"alpha": [1_000_000.0]}},
            inner_splitters=inner_splitter,
            scorers=scorers,
            outer_splitter=outer_splitter,
            selection_freq=freq,
            store_additional_data=["alpha"],
        )

        assert len(results) > freq
        for result in results:
            assert result["model_choice"][4]["alpha"] == 1_000_000.0

    def test_selection_freq_larger_than_n_periods(
        self,
        df,
        models_and_hyperparameters,
        inner_splitter,
        scorers,
        outer_splitter,
    ):
        pl = PanelLearner(df=df, xcats=["factor1", "factor2", "return"])

        results = pl.run(
            name="test",
            models=models_and_hyperparameters[0],
            hyperparameters=models_and_hyperparameters[1],
            inner_splitters=inner_splitter,
            scorers=scorers,
            outer_splitter=outer_splitter,
            selection_freq=10_000,
        )

        # With a frequency exceeding the number of periods, selection happens once
        # (period 0) and every subsequent period reuses that single selection.
        assert results[0]["model_choice"][2] != 0
        assert all(result["model_choice"][2] == 0 for result in results[1:])
        assert all(
            result["model_choice"][1] == results[0]["model_choice"][1]
            for result in results
        )
        assert all(
            result["model_choice"][3] == results[0]["model_choice"][3]
            for result in results
        )


class TestRunValidation:
    @pytest.mark.parametrize("selection_freq", [0, -1, -12])
    def test_non_positive_selection_freq_raises(
        self,
        df,
        models_and_hyperparameters,
        inner_splitter,
        scorers,
        outer_splitter,
        selection_freq,
    ):
        pl = PanelLearner(df=df, xcats=["factor1", "factor2", "return"])

        with pytest.raises(ValueError):
            pl.run(
                name="test",
                models=models_and_hyperparameters[0],
                inner_splitters=inner_splitter,
                hyperparameters=models_and_hyperparameters[1],
                scorers=scorers,
                outer_splitter=outer_splitter,
                selection_freq=selection_freq,
            )

    @pytest.mark.parametrize("selection_freq", [1.5, "12", True])
    def test_non_integer_selection_freq_raises(
        self,
        df,
        models_and_hyperparameters,
        inner_splitter,
        scorers,
        outer_splitter,
        selection_freq,
    ):
        pl = PanelLearner(df=df, xcats=["factor1", "factor2", "return"])

        with pytest.raises(TypeError):
            pl.run(
                name="test",
                models=models_and_hyperparameters[0],
                inner_splitters=inner_splitter,
                hyperparameters=models_and_hyperparameters[1],
                scorers=scorers,
                outer_splitter=outer_splitter,
                selection_freq=selection_freq,
            )
