import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from macrosynergy.learning.preprocessing.imputers.imputers import EstimatorImputer

pytestmark = pytest.mark.skipif(
    Version(sklearn.__version__) < Version("1.4"),
    reason="Need a version of sklearn that allows NaNs in input data",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
CIDS = ("AUD", "CAD", "GBP", "EUR")
DATES = pd.date_range("2020-01-01", periods=12, freq="M")


def make_panel(
    cids=CIDS,
    dates=DATES,
    cols=("feature_a", "feature_b", "feature_c"),
    random_state=42,
) -> pd.DataFrame:
    """Return a well-formed panel DataFrame with a MultiIndex (cid, real_date)."""
    rng = np.random.default_rng(random_state)
    idx = pd.MultiIndex.from_product([cids, dates], names=["cid", "real_date"])
    data = rng.standard_normal((len(idx), len(cols)))
    return pd.DataFrame(data, index=idx, columns=list(cols))


def inject_nan(df: pd.DataFrame, col: str, frac: float, random_state=0) -> pd.DataFrame:
    """Return a copy of df with `frac` of `col` set to NaN."""
    df = df.copy()
    mask = df[col].sample(frac=frac, random_state=random_state).index
    df.loc[mask, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def clean_panel():
    return make_panel()


@pytest.fixture
def panel_with_nans():
    df = make_panel()
    df = inject_nan(df, "feature_a", frac=0.3)
    df = inject_nan(df, "feature_b", frac=0.2)
    return df


@pytest.fixture
def fitted_imputer(panel_with_nans):
    imp = EstimatorImputer(estimator=RandomForestRegressor(random_state=0))
    imp.fit(panel_with_nans)
    return imp


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
class TestInputValidation:
    def test_rejects_non_dataframe(self):
        imp = EstimatorImputer()
        with pytest.raises(TypeError):
            imp.fit(np.ones((10, 3)))

    def test_rejects_wrong_index_names(self):
        imp = EstimatorImputer()
        df = pd.DataFrame(
            np.ones((6, 2)),
            index=pd.MultiIndex.from_product(
                [["A"], range(6)], names=["country", "date"]
            ),
            columns=["x", "y"],
        )
        with pytest.raises(ValueError):
            imp.fit(df)

    def test_transform_before_fit_raises(self, clean_panel):
        imp = EstimatorImputer()
        with pytest.raises(NotFittedError):
            imp.transform(clean_panel)

    def test_transform_mismatched_columns_raises(self, fitted_imputer):
        bad = make_panel(cols=("feature_a", "feature_b", "WRONG"))
        with pytest.raises(ValueError, match="Input columns differ"):
            fitted_imputer.transform(bad)


# ---------------------------------------------------------------------------
# Fit behaviour
# ---------------------------------------------------------------------------
class TestFitBehaviour:
    def test_fit_returns_self(self, panel_with_nans):
        imp = EstimatorImputer(estimator=RandomForestRegressor(random_state=0))
        assert imp.fit(panel_with_nans) is imp

    def test_feature_names_in_set(self, panel_with_nans):
        imp = EstimatorImputer(estimator=RandomForestRegressor(random_state=0)).fit(
            panel_with_nans
        )
        assert list(imp.feature_names_in_) == list(panel_with_nans.columns)

    def test_models_trained_only_for_missing_features(self, panel_with_nans):
        imp = EstimatorImputer(estimator=RandomForestRegressor(random_state=0)).fit(
            panel_with_nans
        )
        assert "feature_c" not in imp.models_
        assert "feature_a" in imp.models_
        assert "feature_b" in imp.models_

    def test_no_models_when_no_missing_values(self, clean_panel):
        imp = EstimatorImputer(estimator=RandomForestRegressor(random_state=0)).fit(
            clean_panel
        )
        assert imp.models_ == {}

    def test_predictor_means_stored(self, panel_with_nans):
        imp = EstimatorImputer(estimator=RandomForestRegressor(random_state=0)).fit(
            panel_with_nans
        )
        assert isinstance(imp.predictor_means_, pd.Series)
        assert set(imp.predictor_means_.index) == set(imp.kept_features_)

    def test_missing_fraction_by_col(self, panel_with_nans):
        imp = EstimatorImputer(estimator=RandomForestRegressor(random_state=0)).fit(
            panel_with_nans
        )
        assert isinstance(imp.missing_fraction_by_col_, pd.Series)
        assert imp.missing_fraction_by_col_["feature_c"] == pytest.approx(0.0)
        assert imp.missing_fraction_by_col_["feature_a"] > 0.0

    def test_missing_fraction_by_cid_and_col(self, panel_with_nans):
        imp = EstimatorImputer(estimator=RandomForestRegressor(random_state=0)).fit(
            panel_with_nans
        )
        df = imp.missing_fraction_by_cid_and_col_
        assert isinstance(df, pd.DataFrame)
        assert set(df.index) == set(CIDS)


# ---------------------------------------------------------------------------
# NaN threshold / column dropping
# ---------------------------------------------------------------------------
class TestNanThreshold:
    def test_all_nan_column_dropped_by_default(self, clean_panel):
        df = clean_panel.copy()
        df["all_nan"] = np.nan
        imp = EstimatorImputer(estimator=RandomForestRegressor(random_state=0)).fit(df)
        assert "all_nan" in imp.dropped_features_
        assert "all_nan" not in imp.kept_features_

    def test_dropped_column_absent_from_output(self, clean_panel):
        df = clean_panel.copy()
        df["all_nan"] = np.nan
        out = EstimatorImputer(
            estimator=RandomForestRegressor(random_state=0)
        ).fit_transform(df)
        assert "all_nan" not in out.columns

    def test_custom_nan_threshold_drops_partial_nan_column(self):
        df = make_panel(cols=("feature_a", "feature_b"))
        df = inject_nan(df, "feature_a", frac=0.6)
        imp = EstimatorImputer(
            estimator=RandomForestRegressor(random_state=0), nan_threshold=0.4
        ).fit(df)
        assert "feature_a" in imp.dropped_features_

    def test_n_features_out_matches_kept(self, panel_with_nans):
        imp = EstimatorImputer(estimator=RandomForestRegressor(random_state=0)).fit(
            panel_with_nans
        )
        assert imp.n_features_out_ == len(imp.kept_features_)

    def test_get_feature_names_out(self, panel_with_nans):
        imp = EstimatorImputer(estimator=RandomForestRegressor(random_state=0)).fit(
            panel_with_nans
        )
        assert list(imp.get_feature_names_out()) == imp.kept_features_


# ---------------------------------------------------------------------------
# Transform behaviour
# ---------------------------------------------------------------------------
class TestTransformBehaviour:
    def test_no_nans_in_output(self, panel_with_nans, fitted_imputer):
        out = fitted_imputer.transform(panel_with_nans)
        assert not out.isna().any().any()

    def test_output_shape(self, panel_with_nans, fitted_imputer):
        out = fitted_imputer.transform(panel_with_nans)
        assert out.shape == (len(panel_with_nans), fitted_imputer.n_features_out_)

    def test_observed_values_unchanged(self, panel_with_nans, fitted_imputer):
        out = fitted_imputer.transform(panel_with_nans)
        observed_mask = panel_with_nans[fitted_imputer.kept_features_].notna()
        pd.testing.assert_frame_equal(
            out[observed_mask],
            panel_with_nans[fitted_imputer.kept_features_][observed_mask],
        )

    def test_output_index_preserved(self, panel_with_nans, fitted_imputer):
        out = fitted_imputer.transform(panel_with_nans)
        pd.testing.assert_index_equal(out.index, panel_with_nans.index)

    def test_output_columns_match_kept_features(self, panel_with_nans, fitted_imputer):
        out = fitted_imputer.transform(panel_with_nans)
        assert list(out.columns) == fitted_imputer.kept_features_

    def test_clean_data_passes_through_unchanged(self, clean_panel):
        imp = EstimatorImputer(estimator=RandomForestRegressor(random_state=0)).fit(
            clean_panel
        )
        out = imp.transform(clean_panel)
        pd.testing.assert_frame_equal(out, clean_panel)

    def test_fit_transform_equivalent_to_fit_then_transform(self, panel_with_nans):
        imp1 = EstimatorImputer(estimator=RandomForestRegressor(random_state=0))
        out1 = imp1.fit_transform(panel_with_nans)

        imp2 = EstimatorImputer(estimator=RandomForestRegressor(random_state=0))
        imp2.fit(panel_with_nans)
        out2 = imp2.transform(panel_with_nans)

        pd.testing.assert_frame_equal(out1, out2)

    def test_imputed_values_within_training_range(self, panel_with_nans):
        imp = EstimatorImputer(estimator=RandomForestRegressor(random_state=0)).fit(
            panel_with_nans
        )
        out = imp.transform(panel_with_nans)
        for col in imp.kept_features_:
            col_min = panel_with_nans[col].min()
            col_max = panel_with_nans[col].max()
            assert out[col].between(col_min, col_max).all(), (
                f"{col}: imputed values outside training range"
            )

    def test_fallback_fills_remaining_nans(self):
        df = make_panel(cols=("sparse", "feature_b", "feature_c"))
        df["sparse"] = np.nan
        df.loc[df.index[0], "sparse"] = 1.0
        imp = EstimatorImputer(
            estimator=RandomForestRegressor(random_state=0), fallback=True
        ).fit(df)
        out = imp.transform(df)
        assert not out["sparse"].isna().any()

    def test_fallback_disabled_leaves_nans(self):
        df = make_panel(cols=("sparse", "feature_b", "feature_c"))
        df["sparse"] = np.nan
        df.loc[df.index[0], "sparse"] = 1.0
        imp = EstimatorImputer(
            estimator=RandomForestRegressor(random_state=0), fallback=False
        ).fit(df)
        out = imp.transform(df)
        assert out["sparse"].isna().any()


# ---------------------------------------------------------------------------
# Configuration: estimator handling
# ---------------------------------------------------------------------------
class TestConfiguration:
    def test_default_estimator_is_random_forest(self, panel_with_nans):
        imp = EstimatorImputer().fit(panel_with_nans)
        for model in imp.models_.values():
            assert isinstance(model, RandomForestRegressor)

    def test_custom_estimator_stored(self):
        lr = LinearRegression()
        imp = EstimatorImputer(estimator=lr)
        assert imp.estimator is lr

    def test_clone_used_per_feature(self, panel_with_nans):
        imp = EstimatorImputer(estimator=RandomForestRegressor(random_state=0)).fit(
            panel_with_nans
        )
        models = list(imp.models_.values())
        assert len(models) >= 2
        for i, m1 in enumerate(models):
            for m2 in models[i + 1 :]:
                assert m1 is not m2

    def test_pipeline_as_estimator(self):
        df = make_panel(cols=("feature_a", "feature_b", "feature_c"))
        df = inject_nan(df, "feature_a", frac=0.3)
        pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
        imp = EstimatorImputer(estimator=pipe).fit(df)
        out = imp.transform(df)
        assert not out.isna().any().any()

    def test_custom_missing_values_sentinel(self):
        df = make_panel(cols=("x", "y", "z"))
        df_sentinel = df.copy()
        df_sentinel.loc[df_sentinel.index[:5], "x"] = -999

        imp = EstimatorImputer(
            estimator=RandomForestRegressor(random_state=0), missing_values=-999
        ).fit(df_sentinel)
        out = imp.transform(df_sentinel)
        assert not out.isna().any().any()
        assert (out.loc[out.index[:5], "x"] != -999).all()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_fit_raises_when_estimator_cannot_handle_nan_predictors(self):
        df = make_panel(cols=("feature_a", "feature_b", "feature_c"))
        df = inject_nan(df, "feature_a", frac=0.3)
        df = inject_nan(df, "feature_b", frac=0.2)
        imp = EstimatorImputer(estimator=LinearRegression())
        with pytest.raises(ValueError, match="feature_a|feature_b"):
            imp.fit(df)

    def test_fit_succeeds_when_only_target_has_nans(self):
        df = make_panel(cols=("feature_a", "feature_b", "feature_c"))
        df = inject_nan(df, "feature_a", frac=0.3)
        imp = EstimatorImputer(estimator=LinearRegression()).fit(df)
        out = imp.transform(df)
        assert not out.isna().any().any()

    def test_pipeline_with_imputer_handles_nan_predictors(self):
        df = make_panel(cols=("feature_a", "feature_b", "feature_c"))
        df = inject_nan(df, "feature_a", frac=0.3)
        df = inject_nan(df, "feature_b", frac=0.2)
        pipe = Pipeline([("imp", SimpleImputer()), ("lr", LinearRegression())])
        imp = EstimatorImputer(estimator=pipe).fit(df)
        out = imp.transform(df)
        assert not out.isna().any().any()

    def test_transform_raises_when_estimator_cannot_handle_nan_predictors(self):
        df = make_panel(cols=("feature_a", "feature_b", "feature_c"))
        df_fit = inject_nan(df, "feature_a", frac=0.3)
        imp = EstimatorImputer(estimator=LinearRegression()).fit(df_fit)

        df_transform = inject_nan(df, "feature_a", frac=0.3)
        df_transform = inject_nan(df_transform, "feature_b", frac=0.2)
        with pytest.raises(ValueError, match="feature_a"):
            imp.transform(df_transform)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_single_feature_no_model_trained(self):
        df = make_panel(cols=("only_col",))
        df = inject_nan(df, "only_col", frac=0.3)
        imp = EstimatorImputer(estimator=RandomForestRegressor(random_state=0)).fit(df)
        assert imp.models_ == {}

    def test_all_rows_missing_in_one_col_dropped_at_threshold_1(self):
        df = make_panel(cols=("feature_a", "feature_b"))
        df["feature_a"] = np.nan
        imp = EstimatorImputer(
            estimator=RandomForestRegressor(random_state=0), nan_threshold=1.0
        ).fit(df)
        assert "feature_a" in imp.dropped_features_

    def test_transform_on_fully_observed_unseen_dates(self, panel_with_nans):
        imp = EstimatorImputer(estimator=RandomForestRegressor(random_state=0)).fit(
            panel_with_nans
        )
        new_dates = pd.date_range("2021-01-01", periods=6, freq="M")
        new_panel = make_panel(cids=CIDS, dates=new_dates, random_state=99)
        out = imp.transform(new_panel)
        assert not out.isna().any().any()

    def test_feature_with_single_observed_row_skipped(self):
        df = make_panel(cols=("sparse", "feature_b", "feature_c"))
        observed_idx = df.index[0]
        df["sparse"] = np.nan
        df.loc[observed_idx, "sparse"] = 1.0
        imp = EstimatorImputer(estimator=RandomForestRegressor(random_state=0))
        imp.fit(df)
        assert "sparse" not in imp.models_
