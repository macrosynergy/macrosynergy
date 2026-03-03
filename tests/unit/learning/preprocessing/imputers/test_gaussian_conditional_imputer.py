from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError

from macrosynergy.learning.preprocessing.imputers.imputers import (
    GaussianConditionalImputer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
CIDS = ("AUD", "CAD", "GBP", "EUR")
DATES = pd.date_range("2020-01-01", periods=12, freq="ME")


def make_panel(
    cids=CIDS,
    dates=DATES,
    cols=("feature_a", "feature_b", "feature_c"),
    random_state=42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    idx = pd.MultiIndex.from_product([cids, dates], names=["cid", "real_date"])
    data = rng.standard_normal((len(idx), len(cols)))
    return pd.DataFrame(data, index=idx, columns=list(cols))


def inject_nan(
    df: pd.DataFrame, col: str, frac: float, random_state: int = 0
) -> pd.DataFrame:
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
    imp = GaussianConditionalImputer()
    imp.fit(panel_with_nans)
    return imp


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
class TestInputValidation:
    def test_rejects_non_dataframe(self):
        imp = GaussianConditionalImputer()
        with pytest.raises(TypeError):
            imp.fit(np.ones((10, 3)))

    def test_rejects_wrong_index_names(self):
        imp = GaussianConditionalImputer()
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
        imp = GaussianConditionalImputer()
        with pytest.raises(NotFittedError):
            imp.transform(clean_panel)

    def test_invalid_fallback_raises(self):
        with pytest.raises(ValueError, match="fallback"):
            GaussianConditionalImputer(fallback="bad_value")

    def test_transform_mismatched_columns_raises(self, fitted_imputer):
        bad = make_panel(cols=("feature_a", "feature_b", "WRONG"))
        with pytest.raises(ValueError, match="Input columns differ"):
            fitted_imputer.transform(bad)


# ---------------------------------------------------------------------------
# Fit behaviour
# ---------------------------------------------------------------------------
class TestFitBehaviour:
    def test_fit_returns_self(self, panel_with_nans):
        imp = GaussianConditionalImputer()
        assert imp.fit(panel_with_nans) is imp

    def test_global_mean_shape(self, panel_with_nans):
        imp = GaussianConditionalImputer().fit(panel_with_nans)
        assert imp.global_mean_.shape == (len(imp.kept_features_),)

    def test_global_covariance_shape(self, panel_with_nans):
        imp = GaussianConditionalImputer().fit(panel_with_nans)
        n = len(imp.kept_features_)
        assert imp.global_covariance_.shape == (n, n)

    def test_feature_names_in_set(self, panel_with_nans):
        imp = GaussianConditionalImputer().fit(panel_with_nans)
        assert list(imp.feature_names_in_) == list(panel_with_nans.columns)


# ---------------------------------------------------------------------------
# NaN threshold
# ---------------------------------------------------------------------------
class TestNanThreshold:
    def test_all_nan_column_dropped_by_default(self, clean_panel):
        df = clean_panel.copy()
        df["all_nan"] = np.nan
        imp = GaussianConditionalImputer().fit(df)
        assert "all_nan" in imp.dropped_features_
        assert "all_nan" not in imp.kept_features_

    def test_dropped_column_absent_from_output(self, clean_panel):
        df = clean_panel.copy()
        df["all_nan"] = np.nan
        out = GaussianConditionalImputer().fit_transform(df)
        assert "all_nan" not in out.columns

    def test_custom_nan_threshold_drops_partial_nan_column(self):
        df = make_panel(cols=("feature_a", "feature_b"))
        df = inject_nan(df, "feature_a", frac=0.6)
        imp = GaussianConditionalImputer(nan_threshold=0.4).fit(df)
        assert "feature_a" in imp.dropped_features_

    def test_n_features_out_matches_kept(self, panel_with_nans):
        imp = GaussianConditionalImputer().fit(panel_with_nans)
        assert imp.n_features_out_ == len(imp.kept_features_)


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
        imp = GaussianConditionalImputer().fit(clean_panel)
        out = imp.transform(clean_panel)
        pd.testing.assert_frame_equal(out, clean_panel)

    def test_fit_transform_equivalent_to_fit_then_transform(self, panel_with_nans):
        imp1 = GaussianConditionalImputer()
        out1 = imp1.fit_transform(panel_with_nans)

        imp2 = GaussianConditionalImputer()
        imp2.fit(panel_with_nans)
        out2 = imp2.transform(panel_with_nans)

        pd.testing.assert_frame_equal(out1, out2)


# ---------------------------------------------------------------------------
# Conditional imputation correctness
# ---------------------------------------------------------------------------
class TestConditionalImputationCorrectness:
    def test_deterministic_two_feature_example(self):
        """Construct a known 2-feature Gaussian and verify the conditional mean."""
        # Known parameters: mean=[0, 0], cov=[[1, 0.8], [0.8, 1]]
        # If x_1 is observed = 1.0 and x_0 is missing:
        #   mu_0|1 = 0 + 0.8 * 1^{-1} * (1.0 - 0) = 0.8
        rng = np.random.default_rng(42)
        n = 500
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.8], [0.8, 1.0]])
        data = rng.multivariate_normal(mean, cov, size=n)

        cids = ["AUD"] * n
        dates = pd.date_range("2000-01-01", periods=n, freq="D")
        idx = pd.MultiIndex.from_arrays([cids, dates], names=["cid", "real_date"])
        df = pd.DataFrame(data, index=idx, columns=["f0", "f1"])

        # Add one row with missing f0 and known f1
        extra_idx = pd.MultiIndex.from_tuples(
            [("AUD", pd.Timestamp("2010-01-01"))], names=["cid", "real_date"]
        )
        extra_row = pd.DataFrame({"f0": [np.nan], "f1": [1.0]}, index=extra_idx)
        df_with_nan = pd.concat([df, extra_row])

        imp = GaussianConditionalImputer().fit(df_with_nan)
        out = imp.transform(df_with_nan)

        imputed_value = out.loc[extra_idx[0], "f0"]
        # The conditional mean should be close to 0.8 (not exact due to sample cov)
        assert_allclose(imputed_value, 0.8, atol=0.15)

    def test_uncorrelated_features_yield_unconditional_mean(self):
        """When features are uncorrelated, conditional mean equals unconditional mean."""
        rng = np.random.default_rng(99)
        n = 500
        data = rng.standard_normal((n, 2))

        cids = ["AUD"] * n
        dates = pd.date_range("2000-01-01", periods=n, freq="D")
        idx = pd.MultiIndex.from_arrays([cids, dates], names=["cid", "real_date"])
        df = pd.DataFrame(data, index=idx, columns=["f0", "f1"])

        extra_idx = pd.MultiIndex.from_tuples(
            [("AUD", pd.Timestamp("2010-01-01"))], names=["cid", "real_date"]
        )
        extra_row = pd.DataFrame({"f0": [np.nan], "f1": [5.0]}, index=extra_idx)
        df_with_nan = pd.concat([df, extra_row])

        imp = GaussianConditionalImputer().fit(df_with_nan)
        out = imp.transform(df_with_nan)

        imputed_value = out.loc[extra_idx[0], "f0"]
        sample_mean_f0 = df["f0"].mean()
        assert_allclose(imputed_value, sample_mean_f0, atol=0.15)


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------
class TestFallbackBehaviour:
    def test_mean_fallback_fills_remaining_nans(self):
        df = make_panel(cols=("f0", "f1"))
        df = inject_nan(df, "f0", frac=0.3)
        imp = GaussianConditionalImputer(fallback="mean").fit(df)
        out = imp.transform(df)
        assert not out.isna().any().any()

    def test_none_fallback_may_leave_nans(self):
        """With fallback='none', if conditional imputation can't fill all NaNs,
        some may remain."""
        # Create a scenario where global model has no complete rows
        idx = pd.MultiIndex.from_arrays(
            [["AUD", "AUD"], pd.date_range("2020-01-01", periods=2, freq="D")],
            names=["cid", "real_date"],
        )
        # Both rows have NaNs in different cols — no complete rows
        df = pd.DataFrame(
            {"f0": [np.nan, 1.0], "f1": [1.0, np.nan]},
            index=idx,
        )
        imp = GaussianConditionalImputer(fallback="none").fit(df)
        out = imp.transform(df)
        # With fallback="none" and degenerate data, NaNs might remain
        # (this test just checks the imputer doesn't crash)
        assert out.shape == (2, 2)

    def test_none_fallback_leaves_nans_on_linalg_failure(self):
        """When np.linalg.solve raises LinAlgError, fallback='none'
        must leave missing values as NaN rather than silently filling."""
        df = make_panel(
            cids=("AUD",), dates=pd.date_range("2020-01-01", periods=50, freq="D")
        )
        df = inject_nan(df, "feature_a", frac=0.1)

        imp = GaussianConditionalImputer(fallback="none").fit(df)

        original_solve = np.linalg.solve

        def failing_solve(a, b):
            raise np.linalg.LinAlgError("Singular matrix")

        with patch("numpy.linalg.solve", side_effect=failing_solve):
            out = imp.transform(df)

        missing_mask = df["feature_a"].isna()
        assert out.loc[missing_mask, "feature_a"].isna().all()

    def test_mean_fallback_fills_after_linalg_failure(self):
        """When np.linalg.solve raises LinAlgError, fallback='mean'
        should still fill the value via the fallback."""
        df = make_panel(
            cids=("AUD",), dates=pd.date_range("2020-01-01", periods=50, freq="D")
        )
        df = inject_nan(df, "feature_a", frac=0.1)

        imp = GaussianConditionalImputer(fallback="mean").fit(df)

        def failing_solve(a, b):
            raise np.linalg.LinAlgError("Singular matrix")

        with patch("numpy.linalg.solve", side_effect=failing_solve):
            out = imp.transform(df)

        assert not out.isna().any().any()

    def test_all_missing_row_with_none_fallback_leaves_nans(self):
        """An all-NaN row with fallback='none' should remain NaN."""
        df = make_panel(
            cids=("AUD",), dates=pd.date_range("2020-01-01", periods=50, freq="D")
        )
        df.iloc[0] = np.nan
        imp = GaussianConditionalImputer(fallback="none").fit(df)
        out = imp.transform(df)
        assert out.iloc[0].isna().all()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_single_feature(self):
        df = make_panel(cols=("only_col",))
        df = inject_nan(df, "only_col", frac=0.3)
        imp = GaussianConditionalImputer(fallback="mean").fit(df)
        out = imp.transform(df)
        assert not out.isna().any().any()

    def test_unseen_cid_uses_global_model(self):
        df_fit = make_panel(cids=("AUD", "CAD"))
        df_fit = inject_nan(df_fit, "feature_a", frac=0.2)
        imp = GaussianConditionalImputer().fit(df_fit)

        # Transform data with a cid not seen during fit
        rng = np.random.default_rng(99)
        new_idx = pd.MultiIndex.from_product(
            [["NEW_CID"], DATES], names=["cid", "real_date"]
        )
        df_new = pd.DataFrame(
            rng.standard_normal((len(new_idx), 3)),
            index=new_idx,
            columns=["feature_a", "feature_b", "feature_c"],
        )
        df_new = inject_nan(df_new, "feature_a", frac=0.3)
        out = imp.transform(df_new)
        assert not out.isna().any().any()

    def test_all_missing_row_filled_with_column_mean(self):
        df = make_panel(
            cids=("AUD",), dates=pd.date_range("2020-01-01", periods=50, freq="D")
        )
        # Make one row all NaN
        df.iloc[0] = np.nan
        imp = GaussianConditionalImputer(fallback="mean").fit(df)
        out = imp.transform(df)
        assert not out.iloc[0].isna().any()

    def test_few_global_complete_rows_still_works(self):
        """Even with < 2 complete rows globally, imputer should not crash."""
        idx = pd.MultiIndex.from_arrays(
            [["AUD"] * 5, pd.date_range("2020-01-01", periods=5, freq="D")],
            names=["cid", "real_date"],
        )
        df = pd.DataFrame(
            {
                "f0": [1.0, np.nan, 3.0, np.nan, 5.0],
                "f1": [np.nan, 2.0, np.nan, 4.0, np.nan],
            },
            index=idx,
        )
        imp = GaussianConditionalImputer(fallback="mean").fit(df)
        out = imp.transform(df)
        assert not out.isna().any().any()
