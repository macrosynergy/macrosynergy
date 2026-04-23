from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

DATE_INDEX_NAME = "real_date"
CIDS_INDEX_NAME = "cid"


class BaseImputer(BaseEstimator, TransformerMixin, ABC):
    """
    Base class for imputers operating on panel data

    Parameters
    ----------
    missing_values : int, float, str, np.nan, None or pandas.NA, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        can be set to either `np.nan` or `pd.NA`.
    nan_threshold : float, default=1.0
        If the proportion of NaNs in column is greater than this, we get rid of
        the column.

    Attributes
    ----------
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit
    missing_fraction_by_col_ : pd.Series
        Fraction of missing values for each column
    missing_fraction_by_cid_and_col_ : pd.DataFrame
        Fraction of missing values for each column split by cid
    dropped_features_ : list
        Names of features to be dropped from the data
    kept_features_ : list
        Names of features that are not dropped
    n_features_out_ : Integral
        Number of features left after transforming
    """

    def __init__(
        self,
        missing_values=np.nan,
        nan_threshold: float = 1.0,
    ):
        self.missing_values = missing_values
        self.nan_threshold = nan_threshold

    def fit(self, X: pd.DataFrame, y=None):
        X = self._validate_input(X)
        self.feature_names_in_ = X.columns.to_numpy().flatten()

        # replace missing_values with np.nan for convenience
        X_nan = X.replace(self.missing_values, np.nan)

        # compute useful reporting / diagnostic info
        nan_mask = X_nan.isna()
        self.missing_fraction_by_col_ = nan_mask.mean(axis=0)
        self.missing_fraction_by_cid_and_col_ = nan_mask.groupby(CIDS_INDEX_NAME).mean()

        # identify columns violating nan threshold
        violations = self.missing_fraction_by_col_ >= self.nan_threshold
        self.dropped_features_ = self.feature_names_in_[violations].tolist()
        self.kept_features_ = self.feature_names_in_[~violations].tolist()
        self.n_features_out_ = len(self.kept_features_)

        # let subclass learn whatever it needs to from the remaining data
        self._fit_fill_values(X=X_nan[self.kept_features_], y=y)

        return self

    def transform(self, X):
        # fit checks and input validation
        check_is_fitted(
            self,
            attributes=[
                "feature_names_in_",
                "missing_fraction_by_col_",
                "missing_fraction_by_cid_and_col_",
                "dropped_features_",
                "kept_features_",
                "n_features_out_",
            ],
        )
        X = self._validate_input(X)

        # replace missing_values with np.nan for convenience
        X = X.replace(self.missing_values, np.nan)

        incoming_cols = list(X.columns)
        expected_cols = list(self.feature_names_in_)
        if incoming_cols != expected_cols:
            raise ValueError(
                f"Input columns differ from fit-time columns.\n"
                f"Expected: {expected_cols}\n"
                f"Got: {incoming_cols}"
            )

        # let subclass do the rest on data with all nan cols removed
        X_imputed = self._transform_with_fill_values(X=X[self.kept_features_].copy())

        return X_imputed

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.kept_features_)

    # ----- hooks for subclasses -----
    @abstractmethod
    def _fit_fill_values(self, X: pd.DataFrame, y=None) -> "BaseImputer":
        """Learn imputation state from X"""
        pass

    @abstractmethod
    def _transform_with_fill_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned imputation state to X"""
        pass

    # ----- helpers -----
    def _validate_input(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas.DataFrame not {type(X)}")

        expected_idx_names = {CIDS_INDEX_NAME, DATE_INDEX_NAME}
        if set(X.index.names) - expected_idx_names:
            raise ValueError(
                f"Input dataframe must have index names {expected_idx_names}"
            )

        return X


class ConstantImputer(BaseImputer):
    """
    Class for imputing missing values with a constant

    Parameters
    ----------
    fill_value :
        Value to replace missing values with. Default is 0.
    missing_values : int, float, str, np.nan, None or pandas.NA, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        can be set to either `np.nan` or `pd.NA`.
    nan_threshold : float, default=1.0
        If the proportion of NaNs in column is greater than this, we get rid of
        the column.

    Attributes
    ----------
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit
    missing_fraction_by_col_ : pd.Series
        Fraction of missing values for each column
    missing_fraction_by_cid_and_col_ : pd.DataFrame
        Fraction of missing values for each column split by cid
    dropped_features_ : list
        Names of features to be dropped from the data
    kept_features_ : list
        Names of features that are not dropped
    n_features_out_ : Integral
        Number of features left after transforming
    """

    def __init__(self, fill_value=0, nan_threshold=1.0, missing_values=np.nan):
        super().__init__(
            nan_threshold=nan_threshold,
            missing_values=missing_values,
        )

        self.fill_value = fill_value

    def _fit_fill_values(self, X: pd.DataFrame, y=None) -> "ConstantImputer":
        # nothing to learn when filling constants
        return self

    def _transform_with_fill_values(self, X: pd.DataFrame) -> pd.DataFrame:
        # can use fillna because the base class ensures all missing values are np.nan
        return X.fillna(self.fill_value)


class CrossSectionalImputer(BaseImputer):
    """
    Impute missing values using the cross-sectional mean across *configured peers*
    at the same real_date (per feature).

    Parameters
    ----------
    peer_map : dict[str, list[str]] or None
        Mapping from target cid -> list of peer cids to use for imputation.
        Example:
            {"CAD": ["USD", "GBP", "EUR"], "USD": ["CAD", "GBP", "EUR"]}

        If None, peers default to "all other cids" (unless default_peers="none").
    default_peers : {"all", "none"}
        Behaviour for cids not present in peer_map:
          - "all": use all other cids as peers
          - "none": do not impute for that cid (unless fallback kicks in)
    fallback : {"none", "zero", "mean"}
        If "mean", any values still missing after peer-based imputation
        are filled with the global mean per feature computed at fit time. If
        "zero" values are filled with 0.
    missing_values : scalar
        Value to treat as missing (converted to np.nan internally).

    Attributes
    ----------
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit
    missing_fraction_by_col_ : pd.Series
        Fraction of missing values for each column
    missing_fraction_by_cid_and_col_ : pd.DataFrame
        Fraction of missing values for each column split by cid
    dropped_features_ : list
        Names of features to be dropped from the data
    kept_features_ : list
        Names of features that are not dropped
    n_features_out_ : Integral
        Number of features left after transforming
    """

    def __init__(
        self,
        peer_map: Union[dict, None] = None,
        default_peers: str = "all",
        fallback: str = "mean",
        missing_values=np.nan,
        nan_threshold=1.0,
    ):
        super().__init__(
            missing_values=missing_values,
            nan_threshold=nan_threshold,
        )

        if default_peers not in {"all", "none"}:
            raise ValueError("default_peers must be one of {'all', 'none'}")

        if fallback not in {"mean", "none", "zero"}:
            raise ValueError("fallback must be one of {'mean', 'none', 'zero'}")

        self.peer_map = peer_map
        self.default_peers = default_peers
        self.fallback = fallback

    def _fit_fill_values(self, X: pd.DataFrame, y=None) -> "CrossSectionalImputer":
        # Learn per-feature global means for optional fallback
        self.global_means_ = X.mean(axis=0, skipna=True)
        return self

    def _resolve_peers(self, target_cid: str, all_cids: Set[str]) -> List[str]:
        # If user provided a peer_map and cid exists there, use it.
        if isinstance(self.peer_map, dict) and target_cid in self.peer_map:
            peers = list(self.peer_map[target_cid] or [])
        else:
            if self.default_peers == "none":
                peers = []
            else:
                peers = [c for c in all_cids if c != target_cid]

        # Keep only peers that exist in the data, and drop the target if accidentally included
        peers = [c for c in peers if c in all_cids and c != target_cid]
        return peers

    def _transform_with_fill_values(self, X: pd.DataFrame) -> pd.DataFrame:
        X_filled = X.copy()

        # universe of cids present in this transform call
        all_cids = set(X.index.get_level_values(CIDS_INDEX_NAME).unique())

        # impute cid-by-cid using its peer set
        for target_cid in all_cids:
            peers = self._resolve_peers(target_cid, all_cids)
            if not peers:
                continue  # nothing to use for this cid

            # rows for this target cid
            target_mask = X_filled.index.get_level_values(CIDS_INDEX_NAME) == target_cid
            if not target_mask.any():
                continue

            target_rows = X_filled.loc[target_mask]
            target_dates = target_rows.index.get_level_values(DATE_INDEX_NAME)

            # pull peer rows only, compute mean by date (per feature)
            peer_rows = X.loc[X.index.get_level_values(CIDS_INDEX_NAME).isin(peers)]
            if peer_rows.empty:
                continue

            peer_date_means = peer_rows.groupby(level=DATE_INDEX_NAME).mean()

            # Align each target row with its date's peer mean
            aligned_means = peer_date_means.reindex(target_dates).set_index(
                target_rows.index
            )

            # fill ONLY missing values for the target cid using aligned peer means
            filled_target = target_rows.where(~target_rows.isna(), aligned_means)

            # write back
            X_filled.loc[target_mask] = filled_target

        # handle values still missing
        if self.fallback == "mean":
            X_filled = X_filled.fillna(self.global_means_)
        elif self.fallback == "zero":
            X_filled = X_filled.fillna(0)

        return X_filled


class EstimatorImputer(BaseImputer):
    """
    Impute missing values using a per-feature sklearn-compatible estimator
    trained on the remaining features at fit time.

    For each feature with missing values, a clone of the provided estimator is
    trained using all other features as predictors, on rows where the target
    feature is observed. At transform time the learned model fills in missing
    values in that feature.

    Parameters
    ----------
    estimator : BaseEstimator or None, default=None
        Any sklearn-compatible estimator (e.g. RandomForestRegressor,
        LinearRegression, Pipeline). If None, defaults to
        RandomForestRegressor().
    fallback : str, default="none"
        Strategy for handling values still missing after model-based imputation.
        - "mean": fill with column means
        - "zero": fill with zeros
        - "none": leave remaining NaNs in place
    missing_values : scalar, default=np.nan
        Value to treat as missing (converted to np.nan internally).
    nan_threshold : float, default=1.0
        If the proportion of NaNs in a column exceeds this threshold, the
        column is dropped entirely.
    complete_rows_only : bool, default=True
        If True, each per-feature model is trained only on rows where all
        predictor columns are also non-NaN. This allows any sklearn estimator
        to be used, not just those that handle NaN natively.
    predictor_fill_value : str, float, int, or None, default="mean"
        How to handle NaN predictor values at transform time. "mean" fills
        with per-column means from fit time, a numeric scalar fills with that
        constant, "skip" skips prediction for rows with NaN predictors
        (leaving them for the fallback to handle), and None applies no fill.

    Attributes
    ----------
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.
    missing_fraction_by_col_ : pd.Series
        Fraction of missing values for each column.
    missing_fraction_by_cid_and_col_ : pd.DataFrame
        Fraction of missing values for each column split by cid.
    dropped_features_ : list
        Names of features dropped due to exceeding nan_threshold.
    kept_features_ : list
        Names of features retained after thresholding.
    n_features_out_ : int
        Number of features remaining after transform.
    models_ : dict[str, Predictor]
        Mapping from feature name -> fitted estimator.
        Only populated for features that had at least one missing value during
        fit and had enough observed rows to train a model.
    predictor_means_ : pd.Series
        Column means of the kept features (computed on training data, used to
        fill missing predictor values before prediction).
    """

    _VALID_FILL_VALUES = {"mean", "skip"}
    _VALID_FALLBACKS = {"mean", "zero", "none"}

    def __init__(
        self,
        estimator: Union[BaseEstimator, None] = None,
        fallback: str = "mean",
        missing_values=np.nan,
        nan_threshold: float = 1.0,
        complete_rows_only: bool = True,
        predictor_fill_value: Union[str, float, int, None] = "mean",
    ):
        super().__init__(
            missing_values=missing_values,
            nan_threshold=nan_threshold,
        )
        if (
            isinstance(predictor_fill_value, str)
            and predictor_fill_value not in self._VALID_FILL_VALUES
        ):
            raise ValueError(
                f"predictor_fill_value must be None, 'mean', or a numeric scalar, "
                f"got '{predictor_fill_value}'"
            )

        if fallback not in self._VALID_FALLBACKS:
            raise ValueError(
                f"fallback must be one of {self._VALID_FALLBACKS}, got '{fallback}'"
            )

        self.estimator = estimator
        self.fallback = fallback
        self.complete_rows_only = complete_rows_only
        self.predictor_fill_value = predictor_fill_value

    # ------------------------------------------------------------------
    # BaseImputer hooks
    # ------------------------------------------------------------------
    def _fit_fill_values(self, X: pd.DataFrame, y=None) -> "EstimatorImputer":
        self.predictor_means_ = X.mean(axis=0, skipna=True)
        self.models_: Dict[str, Any] = {}

        base_estimator = (
            self.estimator if self.estimator is not None else RandomForestRegressor()
        )
        features = list(X.columns)

        for target_col in features:
            target = X[target_col]
            observed_mask = target.notna()

            if observed_mask.all():
                continue

            predictor_cols = [c for c in features if c != target_col]
            if not predictor_cols:
                continue

            X_train = X.loc[observed_mask, predictor_cols]
            y_train = target.loc[observed_mask]

            if self.complete_rows_only:
                complete_mask = X_train.notna().all(axis=1)
                X_train = X_train.loc[complete_mask]
                y_train = y_train.loc[complete_mask]

            if len(y_train) < 2:
                continue

            model = clone(base_estimator)
            try:
                model.fit(X_train, y_train)
            except Exception as exc:
                raise ValueError(
                    f"Estimator failed to fit for target feature '{target_col}': {exc}"
                ) from exc
            self.models_[target_col] = model

        return self

    def _transform_with_fill_values(self, X: pd.DataFrame) -> pd.DataFrame:
        X_filled = X.copy().astype(np.float64)
        features = list(X.columns)

        for target_col, model in self.models_.items():
            missing_mask = X_filled[target_col].isna()
            if not missing_mask.any():
                continue

            predictor_cols = [c for c in features if c != target_col]
            X_pred_raw = X_filled.loc[missing_mask, predictor_cols]

            if self.predictor_fill_value == "skip":
                predict_mask = X_pred_raw.notna().all(axis=1)
                if not predict_mask.any():
                    continue
                X_pred = X_pred_raw.loc[predict_mask]
                rows_to_fill = missing_mask & predict_mask.reindex(
                    missing_mask.index, fill_value=False
                )
            elif self.predictor_fill_value == "mean":
                X_pred = X_pred_raw.fillna(self.predictor_means_)
                rows_to_fill = missing_mask
            elif self.predictor_fill_value is None:
                X_pred = X_pred_raw
                rows_to_fill = missing_mask
            else:
                X_pred = X_pred_raw.fillna(self.predictor_fill_value)
                rows_to_fill = missing_mask

            try:
                X_filled.loc[rows_to_fill, target_col] = model.predict(X_pred)
            except Exception as exc:
                raise ValueError(
                    f"Estimator failed to predict for target feature '{target_col}': {exc}"
                ) from exc

        if self.fallback == "mean":
            X_filled = X_filled.fillna(self.predictor_means_)
        elif self.fallback == "zero":
            X_filled = X_filled.fillna(0)

        return X_filled


class GaussianConditionalImputer(BaseImputer):
    """
    Impute missing values using the closed-form Gaussian conditional mean.

    For each row with missing values, the imputer partitions the feature
    vector into observed (o) and missing (m) components and computes:

        mu_{m|o} = mu_m + Sigma_{mo} @ Sigma_{oo}^{-1} @ (x_o - mu_o)

    A single global Gaussian (mean + Ledoit-Wolf covariance) is fitted on
    all complete rows across all cross-section identifiers.

    Parameters
    ----------
    fallback : {"mean", "zero", "none"}, default="mean"
        Strategy for any values still missing after conditional imputation:
        - "mean": fill with column means
        - "zero": fill with zeros
        - "none": leave remaining NaNs in place
    missing_values : scalar, default=np.nan
        Value to treat as missing (converted to np.nan internally).
    nan_threshold : float, default=1.0
        If the proportion of NaNs in a column exceeds this threshold, the
        column is dropped entirely.
    """

    _VALID_FALLBACKS = {"mean", "zero", "none"}

    def __init__(
        self,
        fallback: str = "mean",
        missing_values=np.nan,
        nan_threshold: float = 1.0,
    ):
        if fallback not in self._VALID_FALLBACKS:
            raise ValueError(
                f"fallback must be one of {self._VALID_FALLBACKS}, got '{fallback}'"
            )
        super().__init__(
            missing_values=missing_values,
            nan_threshold=nan_threshold,
        )
        self.fallback = fallback

    def _fit_fill_values(self, X: pd.DataFrame, y=None) -> "GaussianConditionalImputer":
        self._fit_global_model(X)
        return self

    def _fit_global_model(self, X: pd.DataFrame) -> None:
        complete_rows = X.dropna()
        if len(complete_rows) >= 2:
            lw = LedoitWolf().fit(complete_rows.values)
            self.global_mean_ = lw.location_
            self.global_covariance_ = lw.covariance_
        else:
            self.global_mean_ = X.mean(axis=0, skipna=True).values.astype(np.float64)
            variances = X.var(axis=0, skipna=True).fillna(1.0).values
            self.global_covariance_ = np.diag(variances)

    def _transform_with_fill_values(self, X: pd.DataFrame) -> pd.DataFrame:
        X_filled = X.copy().astype(np.float64)

        for idx_label, row in X_filled.iterrows():
            missing_mask = row.isna().values
            if not missing_mask.any():
                continue

            if missing_mask.all():
                X_filled.loc[idx_label, :] = np.nan
                continue

            imputed = self._conditional_mean(
                x_observed=row.values,
                mean=self.global_mean_,
                cov=self.global_covariance_,
                missing_mask=missing_mask,
            )
            X_filled.loc[idx_label, :] = imputed

        if self.fallback == "mean":
            fallback_means = pd.Series(
                self.global_mean_, index=X.columns, dtype=np.float64
            )
            X_filled = X_filled.fillna(fallback_means)
        elif self.fallback == "zero":
            X_filled = X_filled.fillna(0)

        return X_filled

    @staticmethod
    def _conditional_mean(
        x_observed: np.ndarray,
        mean: np.ndarray,
        cov: np.ndarray,
        missing_mask: np.ndarray,
    ) -> np.ndarray:
        obs_idx = np.where(~missing_mask)[0]
        mis_idx = np.where(missing_mask)[0]

        mu_o = mean[obs_idx]
        mu_m = mean[mis_idx]
        sigma_oo = cov[np.ix_(obs_idx, obs_idx)]
        sigma_mo = cov[np.ix_(mis_idx, obs_idx)]

        x_o = x_observed[obs_idx]

        try:
            w = np.linalg.solve(sigma_oo, x_o - mu_o)
            imputed_missing = mu_m + sigma_mo @ w
        except np.linalg.LinAlgError:
            imputed_missing = np.full_like(mu_m, np.nan)

        result = x_observed.copy()
        result[missing_mask] = imputed_missing
        return result
