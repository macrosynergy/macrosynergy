from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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
    peer_map : dict[str, list[str]] | None
        Mapping from target cid -> list of peer cids to use for imputation.
        Example:
            {"CAD": ["USD", "GBP", "EUR"], "USD": ["CAD", "GBP", "EUR"]}

        If None, peers default to "all other cids" (unless default_peers="none").
    default_peers : {"all", "none"}
        Behaviour for cids not present in peer_map:
          - "all": use all other cids as peers
          - "none": do not impute for that cid (unless fallback kicks in)
    fallback : {"none", "global_mean"}
        If "global_mean", any values still missing after peer-based imputation
        are filled with the global mean per feature computed at fit time.
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
        peer_map: dict | None = None,
        default_peers: str = "all",
        fallback: str = "global_mean",
        missing_values=np.nan,
        nan_threshold=1.0,
    ):
        super().__init__(
            missing_values=missing_values,
            nan_threshold=nan_threshold,
        )

        if default_peers not in {"all", "none"}:
            raise ValueError("default_peers must be one of {'all', 'none'}")

        if fallback not in {"global_mean", "none"}:
            raise ValueError("fallback must be one of {'global_mean', 'none'}")

        self.peer_map = peer_map
        self.default_peers = default_peers
        self.fallback = fallback

    def _fit_fill_values(self, X: pd.DataFrame, y=None) -> "CrossSectionalImputer":
        # Learn per-feature global means for optional fallback
        self.global_means_ = X.mean(axis=0, skipna=True)
        return self

    def _resolve_peers(self, target_cid: str, all_cids: set[str]) -> list[str]:
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

        # fallback: fill any remaining NaNs with the global mean per feature
        if self.fallback == "global_mean":
            X_filled = X_filled.fillna(self.global_means_)

        return X_filled
