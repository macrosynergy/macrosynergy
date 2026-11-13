from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import warnings
import datetime
import numbers

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ZnScoreAverager(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        neutral="zero",
        use_signs=False,
    ):
        """
        Point-in-time factor normalization before feature averaging for a conceptual
        parity signal.

        :deprecated: This class is deprecated and will be replaced in a future release
            by a transformer that applies PiT z-scoring to each feature, without
            any averaging.

        Parameters
        ----------
        neutral : str, default='zero'
            Specified neutral value for a normalisation. This can take either 'zero' or
            'mean'. If the neutral level is zero, each feature is standardised by dividing
            by the mean absolute deviation. If the neutral level is mean, each feature is
            normalised by subtracting the mean and dividing by the standard deviation. Any
            statistics are computed according to a point-in-time principle.
        use_signs : bool, default=False
            Boolean to specify whether or not to return the signs of the benchmark signal
            instead of the signal itself.

        Notes
        -----
        When clear priors on underlying market drivers are available, a useful signal
        can be constructed by averaging PiT :math:`z_{n}`-scores of each feature/factor.
        This is often a competitive signal.

        A :math:`z_{n}`-score involves standardising a feature PiT by subtracting a
        `neutral` value and dividing by a measure of historic dispersion.
        In this class, the neutral value can be set to zero or the mean. An option for the
        median will be added in the future. The dispersion measure used is the mean
        absolute deviation for the zero neutral value and the standard deviation for the
        mean neutral value. The dispersion measure is calculated using all training
        information and test information until (and including) that test time, reflecting
        the information available to a portfolio manager at that time.

        We include that test time since features are assumed to lag behind returns, with
        the timestamp in the index representing the date of the returns.
        """

        warnings.warn(
            "ZnScoreAverager is deprecated and will be replaced in a future "
            "release by a transformer that applies PiT z-scoring to each feature, "
            "without any averaging, as per the current implementation.",
            DeprecationWarning,
        )

        if not isinstance(neutral, str):
            raise TypeError("'neutral' must be a string.")

        if neutral not in ["zero", "mean"]:
            raise ValueError("'neutral' must be either 'zero' or 'mean'.")

        if not isinstance(use_signs, bool):
            raise TypeError("'use_signs' must be a boolean.")

        self.neutral = neutral
        self.use_signs = use_signs
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        """
        Extract relevant standardisation/normalisation statistics.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.
        y : Any, default=None
            Placeholder for scikit-learn compatibility.
        """
        # Checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the ZnScoreAverager must be a pandas "
                "dataframe. If used as part of an sklearn pipeline, ensure that previous "
                "steps return a pandas dataframe."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not X.index.get_level_values(0).dtype == "object":
            raise TypeError("The outer index of X must be strings.")
        if not X.index.get_level_values(1).dtype == "datetime64[ns]":
            raise TypeError("The inner index of X must be datetime.date.")
        if not X.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            raise ValueError(
                "All columns in the input feature matrix for a panel selector ",
                "must be numeric.",
            )
        if X.isnull().values.any():
            raise ValueError(
                "The input feature matrix for a panel selector must not contain any "
                "missing values."
            )

        # fit

        self.training_n: int = len(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns

        if self.neutral == "mean":
            # calculate the mean and sum of squares of each feature
            means: pd.Series = X.mean(axis=0)
            sum_squares: pd.Series = np.sum(np.square(X), axis=0)
            self.training_means: pd.Series = means
            self.training_sum_squares: pd.Series = sum_squares
        else:
            # calculate the mean absolute deviation of each feature
            # TODO: maybe use the median?
            mads: pd.Series = np.mean(np.abs(X), axis=0)
            self.training_mads: pd.Series = mads

        return self

    def transform(self, X):
        """
        Create an OOS conceptual parity signal by averaging PiT z-scores of features.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.

        Returns
        -------
        pd.DataFrame
            Output signal.
        """
        # checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the ZnScoreAverager must be a pandas "
                "dataframe. If used as part of an sklearn pipeline, ensure that previous "
                "steps return a pandas dataframe."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")
        if not X.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            raise ValueError(
                "All columns in the input feature matrix for `ZnScoreAverager` ",
                "must be numeric.",
            )
        if X.isnull().values.any():
            raise ValueError(
                "The input feature matrix for `ZnScoreAverager` must not contain any "
                "missing values."
            )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The input feature matrix must have the same number of columns as the "
                "training feature matrix."
            )
        if not X.columns.equals(self.feature_names_in_):
            raise ValueError(
                "The input feature matrix must have the same columns as the training "
                "feature matrix."
            )

        X = X.sort_index(level="real_date")

        # transform
        signal_df = pd.DataFrame(index=X.index, columns=["signal"], dtype="float")

        if self.neutral == "zero":
            training_mads: pd.Series = self.training_mads
            training_n: int = self.training_n
            X_abs = X.abs()

            # We obtain a vector with the number of elements in each expanding window.
            # The count does not necessarily increase uniformly, since some cross-sections
            # may be missing for some dates.
            expanding_count = self._get_expanding_count(X)

            # We divide the expanding sum by the number of elements in each expanding window.
            X_test_expanding_mads = (
                X_abs.groupby(level="real_date").sum().expanding().sum()
                / expanding_count
            )
            n_total = training_n + expanding_count

            X_expanding_mads: pd.DataFrame = (
                (training_n) * training_mads + (expanding_count) * X_test_expanding_mads
            ) / n_total

            standardised_X: pd.DataFrame = (X / X_expanding_mads).fillna(0)

        elif self.neutral == "mean":
            training_means: pd.Series = self.training_means
            training_sum_squares: pd.Series = self.training_sum_squares
            training_n: int = self.training_n

            expanding_count = self._get_expanding_count(X)
            n_total = training_n + expanding_count

            X_square = X**2
            X_test_expanding_sum_squares = (
                X_square.groupby(level="real_date").sum().expanding().sum()
            )
            X_test_expanding_mean = (
                X.groupby(level="real_date").sum().expanding().sum() / expanding_count
            )

            X_expanding_means = (
                (training_n) * training_means
                + (expanding_count) * X_test_expanding_mean
            ) / n_total

            X_expanding_sum_squares = (
                training_sum_squares + X_test_expanding_sum_squares
            )
            comp1 = (X_expanding_sum_squares) / (n_total - 1)
            comp2 = 2 * np.square(X_expanding_means) * (n_total) / (n_total - 1)
            comp3 = (n_total) * np.square(X_expanding_means) / (n_total - 1)
            X_expanding_std: pd.Series = np.sqrt(comp1 - comp2 + comp3)
            standardised_X: pd.DataFrame = (
                (X - X_expanding_means) / X_expanding_std
            ).fillna(0)

        else:
            raise ValueError("neutral must be either 'zero' or 'mean'.")

        benchmark_signal: pd.DataFrame = pd.DataFrame(
            np.mean(standardised_X, axis=1), columns=["signal"], dtype="float"
        )
        signal_df.loc[benchmark_signal.index] = benchmark_signal

        if self.use_signs:
            return np.sign(signal_df).astype(int)

        return signal_df

    def _get_expanding_count(self, X):
        """
        Get the number of non-NaN values in each expanding window.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Expanding count.
        """

        return X.groupby(level="real_date").count().expanding().sum().to_numpy()


class PanelPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, kaiser_criterion=False, adjust_signs=False):
        """
        PCA transformer for panel data.

        Parameters
        ----------
        n_components : int, float, or None, default=None
            Number of components to keep. If None, all components are kept.
            If an integer, keep the first `n_components` components.
            If a float, keep the top components that explain the percentage of variance
            specified by the float.
        kaiser_criterion : bool, default=False
            If True, only components with eigenvalues greater or equal to one are kept.
        adjust_signs : bool, default=False
            If True, adjust signs of eigenvectors so that projected training features
            are positively correlated with a provided target variable. This is useful for
            consistency when used in a sequential learning pipeline through time.

        Notes
        -----
        (P)rinicpal (C)omponent (A)nalysis is an unsupervised method that projects an
        input feature matrix onto a new basis of orthogonal vectors, called principal
        components. This basis is chosen to be the directions of maximal variability within
        the estimated covariance matrix of the input features. The principal components are
        ordered by the amount of variance they explain. Typically, the first few components
        can be attributed to "signal" whilst the remaining components are "noise".

        Most modern implementations use the singular value decomposition (SVD) to perform
        PCA. We estimate a covariance matrix because it follows that an alternative
        covariance estimator can be used for PCA, which may affect results. However, this
        behaviour will be introduced in a future release.
        """
        if n_components is not None:
            if not isinstance(n_components, numbers.Number) or isinstance(
                n_components, bool
            ):
                raise TypeError("n_components must be a number or None.")
            if n_components <= 0:
                raise ValueError("n_components must be greater than 0.")
            if isinstance(n_components, float) and n_components > 1:
                raise ValueError(
                    "If n_components is a float, it must be between 0 and 1."
                )

        if not isinstance(kaiser_criterion, bool):
            raise TypeError("kaiser_criterion must be a boolean.")

        if not isinstance(adjust_signs, bool):
            raise TypeError("adjust_signs must be a boolean.")

        self.n_components = n_components
        self.kaiser_criterion = kaiser_criterion
        self.adjust_signs = adjust_signs
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        """
        Fit method to determine an eigenbasis for the PCA.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.
        y : pd.DataFrame, pd.Series or np.ndarray, default=None
            Target variable.

        Notes
        -----
        The target variable `y` is only ever used to adjust the signs
        of the eigenvectors to ensure consistency of eigenvector signs
        when retrained over time. This does not affect the PCA itself.
        """
        # Checks
        self._check_fit_params(X, y)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns

        # Estimate covariance matrix and perform eigendecomposition
        covariance_matrix = X.cov()
        evals, evecs = np.linalg.eigh(covariance_matrix.values)

        # Sort eigenvalues and eigenvectors by descending eigenvalue order
        sorted_idx = np.argsort(evals)[::-1]

        # Store eigenvalues and eigenvectors
        self.adjusted_evals = evals[sorted_idx]
        self.adjusted_evecs = evecs[:, sorted_idx]

        if self.kaiser_criterion:
            # Get eigenvalues greater or equal to one
            mask = self.adjusted_evals >= 1
            self.adjusted_evals = self.adjusted_evals[mask]
            self.adjusted_evecs = self.adjusted_evecs[:, mask]
        elif isinstance(self.n_components, int):
            # Keep first n_components components
            self.adjusted_evals = self.adjusted_evals[: self.n_components]
            self.adjusted_evecs = self.adjusted_evecs[:, : self.n_components]
        elif isinstance(self.n_components, float):
            # Keep components that explain a certain percentage of difference.
            variance_explained = self.adjusted_evals / np.sum(self.adjusted_evals)
            cumulative_variance_explained = np.cumsum(variance_explained)
            mask = cumulative_variance_explained <= self.n_components
            self.adjusted_evals = self.adjusted_evals[mask]
            self.adjusted_evecs = self.adjusted_evecs[:, mask]

        # Adjust signs of eigenvectors so that projected data is positively correlated with y
        if self.adjust_signs:
            # Convert y to numpy array
            if isinstance(y, pd.Series):
                y = y.values
            elif isinstance(y, pd.DataFrame):
                y = y.values.reshape(-1)

            # Calculate correlation between projected data and y
            # If correlation is negative, flip the sign of the eigenvector
            for i in range(self.adjusted_evecs.shape[1]):
                if np.corrcoef(X.values @ self.adjusted_evecs[:, i], y)[0, 1] < 0:
                    self.adjusted_evecs[:, i] *= -1

        return self

    def transform(self, X):
        """
        Project input features onto the principal components.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.

        Returns
        -------
        pd.DataFrame
            Projected features.
        """
        # Checks
        self._check_transform_params(X)

        # Project data onto the principal components
        return pd.DataFrame(
            index=X.index,
            columns=[f"PCA {i+1}" for i in range(self.adjusted_evecs.shape[1])],
            data=X.values @ self.adjusted_evecs,
        )

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names produced by the transformation.

        Parameters
        ----------
        input_features : None
            This parameter has no effect and is included for compatibility
            with the scikit-learn API.
        """
        return np.array([f"PCA {i+1}" for i in range(self.adjusted_evecs.shape[1])])

    def _check_fit_params(self, X, y):
        """
        Checks the input data for the fit method.

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series or pandas.DataFrame
            The target vector.
        """
        # Checks only necessary on X
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the selector must be a pandas dataframe. ",
                "If used as part of an sklearn pipeline, ensure that previous steps ",
                "return a pandas dataframe.",
            )
        if not X.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            raise ValueError(
                "All columns in the input feature matrix for PanelPCA",
                " must be numeric.",
            )
        if X.isnull().values.any():
            raise ValueError(
                "The input feature matrix for PanelPCA must not contain any "
                "missing values."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("The input feature matrix for X must be multi-indexed.")
        if not X.index.get_level_values(0).dtype == "object":
            raise TypeError("The outer index of X must be strings.")
        if not X.index.get_level_values(1).dtype == "datetime64[ns]":
            raise TypeError("The inner index of X must be datetime.date.")

        # Checks on y only necessary if adjust_signs is True
        if self.adjust_signs:
            if y is None:
                raise ValueError(
                    "If `adjust_signs` is True, a target variable must be provided. "
                    "PCA is an unsupervised method, so providing a target variable "
                    "does not affect the PCA. Eigenvectors, however, are unique up to "
                    "its sign, meaning that a target variable can be used to ensure "
                    "consistency in the signs of the eigenvectors through time and over repeated runs."
                )
            else:
                if not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray)):
                    raise TypeError(
                        "y must be a pandas Series, DataFrame, or numpy array."
                    )
                if not y.shape[0] == X.shape[0]:
                    raise ValueError("y must have the same number of rows as X.")
                if not isinstance(y, np.ndarray):
                    if not np.issubdtype(y.values.dtype, np.number):
                        raise ValueError("The target vector must be numeric.")
                    if y.isnull().values.any():
                        raise ValueError(
                            "The target vector must not contain any missing values."
                        )
                else:
                    if not np.issubdtype(y.dtype, np.number):
                        raise ValueError("The target vector must be numeric.")
                    if np.isnan(y).any():
                        raise ValueError(
                            "The target vector must not contain any missing values."
                        )

    def _check_transform_params(self, X):
        """
        Input checks for the transform method.

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the PanelPCA must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not X.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            raise ValueError(
                "All columns in the input feature matrix for PanelPCA",
                " must be numeric.",
            )
        if X.isnull().values.any():
            raise ValueError(
                "The input feature matrix for PanelPCA must not contain any "
                "missing values."
            )
        if not X.index.get_level_values(0).dtype == "object":
            raise TypeError("The outer index of X must be strings.")
        if not X.index.get_level_values(1).dtype == "datetime64[ns]":
            raise TypeError("The inner index of X must be datetime.date.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The input feature matrix must have the same number of columns as the "
                "training feature matrix."
            )
        if not X.columns.equals(self.feature_names_in_):
            raise ValueError(
                "The input feature matrix must have the same columns as the training "
                "feature matrix."
            )

class ReturnClusterer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        col_name: str,
        n_clusters: Optional[int] = None,
        cluster_map: Optional[Dict[str, List[str]]] = None,
        linkage: str = "average",
        min_overlap: int = 36,
        relative_returns: bool = True,
    ):
        """
        Transformer that appends a categorical column holding the group of each
        cross-section. Groups are either given or learnt by clustering the
        cross-sections on the correlations of their returns.

        Parameters
        ----------
        col_name : str
            Name of the categorical column appended to the feature matrix.
        n_clusters : int, default=None
            Number of groups to form by clustering. Required when `cluster_map` is
            None and ignored otherwise.
        cluster_map : Dict[str, List[str]], default=None
            Mapping of group names to list-likes of cross-sections. If provided, no
            clustering is performed and every training cross-section must belong to
            exactly one group.
        linkage : str, default="average"
            Linkage method used to merge clusters. One of "average", "complete" or
            "single".
        min_overlap : int, default=36
            Minimum number of dates two cross-sections must share for the correlation
            of their returns to be estimated. Pairs with less overlap are treated as
            uncorrelated.
        relative_returns : bool, default=True
            Whether to cluster on returns relative to the cross-sectional mean of
            each date, rather than raw returns.

        Attributes
        ----------
        clusters_ : Dict[str, List[str]]
            Mapping of each group to the list of its cross-sections. Groups are
            labelled by the keys of `cluster_map`, or by integers starting at zero
            when formed by clustering.
        cid_to_group_ : Dict[str, Union[str, int]]
            Mapping of each cross-section to its group.
        categories_ : List
            Sorted groups, used as the categories of the appended column.
        n_features_in_ : int
            Number of columns of the feature matrix seen in fit.
        feature_names_in_ : np.ndarray
            Columns of the feature matrix seen in fit.

        Notes
        -----
        When clustering, the training targets are pivoted to one time series per
        cross-section and the Pearson correlation of each pair is computed over
        their common dates. Pairs sharing fewer than `min_overlap` dates, are
        treated as uncorrelated. The correlations are turned into distances
        :math:`d_{ij} = \sqrt{0.5 (1 - \rho_{ij})}` and the cross-sections are
        merged by agglomerative hierarchical clustering with the chosen linkage
        until at most `n_clusters` groups remain.
        """
        self.col_name = col_name
        self.n_clusters = n_clusters
        self.cluster_map = cluster_map
        self.linkage = linkage
        self.min_overlap = min_overlap
        self.relative_returns = relative_returns

    def fit(self, X, y=None):
        """
        Assign each training cross-section to a group, from `cluster_map` or by
        clustering the targets.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix, multi-indexed by `cid` and `real_date`.
        y : pd.Series, pd.DataFrame or np.ndarray, default=None
            Target vector with one value per row of X. Required when `cluster_map`
            is None and ignored otherwise.

        Returns
        -------
        self
            Fitted transformer.
        """
        self._check_init_params()
        self._check_X(X)

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns)

        cids = sorted(X.index.get_level_values(0).unique())

        if self.cluster_map is not None:
            _validate_groups(self.cluster_map, cids)
            self.clusters_ = {
                group: list(group_cids)
                for group, group_cids in self.cluster_map.items()
            }
            self.cid_to_group_ = {
                cid: group
                for group, group_cids in self.clusters_.items()
                for cid in group_cids
            }
        else:
            self.clusters_, self.cid_to_group_ = self._cluster_groups(X, y, cids)

        self.categories_ = sorted(set(self.cid_to_group_.values()))

        return self

    def transform(self, X):
        """
        Append the categorical group column to the feature matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix, multi-indexed by `cid` and `real_date`.

        Returns
        -------
        pd.DataFrame
            Copy of X with the group of each cross-section in column `col_name`.
        """
        check_is_fitted(self, "cid_to_group_")
        self._check_X(X)

        X = X.copy()
        cids = X.index.get_level_values(0)
        X[self.col_name] = pd.Categorical(
            values=cids.map(self.cid_to_group_),
            categories=self.categories_,
        )

        return X

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names produced by the transformation.

        Parameters
        ----------
        input_features : None
            This parameter has no effect and is included for compatibility
            with the scikit-learn API.

        Returns
        -------
        np.ndarray
            Columns of the training feature matrix followed by `col_name`.
        """
        check_is_fitted(self, "cid_to_group_")

        return np.array([*self.feature_names_in_, self.col_name], dtype=object)

    def _cluster_groups(self, X, y, cids):
        """
        Form groups by hierarchical agglomerative clustering of the training targets.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix, multi-indexed by `cid` and `real_date`.
        y : pd.Series, pd.DataFrame or np.ndarray
            Target vector with one value per row of X.
        cids : list
            Sorted training cross-sections.

        Returns
        -------
        Tuple[dict, dict]
            Mapping of each group to its cross-sections and mapping of each
            cross-section to its group.
        """
        if self.n_clusters > len(cids):
            warnings.warn(
                f"`n_clusters` ({self.n_clusters}) is larger than the number of "
                f"training cross-sections ({len(cids)}). Falling back to a single "
                f"cluster.",
                UserWarning,
            )
            n_clusters = 1
        else:
            n_clusters = self.n_clusters

        y_vals = np.asarray(y, dtype=float).reshape(len(y))
        rets = pd.Series(y_vals, index=X.index)

        return _cluster_returns(
            rets=rets,
            cids=cids,
            n_clusters=n_clusters,
            min_overlap=self.min_overlap,
            linkage=self.linkage,
            relative_returns=self.relative_returns,
        )

    def _check_init_params(self):

        for arg, val, ok_type in [
            ("col_name", self.col_name, str),
            ("n_clusters", self.n_clusters, (int, type(None))),
            ("cluster_map", self.cluster_map, (dict, type(None))),
            ("linkage", self.linkage, str),
            ("min_overlap", self.min_overlap, int),
            ("relative_returns", self.relative_returns, (bool, np.bool_)),
        ]:
            if not isinstance(val, ok_type):
                raise TypeError(f"{arg} must be of type {ok_type}")

        if self.cluster_map is not None:
            for group, group_cids in self.cluster_map.items():
                if not isinstance(group, str):
                    raise TypeError(
                        f"The cross-sections of group {group} in cluster_map must "
                        f"be strings."
                    )
                if not pd.api.types.is_list_like(group_cids):
                    raise TypeError(
                        f"The cross-sections of group {group} in cluster_map must "
                        f"be a list-like of strings."
                    )
        elif self.n_clusters is None:
            raise ValueError("n_clusters must be provided when cluster_map is None.")

        valid_linkages = ["average", "complete", "single"]
        if self.linkage not in valid_linkages:
            raise ValueError(f"linkage must be one of {valid_linkages}")

    def _check_X(self, X):
        """
        Checks shared by the fit and transform methods. The values of X are not
        used, so they may be non-numeric or missing.

        Parameters
        ----------
        X : pd.DataFrame
            The feature matrix.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the ReturnClusterer must be a pandas "
                "dataframe."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("The input feature matrix for X must be multi-indexed.")
        if self.col_name in X.columns:
            raise ValueError(
                f"The input feature matrix already has a column named "
                f"{self.col_name!r}. Choose a different col_name."
            )


# TODO: dont duplicate code
def _cluster_returns(
    rets: pd.Series,
    cids: List[str],
    n_clusters: int,
    min_overlap: int,
    linkage: str,
    relative_returns: bool = False,
) -> Tuple[Dict[int, List[str]], Dict[str, int]]:
    labels = [1]
    if len(cids) != 1:
        rets = rets.unstack(level=0).reindex(columns=cids)
        if relative_returns:
            rets = rets.sub(rets.mean(axis=1), axis=0)
        corr = rets.corr(min_periods=min_overlap).to_numpy()

        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1)

        dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
        np.fill_diagonal(dist, 0)

        linkage_matrix = hierarchy.linkage(
            y=squareform(dist, checks=False), method=linkage
        )
        labels = hierarchy.fcluster(
            Z=linkage_matrix,
            t=n_clusters,
            criterion="maxclust",
        )

    # Label groups
    relabel = {}
    for label in labels:
        relabel.setdefault(label, len(relabel))
    cid_to_group = {cid: relabel[label] for cid, label in zip(cids, labels)}

    clusters = {
        group: [cid for cid in cids if cid_to_group[cid] == group]
        for group in range(len(relabel))
    }

    return clusters, cid_to_group


# TODO: dont duplicate code
def _validate_groups(
    groups: Dict[str, List[str]], cids: List[str]
) -> Dict[str, str]:
    if not isinstance(groups, dict):
        raise TypeError(
            "cluster_map must be a dictionary mapping group names to lists of "
            "cross-sections"
        )

    # Each listed cid may belong to one group only
    listed_cid_to_group = {}
    for group, group_cids in groups.items():
        for cid in group_cids:
            if cid in listed_cid_to_group and listed_cid_to_group[cid] != group:
                raise ValueError(
                    f"Cross-section {cid} is assigned to more than one group: "
                    f"{listed_cid_to_group[cid]} and {group}."
                )
            listed_cid_to_group[cid] = group

    # Every training cid must belong to a group
    missing_cids = [cid for cid in cids if cid not in listed_cid_to_group]
    if len(missing_cids) > 0:
        raise ValueError(
            f"The following cross-sections in the training data are not assigned "
            f"to any group: {missing_cids}."
        )

    return {cid: listed_cid_to_group[cid] for cid in cids}


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    from macrosynergy.management import (
        categories_df,
        make_qdf,
    )

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    """Example: Unbalanced panel """

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2000-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2000-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2000-01-01", "2020-12-31", -0.1, 2, 0.8, 0.3]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {
        "GBP": (
            pd.Timestamp(year=2009, month=1, day=1),
            pd.Timestamp(year=2012, month=6, day=30),
        ),
        "CAD": (
            pd.Timestamp(year=2015, month=1, day=1),
            pd.Timestamp(year=2100, month=1, day=1),
        ),
    }

    train = categories_df(
        df=dfd, xcats=xcats, cids=cids, val="value", blacklist=black, freq="M", lag=1
    ).dropna()
    train = train[
        train.index.get_level_values(1) >= pd.Timestamp(year=2005, month=8, day=1)
    ]

    X_train = train.drop(columns=["XR"])
    y_train = train["XR"]

    # Neutral = zero
    zn = ZnScoreAverager(neutral="zero", use_signs=False)
    zn.fit(X_train, y_train)
    print(zn.transform(X_train))

    # Neutral = mean, use_signs = False
    zn = ZnScoreAverager(neutral="mean", use_signs=False)
    zn.fit(X_train, y_train)
    print(zn.transform(X_train))

    # Neutral = mean, use_signs = True
    zn = ZnScoreAverager(neutral="mean", use_signs=True)
    zn.fit(X_train, y_train)
    print(zn.transform(X_train))

    # Clustered groups
    rc = ReturnClusterer(col_name="GROUP", n_clusters=2)
    rc.fit(X_train, y_train)
    print(rc.clusters_)
    print(rc.transform(X_train))