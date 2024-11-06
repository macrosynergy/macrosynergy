import numpy as np
import pandas as pd
import warnings
import datetime
import numbers

from sklearn.base import BaseEstimator, TransformerMixin


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