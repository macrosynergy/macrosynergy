import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin

import numbers


class BaseWeightedRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        model,
        sign_weighted=False,
        time_weighted=False,
        half_life=12 * 21,
    ):
        """
        Weighted regression model to prioritize contribution from certain samples over
        others during training.

        Parameters
        ----------
        model : RegressorMixin
            The underlying model to be trained with weighted sample contributions.
        sign_weighted : bool, optional
            Flag to weight samples based on the frequency of the label's sign in a
            training set.
        time_weighted : bool, optional
            Flag to weight samples based on the recency of the sample.
        half_life : numbers.Number , optional
            Half-life of the exponential decay function used to calculate the time weights.
            This should be expressed in units of the native dataset frequency. Default is
            12 * 21, which corresponds to a half-life of 1-year for a daily dataset.

        Notes
        -----
        Sign-weighted regression models are useful when the dependent (return) variable
        has a directional bias. By assigning higher weights to the less frequent class,
        the model is encouraged to learn equally from both positive and negative return
        samples, irrespective of class imbalance.

        Time-weighted regression models are useful when the practitioner holds the prior
        that more recent samples are more informative than older samples. By assigning
        exponentially decaying weights to samples based on their recency, the model is
        encouraged to prioritize newer information.
        """
        # Checks
        self._check_init_params(model, sign_weighted, time_weighted, half_life)

        # Attributes
        self.model = model
        self.sign_weighted = sign_weighted
        self.time_weighted = time_weighted
        self.half_life = None if not time_weighted else half_life

    def fit(
        self,
        X,
        y,
    ):
        """
        Learn optimal weighted regression parameters.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Pandas dataframe or numpy array of input features.
        y : pd.Series or pd.DataFrame or np.ndarray
            Pandas series, dataframe or numpy array of targets associated with each sample
            in X.
        """
        # Checks
        self._check_fit_params(X, y)

        # Fit
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.sample_weights = self._calculate_sample_weights(y)
        self.model.fit(X, y, sample_weight=self.sample_weights)
        if hasattr(self.model, "coef_"):
            self.coef_ = self.model.coef_
        if hasattr(self.model, "intercept_"):
            self.intercept_ = self.model.intercept_

        return self

    def predict(self, X):
        """
        Predict dependent variable using the fitted weighted regression model.

        Parameters
        ----------
        X : pd.DataFrame or numpy array
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray
            Numpy array of predictions.
        """
        # Checks
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                "Input feature matrix for weighted regressors must be either a pandas "
                "dataframe or numpy array. If used as part of an sklearn pipeline, ensure "
                "that previous steps return a pandas dataframe or numpy array."
            )
        if self.time_weighted and not isinstance(X, pd.DataFrame):
            raise TypeError(
                "When time weighting is enabled, the input feature matrix for weighted "
                "regressors must be a pandas dataframe."
            )
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError(
                    "When the input feature matrix for weighted regressor forecasts is a "
                    "numpy array, it must have two dimensions. If used as part of an "
                    "sklearn pipeline, ensure that previous steps return a two-dimensional "
                    "data structure."
                )

        if X.shape[1] != self.p:
            raise ValueError(
                "The number of features in the input feature matrix must match the number "
                "seen in training."
            )

        # Predict
        return self.model.predict(X)

    def _calculate_sample_weights(
        self,
        y,
    ):
        """
        Determine sample weights based on the sign and time weighting flags.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame or np.ndarray
            Target vector associated with each sample in X.
        """
        if self.sign_weighted and self.time_weighted:
            sign_weights = self._calculate_sign_weights(y)
            time_weights = self._calculate_time_weights(y)
            return sign_weights * time_weights
        elif self.sign_weighted:
            return self._calculate_sign_weights(y)
        elif self.time_weighted:
            return self._calculate_time_weights(y)
        else:
            return np.ones(y.shape[0])

    def _calculate_sign_weights(
        self,
        targets,
    ):
        """
        Calculate balanced inverse frequency weights for positive and negative signs
        in the target vector.

        Parameters
        ----------
        targets : pd.Series or pd.DataFrame or np.ndarray
            Dependent variable.

        Returns
        -------
        sample_weights : np.ndarray
            Numpy array of sample weights.
        """
        pos_sum = np.sum(targets >= 0)
        neg_sum = np.sum(targets < 0)

        pos_weight = len(targets) / (2 * pos_sum) if pos_sum > 0 else 0
        neg_weight = len(targets) / (2 * neg_sum) if neg_sum > 0 else 0

        sample_weights = np.where(targets >= 0, pos_weight, neg_weight)
        return sample_weights

    def _calculate_time_weights(
        self,
        targets,
    ):
        """
        Calculate exponentially decaying weights based on the recency of the sample in the
        panel.

        Parameters
        ----------
        targets : pd.Series or pd.DataFrame
            Dependent variable.

        Returns
        -------
        sample_weights : np.ndarray
            Numpy array of sample weights.
        """
        dates = sorted(targets.index.get_level_values(1).unique(), reverse=True)
        num_dates = len(dates)
        weights = np.power(2, -np.arange(num_dates) / self.half_life)

        weight_map = dict(zip(dates, weights))
        sample_weights = targets.index.get_level_values(1).map(weight_map).to_numpy()

        return sample_weights

    def _check_init_params(
        self,
        model,
        sign_weighted,
        time_weighted,
        half_life,
    ):
        """
        Checks for constructor parameters.
        """
        if not isinstance(model, RegressorMixin):
            raise TypeError(
                "The model parameter must be an instance of a sklearn regressor."
            )
        if not isinstance(sign_weighted, bool):
            raise TypeError("The sign_weighted parameter must be a boolean.")
        if not isinstance(time_weighted, bool):
            raise TypeError("The time_weighted parameter must be a boolean.")
        if not isinstance(half_life, numbers.Number) or isinstance(half_life, bool):
            raise TypeError("The half_life parameter must be a number.")
        if half_life <= 0:
            raise ValueError("The half_life parameter must be a positive number.")

    def _check_fit_params(
        self,
        X,
        y,
    ):
        """
        Checks for fit method parameters.
        """
        # X
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                "Input feature matrix for weighted regressors must be either a pandas "
                "dataframe or numpy array."
            )
        elif self.time_weighted and not isinstance(X, pd.DataFrame):
            raise TypeError(
                "When time weighting is enabled, the input feature matrix for weighted "
                "regressors must be a pandas dataframe."
            )
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError(
                    "When the input feature matrix for weighted regressor forecasts is a "
                    "numpy array, it must have two dimensions."
                )
        # y
        if not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError(
                "Target vector for weighted regressors must be either a pandas series, "
                "dataframe or numpy array."
            )
        elif self.time_weighted and not isinstance(y, (pd.Series, pd.DataFrame)):
            raise TypeError(
                "When time weighting is enabled, the target vector for weighted regressors "
                "must be either a pandas series or pandas dataframe."
            )
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(
                    "The dependent variable dataframe must have only one column. If used "
                    "as part of an sklearn pipeline, ensure that previous steps return "
                    "a pandas series or dataframe."
                )
        if isinstance(y, np.ndarray):
            if y.ndim != 1:
                raise ValueError(
                    "When the target vector for weighted regressor forecasts is a numpy "
                    "array, it must have one dimension."
                )
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "The number of samples in the input feature matrix must match the number "
                "of samples in the target vector."
            )

        # Joint X and y checks
        if len(X) != len(y):
            raise ValueError(
                "The number of samples in the input feature matrix must match the number "
                "of samples in the target vector."
            )
        if self.time_weighted:
            if not y.index.equals(X.index):
                raise ValueError(
                    "When time weighting is enabled, the target vector and input feature "
                    "matrix must have the same index."
                )


class SignWeightedRegressor(BaseWeightedRegressor):
    def __init__(self, model):
        """
        Regressor with sign-weighted sample weights.

        Parameters
        ----------
        model : RegressorMixin
            The underlying regression model to be trained with weighted samples.

        Notes
        -----
        By weighting the contribution of different training samples based on the
        sign of the label, the model is encouraged to learn equally from both positive
        and negative return samples, irrespective of class imbalance. If there are more
        positive targets than negative targets in the training set, then the negative
        target samples are given a higher weight in the model training process. The
        opposite is true if there are more negative targets than positive targets.
        """

        super().__init__(model, sign_weighted=True, time_weighted=False)


class TimeWeightedRegressor(BaseWeightedRegressor):
    def __init__(
        self,
        model,
        half_life,
    ):
        """
        Regressor with time-weighted sample weights.

        Parameters
        ----------
        model : RegressorMixin
            The underlying regression model to be trained with weighted samples.
        half_life : numbers.Number
            Half-life of the exponential decay function used to calculate the time weights.

        Notes
        -----
        By weighting the contribution of different training samples based on the
        timestamp, the model is encouraged to prioritise newer information. The half-life
        denotes the number of time periods in units of the native data frequency for the
        weight attributed to the most recent sample to decay by half.
        """
        super().__init__(
            model, sign_weighted=False, time_weighted=True, half_life=half_life
        )
