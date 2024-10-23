"""
Collection of custom scikit-learn transformer classes.
"""

import datetime
import warnings
import numbers
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import ElasticNet, Lars, Lasso

from macrosynergy.compat import OneToOneFeatureMixin

from macrosynergy.learning.random_effects import RandomEffects


class LarsSelector(BaseEstimator, SelectorMixin):
    def __init__(self, fit_intercept = False, n_factors = 10):
        """
        Statistical feature selection using LARS.  

        :param <bool> fit_intercept: Whether to fit an intercept term in the LARS model.
        :param <int> n_factors: Number of factors to select. 
        """
        # Checks 
        if not isinstance(fit_intercept, bool):
            raise TypeError("'fit_intercept' must be a boolean.")
        if not isinstance(n_factors, int):
            raise TypeError("'n_factors' must be an integer.")
        if n_factors <= 0:
            raise ValueError("'n_factors' must be a positive integer.")
        
        # Attributes
        self.fit_intercept = fit_intercept
        self.n_factors = n_factors

    def fit(self, X, y):
        """
        Fit method for LARS to obtain the selected features.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.Series,pd.DataFrame]> y: Pandas series or dataframe of targets
            associated with each sample in X.
        """
        # Checks 
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the LARS selector must be a pandas dataframe. ",
                "If used as part of an sklearn pipeline, ensure that previous steps ",
                "return a pandas dataframe."
            )
        if not (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)):
            raise TypeError(
                "Target vector for the LARS selector must be a pandas series or dataframe. ",
                "If used as part of an sklearn pipeline, ensure that previous steps ",
                "return a pandas series or dataframe."
            )
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(
                    "The target dataframe must have only one column. If used as part of ",
                    "an sklearn pipeline, ensure that previous steps return a pandas ",
                    "series or dataframe."
                )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(y.index, pd.MultiIndex):
            raise ValueError("y must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")
        if not isinstance(y.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of y must be datetime.date.")
        if not X.index.equals(y.index):
            raise ValueError(
                "The indices of the input dataframe X and the output dataframe y don't "
                "match."
            )
        
        # Store the names of the features and dataframe dimensions
        self.feature_names_in_ = X.columns
        self.n = len(X)
        self.p = X.shape[1]

        # Standardise the features for fair comparison
        X = ((X - X.mean()) / X.std()).copy()

        # Fit the model
        lars = Lars(fit_intercept = self.fit_intercept, n_nonzero_coefs = self.n_factors)
        lars.fit(X.values, y.values.reshape(-1, 1))
        coefs = lars.coef_

        self.mask = [True if coef != 0 else False for coef in coefs]

        return self
    
    def transform(self, X):
        """
        Transform method to return only the selected features of the dataframe.

        :param <pd.DataFrame> X: Pandas dataframe of input features.

        :return <pd.DataFrame>: Pandas dataframe of input features selected based
            on LARS' feature selection capabilities.
        """
        # checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the LARS selector must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")
        if not X.shape[-1] == self.p:
            raise ValueError(
                "The number of columns of the dataframe to be transformed, X, doesn't "
                "match the number of columns of the training dataframe."
            )
        if sum(self.mask) == 0:
            # Then no features were selected
            # Then at the given time, no trading decisions can be made based on these features
            warnings.warn(
                "No features were selected. At the given time, no trading decisions can be made based on these features.",
                RuntimeWarning,
            )
            return X.iloc[:, :0]
        
        return X.loc[:, self.mask]
    
    def _get_support_mask(self):
        """
        Private method to return a boolean mask of the features selected for the Pandas
        dataframe.
        """
        return self.mask
    
    def get_feature_names_out(self):
        """
        Method to mask feature names according to selected features.
        """
        if self.feature_names_in_ is None:
            raise NotFittedError(
                "The LarsSelector selector has not been fitted. Please fit the selector ",
                "before calling get_feature_names_out()."
            )

        return self.feature_names_in_[self.get_support(indices=False)]

class ENetSelector(BaseEstimator, SelectorMixin):
    def __init__(
        self, alpha: Union[float, int] = 1.0, l1_ratio=0.5, positive: bool = True
    ):
        """
        Transformer class to use "Elastic Net" as a feature selection algorithm.
        Given two hyper-parameters, `alpha` and `l1_ratio`, the Elastic Net model is fit and
        the non-zero coefficients are used to extract features from an input dataframe.
        If `positive` is set to True, the underlying features as input to the Elastic Net
        are expected to be positively correlated with the target variable.

        :param <Union[float, int]> alpha: the regularisation imposed by the Elastic Net.
        :param <float> l1_ratio: the ratio of L1 to L2 regularisation. If 0, the penalty is
            an L2 penalty. If 1, it is an L1 penalty. If 0 < l1_ratio < 1, the penalty is
            a combination of L1 and L2. As per the `scikit-learn` documentation,
            setting the `l1_ratio` <= 0.01 is not reliable.
        :param <bool> positive: boolean to restrict estimated Elastic Net coefficients to
            be positive.
        """
        if not (isinstance(alpha, float) or isinstance(alpha, int)):
            raise TypeError(
                "The 'alpha' hyper-parameter must be either a float or int."
            )
        if alpha < 0:
            raise ValueError("The 'alpha' hyper-parameter must be non-negative.")

        if not (isinstance(l1_ratio, float) or isinstance(l1_ratio, np.floating)):
            raise TypeError("The 'l1_ratio' hyper-parameter must be a float.")
        if (l1_ratio < 0) or (l1_ratio > 1):
            raise ValueError(
                "The 'l1_ratio' hyper-parameter must be in the interval [0,1]."
            )
        if l1_ratio <= 0.01:
            warnings.warn(
                "Setting the 'l1_ratio' hyper-parameter to be <= 0.01 is not reliable.",
                UserWarning,
            )
        if not isinstance(positive, bool):
            raise TypeError("The 'positive' hyper-parameter must be a boolean.")

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.positive = positive
        self.feature_names_in_ = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]):
        """
        Fit method to fit an Elastic Net regression and obtain the selected features.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.Series,pd.DataFrame]> y: Pandas series or dataframe of targets
            associated with each sample in X.
        """
        # checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the Elastic Net selector must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)):
            raise TypeError(
                "Target vector for the Elastic Net selector must be a pandas series or dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas series or dataframe."
            )
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(
                    "The target dataframe must have only one column. If used as part of "
                    "an sklearn pipeline, ensure that previous steps return a pandas "
                    "series or dataframe."
                )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(y.index, pd.MultiIndex):
            raise ValueError("y must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")
        if not isinstance(y.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of y must be datetime.date.")

        if not X.index.equals(y.index):
            raise ValueError(
                "The indices of the input dataframe X and the output dataframe y "
                "don't match as input to the Elastic Net selector."
            )

        self.feature_names_in_ = np.array(X.columns)
        self.p = X.shape[1]

        if self.positive:

            self.enet = ElasticNet(
                alpha=self.alpha, l1_ratio=self.l1_ratio, positive=True
            ).fit(X, y)
        else:
            self.enet = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio).fit(X, y)

        self.selected_ftr_idxs = [i for i in range(self.p) if self.enet.coef_[i] != 0]

        return self

    def _get_support_mask(self):
        """
        Private method to return a boolean mask of the features selected for the Pandas dataframe.
        """
        mask = np.zeros(self.p, dtype=bool)
        mask[self.selected_ftr_idxs] = True
        return mask

    def get_support(self, indices=False):
        """
        Method to return a mask, or integer index, of the features selected for the Pandas dataframe.

        :param <bool> indices: Boolean to specify whether to return the column indices of the selected features instead of a boolean mask

        :return <np.ndarray>: Boolean mask or integer index of the selected features
        """
        if self.feature_names_in_ is None:
            raise NotFittedError(
                "The Elastic Net selector has not been fitted. Please fit the selector before calling get_support()."
            )
        if not isinstance(indices, (bool, np.bool_)):
            raise ValueError("The 'indices' parameter must be a boolean.")
        if indices:
            return self.selected_ftr_idxs
        else:
            mask = self._get_support_mask()
            return mask

    def get_feature_names_out(self):
        """
        Method to mask feature names according to selected features.
        """
        if self.feature_names_in_ is None:
            raise NotFittedError(
                "The Elastic Net selector has not been fitted. Please fit the selector before calling get_feature_names_out()."
            )

        return self.feature_names_in_[self.get_support(indices=False)]

    def transform(self, X: pd.DataFrame):
        """
        Transform method to return only the selected features of the dataframe.

        :param <pd.DataFrame> X: Pandas dataframe of input features.

        :return <pd.DataFrame>: Pandas dataframe of input features selected based
            on the Elastic Net's feature selection capabilities.
        """
        # checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the Elastic Net selector must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")
        if not X.shape[-1] == self.p:
            raise ValueError(
                "The number of columns of the dataframe to be transformed, X, doesn't "
                "match the number of columns of the training dataframe."
            )
        if len(self.selected_ftr_idxs) == 0:
            # Then no features were selected
            # Then at the given time, no trading decisions can be made based on these features
            warnings.warn(
                "No features were selected. At the given time, no trading decisions can be made based on these features.",
                UserWarning,
            )
            return X.iloc[:, :0]

        return X.iloc[:, self.selected_ftr_idxs]


class LassoSelector(BaseEstimator, SelectorMixin):
    def __init__(self, alpha: Union[float, int], positive: bool = True):
        """
        Transformer class to use the Lasso as a feature selection algorithm.
        Given a hyper-parameter, alpha, the Lasso model is fit and
        the non-zero coefficients are used to extract features from an input dataframe.
        The underlying features as input to the Lasso model are expected to be positively
        correlated with the target variable.

        :param <Union[float, int]> alpha: the regularisation imposed by the Lasso.
        :param <bool> positive: boolean to restrict estimated Lasso coefficients to
            be positive.
        """
        if not (isinstance(alpha, float) or isinstance(alpha, int)):
            raise TypeError(
                "The 'alpha' hyper-parameter must be either a float or int."
            )
        if alpha < 0:
            raise ValueError("The 'alpha' hyper-parameter must be non-negative.")
        if not isinstance(positive, bool):
            raise TypeError("The 'positive' hyper-parameter must be a boolean.")

        self.alpha = alpha
        self.positive = positive
        self.feature_names_in_ = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]):
        """
        Fit method to fit a Lasso regression and obtain the selected features.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.Series,pd.DataFrame]> y: Pandas series or dataframe of targets
            associated with each sample in X.
        """
        # checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the LASSO selector must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)):
            raise TypeError(
                "Target vector for the LASSO selector must be a pandas series or dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas series or dataframe."
            )
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(
                    "The target dataframe must have only one column. If used as part of "
                    "an sklearn pipeline, ensure that previous steps return a pandas "
                    "series or dataframe."
                )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(y.index, pd.MultiIndex):
            raise ValueError("y must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")
        if not isinstance(y.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of y must be datetime.date.")

        if not X.index.equals(y.index):
            raise ValueError(
                "The indices of the input dataframe X and the output dataframe y "
                "don't match as input to the LASSO selector."
            )

        self.feature_names_in_ = np.array(X.columns)
        self.p = X.shape[1]

        if self.positive:
            self.lasso = Lasso(alpha=self.alpha, positive=True).fit(X, y)
        else:
            self.lasso = Lasso(alpha=self.alpha).fit(X, y)

        self.selected_ftr_idxs = [i for i in range(self.p) if self.lasso.coef_[i] != 0]

        return self

    def _get_support_mask(self):
        """
        Private method to return a boolean mask of the features selected for the Pandas dataframe.
        """
        mask = np.zeros(self.p, dtype=bool)
        mask[self.selected_ftr_idxs] = True
        return mask

    def get_support(self, indices=False):
        """
        Method to return a mask, or integer index, of the features selected for the Pandas dataframe.

        :param <bool> indices: Boolean to specify whether to return the column indices of the selected features instead of a boolean mask

        :return <np.ndarray>: Boolean mask or integer index of the selected features
        """
        if self.feature_names_in_ is None:
            raise NotFittedError(
                "The LASSO selector has not been fitted. Please fit the selector before calling get_support()."
            )
        if not isinstance(indices, (bool, np.bool_)):
            raise ValueError("The 'indices' parameter must be a boolean.")
        if indices:
            return self.selected_ftr_idxs
        else:
            mask = self._get_support_mask()
            return mask

    def get_feature_names_out(self):
        """
        Method to mask feature names according to selected features.
        """
        if self.feature_names_in_ is None:
            raise NotFittedError(
                "The LASSO selector has not been fitted. Please fit the selector before calling get_feature_names_out()."
            )

        return self.feature_names_in_[self.get_support(indices=False)]

    def transform(self, X: pd.DataFrame):
        """
        Transform method to return only the selected features of the dataframe.

        :param <pd.DataFrame> X: Pandas dataframe of input features.

        :return <pd.DataFrame>: Pandas dataframe of input features selected based
            on the Lasso's feature selection capabilities.
        """
        # checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the LASSO selector must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")
        if not X.shape[-1] == self.p:
            raise ValueError(
                "The number of columns of the dataframe to be transformed, X, doesn't "
                "match the number of columns of the training dataframe."
            )
        if len(self.selected_ftr_idxs) == 0:
            # Then no features were selected
            # Then at the given time, no trading decisions can be made based on these features
            warnings.warn(
                "No features were selected. At the given time, no trading decisions can be made based on these features.",
                UserWarning,
            )
            return X.iloc[:, :0]

        return X.iloc[:, self.selected_ftr_idxs]


class MapSelector(BaseEstimator, SelectorMixin):
    def __init__(self, threshold: float = 0.05, positive: bool = False):
        """
        Selector class to select features from a training set
        based on the Macrosynergy panel test. This test involves creating
        a linear mixed effects model with period-specific random effects to
        account for cross-sectional correlations. The p-value for the slope
        parameter is used to perform the significance test.

        :param <float> threshold: Significance threshold. This should be in
            the interval (0,1). Default is 0.05.
        :param <bool> positive: Boolean indicating whether or not to only keep features
            with positive estimated model coefficients. Default is False.
        """
        if type(threshold) != float:
            raise TypeError("The threshold must be a float.")
        if (threshold <= 0) or (threshold > 1):
            raise ValueError(
                "The threshold must be in between 0 (inclusive) and 1 (exclusive)."
            )
        if not isinstance(positive, (bool, np.bool_)):
            raise TypeError("The 'positive' parameter must be a boolean.")

        self.threshold = threshold
        self.positive = positive
        self.feature_names_in_ = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]):
        """
        Fit method to assess significance of each feature using
        the Macrosynergy panel test.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.Series, pd.DataFrame]> y: Pandas series or dataframe of targets
            associated with each sample in X.
        """
        # Checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix for the MAP selector must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)):
            raise TypeError(
                "Target vector for the MAP selector must be a pandas series or dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas series or dataframe."
            )
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(
                    "The target dataframe must have only one column. If used as part of "
                    "an sklearn pipeline, ensure that previous steps return a pandas "
                    "series or dataframe."
                )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(y.index, pd.MultiIndex):
            raise ValueError("y must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")
        if not isinstance(y.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of y must be datetime.date.")
        if not X.index.equals(y.index):
            raise ValueError(
                "The indices of the input dataframe X and the output dataframe y don't "
                "match."
            )

        self.ftrs = []
        self.feature_names_in_ = np.array(X.columns)

        # For each column, obtain Wald test p-value
        # Keep significant features
        for col in self.feature_names_in_:
            ftr = X[col]

            re = RandomEffects(fit_intercept=True).fit(ftr, y)
            pval = re.pvals[col]
            if pval < self.threshold:
                if self.positive:
                    if re.params[col] > 0:
                        self.ftrs.append(col)
                else:
                    self.ftrs.append(col)

        return self

    def _get_support_mask(self):
        """
        Private method to return a boolean mask of the features selected for the Pandas dataframe.
        """
        mask = [col in self.ftrs for col in self.feature_names_in_]
        return np.array(mask)

    def get_support(self, indices=False):
        """
        Method to return a mask, or integer index, of the features selected for the Pandas dataframe.

        :param <bool> indices: Boolean to specify whether to return the column indices of the selected features instead of a boolean mask

        :return <np.ndarray>: Boolean mask or integer index of the selected features
        """
        if not isinstance(indices, (bool, np.bool_)):
            raise TypeError("The 'indices' parameter must be a boolean.")
        if self.feature_names_in_ is None:
            raise NotFittedError(
                "The MAP selector has not been fitted. Please fit the selector before calling get_support()."
            )
        mask = self._get_support_mask()

        if indices:
            return np.where(mask)[0]

        return mask

    def get_feature_names_out(self):
        """
        Method to mask feature names according to selected features.
        """
        if self.feature_names_in_ is None:
            raise ValueError(
                "The feature names are not available. Please fit the transformer first."
            )

        return self.feature_names_in_[self.get_support(indices=False)]

    def transform(self, X: pd.DataFrame):
        """
        Transform method to return the significant training features.

        :param <pd.DataFrame> X: Pandas dataframe of input features.

        :return <pd.DataFrame>: Pandas dataframe of input features selected
            based on the Macrosynergy panel test.
        """
        # checks
        if type(X) != pd.DataFrame:
            raise TypeError(
                "Input feature matrix for the MAP selector must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")
        if not np.all(np.array(X.columns) == self.feature_names_in_):
            raise ValueError(
                "The columns of the dataframe to be transformed, X, don't match the "
                "columns of the training dataframe."
            )
        # transform
        if self.ftrs == []:
            # Then no features were selected
            # Then at the given time, no trading decisions can be made based on these features
            warnings.warn(
                "No features were selected. At the given time, no trading decisions can be made based on these features."
            )
            return X.iloc[:, :0]

        return X[self.ftrs]


class FeatureAverager(BaseEstimator, TransformerMixin):
    def __init__(self, use_signs: Optional[bool] = False):
        """
        Transformer class to combine features into a benchmark signal by averaging.

        :param <Optional[bool]> use_signs: Boolean to specify whether or not to return the
            signs of the benchmark signal instead of the signal itself. Default is False.
        """
        if not isinstance(use_signs, bool):
            raise TypeError("'use_signs' must be a boolean.")

        self.use_signs = use_signs

    def fit(self, X: pd.DataFrame, y: Any = None):
        """
        Fit method. Since this transformer is a simple averaging of features,
        no fitting is required.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Any> y: Placeholder for scikit-learn compatibility.
        """

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method to average features into a benchmark signal.
        If use_signs is True, the signs of the benchmark signal are returned.

        :param <pd.DataFrame> X: Pandas dataframe of input features.

        :return <pd.DataFrame>: Pandas dataframe of benchmark signal.
        """
        # checks
        if type(X) != pd.DataFrame:
            raise TypeError(
                "Input feature matrix for the FeatureAverager must be a pandas dataframe."
                " If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")

        # transform
        signal_df = X.mean(axis=1).to_frame(name="signal")
        if self.use_signs:
            return np.sign(signal_df).astype(int)

        return signal_df


class ZnScoreAverager(BaseEstimator, TransformerMixin):
    def __init__(self, neutral: str = "zero", use_signs: bool = False):
        """
        Transformer class to combine features into a benchmark signal
        according to a specified normalisation.

        :param <str> neutral: Specified neutral value for a normalisation. This can
            take either 'zero' or 'mean'. If the neutral level is zero, each feature
            is standardised by dividing by the mean absolute deviation. If the neutral
            level is mean, each feature is normalised by subtracting the mean and
            dividing by the standard deviation. Any statistics are computed according
            to a point-in-time principle. Default is 'zero'.
        :param <bool> use_signs: Boolean to specify whether or not to return the
            signs of the benchmark signal instead of the signal itself. Default is False.
        """
        if not isinstance(neutral, str):
            raise TypeError("'neutral' must be a string.")

        if neutral not in ["zero", "mean"]:
            raise ValueError("neutral must be either 'zero' or 'mean'.")

        if not isinstance(use_signs, bool):
            raise TypeError("'use_signs' must be a boolean.")

        self.neutral = neutral
        self.use_signs = use_signs

    def fit(self, X: pd.DataFrame, y: Any = None):
        """
        Fit method to extract relevant standardisation/normalisation statistics from a
        training set so that PiT statistics can be computed in the transform method for
        a hold-out set.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Any> y: Placeholder for scikit-learn compatibility.
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

        # fit
        self.training_n: int = len(X)

        if self.neutral == "mean":
            # calculate the mean and sum of squares of each feature
            means: pd.Series = X.mean(axis=0)
            sum_squares: pd.Series = np.sum(np.square(X), axis=0)
            self.training_means: pd.Series = means
            self.training_sum_squares: pd.Series = sum_squares
        else:
            # calculate the mean absolute deviation of each feature
            mads: pd.Series = np.mean(np.abs(X), axis=0)
            self.training_mads: pd.Series = mads

        return self

    def transform(self, X: pd.DataFrame):
        """
        Transform method to compute an out-of-sample benchmark signal for each unique
        date in the input test dataframe. At a given test time, the relevant statistics
        (implied by choice of neutral value) are calculated using all training information
        and test information until (and including) that test time, since the test time
        denotes the time at which the return was available and the features lag behind
        the returns.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
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
        Helper method to get the number of non-NaN values in each expanding window.

        :param <pd.DataFrame> X: Pandas dataframe of input features.

        :return <np.ndarray>: Numpy array of expanding counts.
        """
        return X.groupby(level="real_date").count().expanding().sum().to_numpy()


class PanelMinMaxScaler(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """
    Transformer class to extend scikit-learn's MinMaxScaler() to panel datasets. It is
    intended to replicate the aforementioned class, but critically returning
    a Pandas dataframe or series instead of a numpy array. This preserves the
    multi-indexing in the inputs after transformation, allowing for the passing
    of standardised features into transformers that require cross-sectional
    and temporal knowledge.

    NOTE: This class is designed to replicate scikit-learn's MinMaxScaler() class.
            It should primarily be used to satisfy the assumptions of various models.
    """

    def fit(self, X, y=None):
        """
        Fit method to determine minimum and maximum values over a training set.

        :param <Union[pd.DataFrame, pd.Series]> X: Pandas dataframe or series.
        :param <Any> y: Placeholder for scikit-learn compatibility.

        :return <PanelMinMaxScaler>: Fitted PanelMinMaxScaler object.
        """
        # checks
        if type(X) not in [pd.DataFrame, pd.Series]:
            raise TypeError(
                "'X' must be a pandas dataframe or series. If used as part of an "
                "sklearn pipeline, ensure that previous steps return a pandas dataframe "
                "or series."
            )

        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")

        # fit
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

        return self

    def transform(self, X):
        """
        Transform method to standardise a panel based on the minimum and maximum values.

        :param <Union[pd.DataFrame, pd.Series]> X: Pandas dataframe or series.

        :return <Union[pd.DataFrame, pd.Series]>: Standardised dataframe or series.
        """
        # checks
        if type(X) not in [pd.DataFrame, pd.Series]:
            raise TypeError(
                "'X' must be a pandas dataframe or series. If used as part of an "
                "sklearn pipeline, ensure that previous steps return a pandas dataframe "
                "or series."
            )

        # transform
        calc = (X - self.mins) / (self.maxs - self.mins)

        return calc


class PanelStandardScaler(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        """
        Transformer class to extend scikit-learn's StandardScaler() to panel datasets.
        It is intended to replicate the aforementioned class, but critically returning
        a Pandas dataframe or series instead of a numpy array. This preserves the
        multi-indexing in the inputs after transformation, allowing for the passing
        of standardised features into transformers that require cross-sectional
        and temporal knowledge.

        NOTE: This class is designed to replicate scikit-learn's StandardScaler() class.
                It is not designed to perform sequential mean and standard deviation
                normalisation like the 'make_zn_scores()' function in 'macrosynergy.panel'
                or 'ZnScoreAverager' in 'macrosynergy.learning'.
                This class should primarily be used to satisfy the assumptions of various
                models, for example the Lasso, Ridge or any neural network.

        :param <bool> with_mean: Boolean to specify whether or not to centre the data.
        :param <bool> with_std: Boolean to specify whether or not to scale the data.
        """
        # checks
        if not isinstance(with_mean, bool):
            raise TypeError("'with_mean' must be a boolean.")
        if not isinstance(with_std, bool):
            raise TypeError("'with_std' must be a boolean.")

        # setup
        self.with_mean = with_mean
        self.with_std = with_std

        self.means = None
        self.stds = None

    def fit(self, X: Union[pd.DataFrame, pd.Series], y: Any = None):
        """
        Fit method to determine means and standard deviations over a training set.

        :param <Union[pd.DataFrame, pd.Series]> X: Pandas dataframe or series.
        :param <Any> y: Placeholder for scikit-learn compatibility.

        :return <PanelStandardScaler>: Fitted PanelStandardScaler object.
        """
        # checks
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series):
            raise TypeError(
                "'X' must be a pandas dataframe or series. If used as part of an sklearn "
                "pipeline, ensure that previous steps return a pandas dataframe or "
                "series."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        # fit
        if self.with_mean:
            self.means = X.mean(axis=0)

        if self.with_std:
            self.stds = X.std(axis=0)

        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]):
        """
        Transform method to standardise a panel based on the means and standard deviations
        learnt from a training set (and the fit method).

        :param <Union[pd.DataFrame, pd.Series]> X: Pandas dataframe or series.

        :return <Union[pd.DataFrame, pd.Series]>: Standardised dataframe or series.
        """
        # checks
        if type(X) not in [pd.DataFrame, pd.Series]:
            raise TypeError(
                "'X' must be a pandas dataframe or series. If used as part of an sklearn "
                "pipeline, ensure that previous steps return a pandas dataframe or "
                "series."
            )

        # transform
        if self.means is not None:
            calc = X - self.means
        else:
            calc = X

        if self.stds is not None:
            calc = calc / self.stds

        return calc

class PanelPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components = None, kaiser_criterion=False, adjust_signs = False):
        """
        PCA transformer for panel data. If kaiser_criterion is True,
        this overrides `n_components`.

        :param <int> n_components: Number of components to keep. If None, all components are kept.
        :param <bool> kaiser_criterion: If True, only components with eigenvalues greater or equal to one are kept.
        :param <bool> adjust_signs: If True, adjust signs of eigenvectors so that projected training features
            are positively correlated with a provided target variable. This is useful for consistency
            when used in a sequential learning pipeline through time. 

        .. note::

          This class is still **experimental**: the predictions
          and the API might change without any deprecation cycle.
        """
        if n_components is not None:
            if not isinstance(n_components, numbers.Number) or isinstance(n_components, bool):
                raise TypeError("n_components must be a number or None.")
            if n_components <= 0:
                raise ValueError("n_components must be greater than 0.")
            
        if not isinstance(kaiser_criterion, bool):
            raise TypeError("kaiser_criterion must be a boolean.")
        
        if not isinstance(adjust_signs, bool):
            raise TypeError("adjust_signs must be a boolean.")
        
        self.n_components = n_components
        self.kaiser_criterion = kaiser_criterion
        self.adjust_signs = adjust_signs
        
    def fit(self, X, y=None):
        """
        Fit method to determine an eigenbasis for the PCA.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas dataframe or series of targets.
            This is only used to adjust the signs of principal components to allow
            for greater consistency during sequential learning. 
        """
        if self.adjust_signs:
            if y is None:
                raise ValueError(
                    "If `adjust_signs` is True, a target variable must be provided. "
                    "PCA is an unsupervised method, so providing a target variable "
                    "does not affect the PCA. Eigenvectors, however, are unique up to "
                    "its sign, meaning that a target variable can be used to ensure "
                    "consistency in the signs of the eigenvectors through time and over repeated runs."
                )
            
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
            mask = (self.adjusted_evals >= 1)
            self.adjusted_evals = self.adjusted_evals[mask]
            self.adjusted_evecs = self.adjusted_evecs[:,mask]
        elif isinstance(self.n_components, int):
            # Keep first n_components components
            self.adjusted_evals = self.adjusted_evals[:self.n_components]
            self.adjusted_evecs = self.adjusted_evecs[:,:self.n_components]
        elif isinstance(self.n_components, float):
            # Keep components that explain a certain percentage of difference. 
            variance_explained = self.adjusted_evals / np.sum(self.adjusted_evals)
            cumulative_variance_explained = np.cumsum(variance_explained)
            mask = (cumulative_variance_explained <= self.n_components)
            self.adjusted_evals = self.adjusted_evals[mask]
            self.adjusted_evecs = self.adjusted_evecs[:,mask]

        # Adjust signs of eigenvectors so that projected data is positively correlated with y
        if y is not None:
            y = y.values
            for i in range(self.adjusted_evecs.shape[1]):
                if np.corrcoef(X.values @ self.adjusted_evecs[:,i], y)[0,1] < 0:
                    self.adjusted_evecs[:,i] *= -1

        return self
            
    def transform(self, X):
        """
        Transform method to project the input features onto the principal components.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        """
        return pd.DataFrame(
            index = X.index,
            columns = [f"PCA {i+1}" for i in range(self.adjusted_evecs.shape[1])],
            data = X.values @ self.adjusted_evecs
        )


if __name__ == "__main__":
    import macrosynergy.management as msm
    from macrosynergy.management import make_qdf

    np.random.seed(1)

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    """Example: Unbalanced panel """

    df_cids2 = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids2.loc["AUD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["CAD"] = ["2013-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["GBP"] = ["2010-01-01", "2020-12-31", 0, 1]
    df_cids2.loc["USD"] = ["2010-01-01", "2020-12-31", 0, 1]

    df_xcats2 = pd.DataFrame(index=xcats, columns=cols)
    df_xcats2.loc["XR"] = ["2010-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats2.loc["CRY"] = ["2010-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats2.loc["GROWTH"] = ["2010-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats2.loc["INFL"] = ["2010-01-01", "2020-12-31", 1, 2, 0.8, 0.5]

    dfd2 = make_qdf(df_cids2, df_xcats2, back_ar=0.75)
    dfd2["grading"] = np.ones(dfd2.shape[0])
    black = {"GBP": ["2009-01-01", "2012-06-30"], "CAD": ["2018-01-01", "2100-01-01"]}
    dfd2 = msm.reduce_df(df=dfd2, cids=cids, xcats=xcats, blacklist=black)

    dfd2 = dfd2.pivot(index=["cid", "real_date"], columns="xcat", values="value")
    X = dfd2.drop(columns=["XR"])
    y = dfd2["XR"]

    selector = LassoSelector(0.2)
    selector.fit(X, y)
    print(selector.transform(X).columns)

    selector = MapSelector(1e-20)
    selector.fit(X, y)
    print(selector.transform(X).columns)

    # Split X and y into training and test sets
    X_train, X_test = (
        X[X.index.get_level_values(1) < pd.Timestamp(day=1, month=1, year=2018)],
        X[X.index.get_level_values(1) >= pd.Timestamp(day=1, month=1, year=2018)],
    )
    y_train, y_test = (
        y[y.index.get_level_values(1) < pd.Timestamp(day=1, month=1, year=2018)],
        y[y.index.get_level_values(1) >= pd.Timestamp(day=1, month=1, year=2018)],
    )

    transformer = PanelPCA(n_components = 2, adjust_signs=True)
    transformer.fit(X_train, y_train)
    print(transformer.transform(X_test))

    selector = ZnScoreAverager(neutral="zero", use_signs=False)
    selector.fit(X_train, y_train)
    print(selector.transform(X_test))

    selector = ZnScoreAverager(neutral="zero")
    selector.fit(X_train, y_train)
    print(selector.transform(X_test))
