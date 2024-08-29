import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin

from typing import Any, Union

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