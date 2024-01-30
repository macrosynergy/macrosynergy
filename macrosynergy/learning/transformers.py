"""
Collection of custom scikit-learn transformer classes.
"""

import numpy as np
import pandas as pd

import datetime

from sklearn.linear_model import Lasso
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin

from statsmodels.tools.tools import add_constant
from statsmodels.regression.mixed_linear_model import MixedLM

from typing import Union, Any, List, Optional

import logging


class LassoSelector(BaseEstimator, TransformerMixin):
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
        if (type(alpha) != float) and (type(alpha) != int):
            raise TypeError(
                "The 'alpha' hyper-parameter must be either a float or int."
            )
        if alpha < 0:
            raise ValueError("The 'alpha' hyper-parameter must be non-negative.")
        if type(positive) != bool:
            raise TypeError("The 'positive' hyper-parameter must be a boolean.")

        self.alpha = alpha
        self.positive = positive

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]):
        """
        Fit method to fit a Lasso regression and obtain the selected features.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.Series,pd.DataFrame]> y: Pandas series or dataframe of targets
            associated with each sample in X.
        """
        # checks
        if type(X) != pd.DataFrame:
            raise TypeError(
                "Input feature matrix for the LASSO selector must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if (type(y) != pd.Series) and (type(y) != pd.DataFrame):
            raise TypeError(
                "Target vector for the LASSO selector must be a pandas series or dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas series or dataframe."
            )
        if type(y) == pd.DataFrame:
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

        self.p = X.shape[-1]

        if self.positive:
            self.lasso = Lasso(alpha=self.alpha, positive=True).fit(X, y)
        else:
            self.lasso = Lasso(alpha=self.alpha).fit(X, y)

        self.selected_ftr_idxs = [i for i in range(self.p) if self.lasso.coef_[i] != 0]

        return self

    def transform(self, X: pd.DataFrame):
        """
        Transform method to return only the selected features of the dataframe.

        :param <pd.DataFrame> X: Pandas dataframe of input features.

        :return <pd.DataFrame>: Pandas dataframe of input features selected based
            on the Lasso's feature selection capabilities.
        """
        # checks
        if type(X) != pd.DataFrame:
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
            # Hence, we return a dataframe of zeros
            return pd.DataFrame(
                index=X.index, columns=["no_signal"], data=0, dtype=np.float16
            )
        
        return X.iloc[:, self.selected_ftr_idxs]


class MapSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float):
        """
        Transformer class to select features from a training set
        based on the Macrosynergy panel test. This test involves creating
        a linear mixed effects model with period-specific random effects to
        account for cross-sectional correlations. The p-value for the slope
        parameter is used to perform the significance test.

        :param <float> threshold: Significance threshold. This should be in
            the interval (0,1).
        """
        if type(threshold) != float:
            raise TypeError("The threshold must be a float.")

        if (threshold <= 0) or (threshold > 1):
            raise ValueError(
                "The threshold must be in between 0 (inclusive) and 1 (exclusive)."
            )

        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]):
        """
        Fit method to assess significance of each feature using
        the Macrosynergy panel test.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.Series, pd.DataFrame]> Pandas series or dataframe of targets
            associated with each sample in X.
        """
        # Checks
        if type(X) != pd.DataFrame:
            raise TypeError(
                "Input feature matrix for the MAP selector must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if (type(y) != pd.Series) and (type(y) != pd.DataFrame):
            raise TypeError(
                "Target vector for the MAP selector must be a pandas series or dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas series or dataframe."
            )
        if type(y) == pd.DataFrame:
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
        self.cols = X.columns

        for col in self.cols:
            ftr = X[col]
            ftr = add_constant(ftr)
            groups = ftr.index.get_level_values(1)
            model = MixedLM(y.values, ftr, groups).fit(reml=False)
            est = model.params.iloc[1]
            pval = model.pvalues.iloc[1]
            if (pval < self.threshold) & (est > 0):
                self.ftrs.append(col)

        return self

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
        if not X.columns.equals(self.cols):
            raise ValueError(
                "The columns of the dataframe to be transformed, X, don't match the "
                "columns of the training dataframe."
            )
        # transform
        if self.ftrs == []:
            # Then no features were selected
            # Then at the given time, no trading decisions can be made based on these features
            # Hence, we return a dataframe of zeros
            return pd.DataFrame(
                index=X.index, columns=["no_signal"], data=0, dtype=np.float16
            )

        return X[self.ftrs]


class FeatureAverager(BaseEstimator, TransformerMixin):
    def __init__(self, use_signs: Optional[bool] = False):
        """
        Transformer class to combine features into a benchmark signal by averaging.

        :param <Optional[bool]> use_signs: Boolean to specify whether or not to return the
            signs of the benchmark signal instead of the signal itself. Default is False.
        """
        if type(use_signs) != bool:
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
        if type(X) != pd.DataFrame:
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
        if type(X) != pd.DataFrame:
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
        if type(with_mean) != bool:
            raise TypeError("'with_mean' must be a boolean.")
        if type(with_std) != bool:
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
        if type(X) not in [pd.DataFrame, pd.Series]:
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


if __name__ == "__main__":
    from macrosynergy.management import make_qdf
    import macrosynergy.management as msm

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

    selector = MapSelector(0.05)
    selector.fit(X, y)
    print(selector.transform(X).columns)

    selector = LassoSelector(10000)
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

    selector = ZnScoreAverager(neutral="zero", use_signs=False)
    selector.fit(X_train, y_train)
    print(selector.transform(X_test))

    selector = ZnScoreAverager(neutral="zero")
    selector.fit(X_train, y_train)
    print(selector.transform(X_test))
