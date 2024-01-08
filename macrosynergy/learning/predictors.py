import inspect

import numpy as np
import pandas as pd

import datetime

from sklearn.base import BaseEstimator, RegressorMixin

from typing import Union


class NaivePredictor(BaseEstimator, RegressorMixin):
    """
    Naive predictor class to output a column of features as the predictions themselves.
    It expects only one column in the input matrix for which predictions are to be made.
    This class would typically be used in the final stage of a `scikit-learn` pipeline,
    following either `FeatureAverager` or `ZnScoreAverager` stages.
    """

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.array(X)


class SignWeightedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor: RegressorMixin, **init_kwargs):
        """
        Custom class to create regressions with sign-weighted training samples.

        :param <RegressorMixin> regressor: Scikit-learn regressor class to be used as the
            underlying model for which a sign-weighted fit is to be performed.

        NOTE: This class is specifically tailored for compatibility with any scikit-learn
        regressor that supports a 'sample_weight' parameter in its 'fit' method. Each
        training sample receives a weight inversely proportional to the frequency of its
        label's sign in the training set. Thus, the model is encouraged to learn equally
        from both positive and negative return samples. If there
        are more positive targets than negative targets in the training set, then the
        negative target samples are given a higher weight in the model training process.
        The opposite is true if there are more negative targets than positive targets.


        """
        if not callable(regressor) or not inspect.isclass(regressor):
            raise TypeError(
                "Regressor must be a class, such as 'LinearRegression'. Please check it isn't an instantiation of a class, such as 'LinearRegression()'."
            )

        if not issubclass(regressor, RegressorMixin) or not issubclass(
            regressor, BaseEstimator
        ):
            raise TypeError(
                "Regressor must be a subclass of both BaseEstimator and RegressorMixin."
            )

        if not hasattr(regressor, "fit") or not callable(getattr(regressor, "fit")):
            raise TypeError("Regressor must have a callable 'fit' method.")

        if not hasattr(regressor, "predict") or not callable(
            getattr(regressor, "predict")
        ):
            raise TypeError("Regressor must have a callable 'predict' method.")

        fit_params = inspect.signature(regressor.fit).parameters
        if "sample_weight" not in fit_params:
            raise ValueError(
                f"The underlying scikit-learn regressor {regressor} must have a 'sample_weight' parameter in its 'fit' method."
            )

        self.model = regressor(**init_kwargs)

    def __calculate_sample_weights(self, targets: Union[pd.DataFrame, pd.Series]):
        """
        Private helper method to calculate the sample weights, chosen by inverse frequency
        of the label's sign in the training set.

        :param <Union[pd.DataFrame, pd.Series]> targets: Pandas series or dataframe of targets.

        :return <tuple[np.ndarray, float, float]>: Tuple of sample weights, positive weight and negative weight.
        """
        pos_sum = np.sum(targets >= 0)
        neg_sum = np.sum(targets < 0)

        pos_weight = len(targets) / (2 * pos_sum) if pos_sum > 0 else 0
        neg_weight = len(targets) / (2 * neg_sum) if neg_sum > 0 else 0

        sample_weights = np.where(targets >= 0, pos_weight, neg_weight)
        return sample_weights, pos_weight, neg_weight

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]):
        """
        Fit method to fit the underlying model with sign weighted samples, as passed
        into the constructor.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
            associated with each sample in X.

        :return <SignWeightedRegressor>
        """
        # Checks
        if type(X) != pd.DataFrame:
            raise TypeError(
                "Input feature matrix for the SignWeightedRegressor must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if (type(y) != pd.Series) and (type(y) != pd.DataFrame):
            raise TypeError(
                "Target vector for the SignWeightedRegressor must be a pandas series or dataframe. "
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

        # Fit
        self.sample_weights, _, _ = self.__calculate_sample_weights(y)
        self.model.fit(X, y, sample_weight=self.sample_weights)
        return self

    def predict(self, X: pd.DataFrame):
        """
        Predict method to make model predictions on the input feature matrix X based on
        the previously fit model.

        :param <pd.DataFrame> X: Pandas dataframe of input features.

        :return <np.ndarray>: Numpy array of predictions.
        """
        # Checks
        if type(X) != pd.DataFrame:
            raise TypeError(
                "Input feature matrix for the SignWeightedRegressor must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")

        # Predict
        return self.model.predict(X)


class TimeWeightedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor: RegressorMixin, half_life: Union[float,int], **init_kwargs):
        """
        Custom class to create regressions with exponentially-weighted training samples.

        :param <RegressorMixin> regressor: Scikit-learn regressor class to be used as the
            underlying model for which a time-weighted fit is to be performed.
        :param <Union[float,int]> half_life: Half-life of an exponential decay function used to
            calculate the sample weights. The half-life is the number of time units for
            the weight to decay to half its original value. The greater the half-life, the
            greater that more historical samples are weighted in the model.

        NOTE: This class is specifically tailored for compatibility with any
        scikit-learn regressor that supports a 'sample_weight' parameter in its 'fit'
        method. Each training sample receives a weight based on the date attributed to
        the sample. The weights are calculated as an exponentially decaying function of
        the date, with the half-life of the decay specified by the user.
        Consequently, during the model training process, more recent samples
        are emphasized due to the higher weight they receive. By weighting based on
        recency, the model is encouraged to prioritise newer information.
        """
        # Checks
        if not callable(regressor) or not inspect.isclass(regressor):
            raise TypeError(
                "Regressor must be a class, such as 'LinearRegression'. Please check it isn't an instantiation of a class, such as 'LinearRegression()'."
            )

        if not issubclass(regressor, RegressorMixin) or not issubclass(
            regressor, BaseEstimator
        ):
            raise TypeError(
                "Regressor must be a subclass of both BaseEstimator and RegressorMixin."
            )

        if not hasattr(regressor, "fit") or not callable(getattr(regressor, "fit")):
            raise TypeError("Regressor must have a callable 'fit' method.")

        if not hasattr(regressor, "predict") or not callable(
            getattr(regressor, "predict")
        ):
            raise TypeError("Regressor must have a callable 'predict' method.")

        fit_params = inspect.signature(regressor.fit).parameters
        if "sample_weight" not in fit_params:
            raise ValueError(
                f"The underlying scikit-learn regressor {regressor} must have a 'sample_weight' parameter in its 'fit' method."
            )
        if type(half_life) != float and type(half_life) != int:
            raise TypeError("The half-life must be either a float or an integer.")
        if half_life <= 0:
            raise ValueError("The half-life must be greater than zero.")
        
        # Init
        self.model = regressor(**init_kwargs)
        self.half_life = half_life

    def __calculate_sample_weights(self, targets: Union[pd.DataFrame, pd.Series]):
        """
        Private helper method to calculate the sample weights, chosen by an exponentially
        decaying function of the sample recency. 

        :param <Union[pd.DataFrame, pd.Series]> targets: Pandas series or dataframe of targets.

        :return <np.ndarray>: Numpy array of sample weights.
        """

        dates = sorted(targets.index.get_level_values(1).unique(), reverse=True)
        num_dates = len(dates)
        weights = [2 ** (-t / self.half_life) for t in np.arange(num_dates)]
        weights = weights / np.sum(weights)

        weight_map = dict(zip(dates, weights))
        self.sample_weights = targets.index.get_level_values(1).map(weight_map)

        return self.sample_weights

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]):
        """
        Fit method to fit the underlying model with time-weighted samples, as passed
        into the constructor.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
            associated with each sample in X.

        :return <TimeWeightedRegressor>
        """
        # Checks 
        if type(X) != pd.DataFrame:
            raise TypeError(
                "Input feature matrix for the TimeWeightedRegressor must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if (type(y) != pd.Series) and (type(y) != pd.DataFrame):
            raise TypeError(
                "Target vector for the TimeWeightedRegressor must be a pandas series or dataframe. "
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
        
        # fit
        sample_weights = self.__calculate_sample_weights(y)
        self.model.fit(X, y, sample_weight=sample_weights)
        return self

    def predict(self, X: pd.DataFrame):
        """
        Predict method to make model predictions on the input feature matrix X based on
        the previously fit model.

        :param <pd.DataFrame> X: Pandas dataframe of input features.

        :return <np.ndarray>: Numpy array of predictions.
        """
        # Checks
        if type(X) != pd.DataFrame:
            raise TypeError(
                "Input feature matrix for the TimeWeightedRegressor must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")
        
        # Predict
        return self.model.predict(X)

if __name__ == "__main__":
    from macrosynergy.management import make_qdf
    import macrosynergy.management as msm
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LinearRegression

    np.random.seed(1)

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    """Example: Unbalanced panel """

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2013-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2010-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2010-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2010-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2010-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2010-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2010-01-01", "2020-12-31", 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {"GBP": ["2009-01-01", "2012-06-30"], "CAD": ["2018-01-01", "2100-01-01"]}
    dfd = msm.reduce_df(df=dfd, cids=cids, xcats=xcats, blacklist=black)

    dfd = dfd.pivot(index=["cid", "real_date"], columns="xcat", values="value")
    X = dfd.drop(columns=["XR"])
    y = dfd["XR"]
    
    # OLS Linear Regression
    lm = LinearRegression()
    lm.fit(X, y)
    lm_preds = lm.predict(X)
    # Sign-Weighted Linear Regression
    swlm = SignWeightedRegressor(LinearRegression)
    swlm.fit(X, y)
    swlm_preds = swlm.predict(X)
    # Time-Weighted Linear Regression
    twlm = TimeWeightedRegressor(LinearRegression, half_life=12*21) # One year half life
    twlm.fit(X, y)
    twlm_preds = twlm.predict(X)

    # Plot results
    plt.title("OLS vs Sign-Weighted vs Time-Weighted Linear Regression on mock random data")
    plt.plot(lm_preds, label="OLS", color="tab:blue",alpha=0.5)
    plt.plot(swlm_preds, label="Sign-Weighted", color="tab:orange", alpha=0.5)
    plt.plot(twlm_preds, label="Time-Weighted", color="tab:green", alpha=0.5)

    plt.legend()
    plt.show()
    
    # Plot histogram of predictions
    fig, ax = plt.subplots(ncols=3)
    ax[0].hist(lm_preds, bins=200, color="tab:blue")
    ax[0].set_title("OLS")
    ax[1].hist(swlm_preds, bins=200, color="tab:orange")
    ax[1].set_title("Sign-Weighted")
    ax[2].hist(twlm_preds, bins=200, color="tab:green")
    ax[2].set_title("Time-Weighted")

    plt.suptitle("Histograms of predictions")
    plt.tight_layout()
    plt.show()
