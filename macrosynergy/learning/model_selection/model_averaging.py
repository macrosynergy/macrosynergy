import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, make_scorer, check_scoring
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import BaseCrossValidator

from macrosynergy.learning import ExpandingKFoldPanelSplit

import numbers

class ModelAveragingRegressor(BaseEstimator, RegressorMixin):
    """
    Ensemble of regression models weighted by their cross-validation performance.

    .. note::
        This class is still **experimental**: the predictions and the API might change
        without any deprecation cycle.

    Parameters
    ----------
    estimators : list of tuples
        List of (name, estimator, param_grid) tuples where:
        - name: str, name of the estimator
        - estimator: scikit-learn regressor object
        - param_grid: dict, hyperparameter grid for GridSearchCV
    scoring : callable, default=make_scorer(r2_score, greater_is_better=True)
        Scikit-learn compatible scorer to evaluate the performance of each estimator during
        cross-validation.
    cv : cross-validation class, default=ExpandingKFoldPanelSplit(n_splits=5)
        Cross-validation splitting strategy. Can be any scikit-learn compatible
        cross-validation class. 
    temperature : str or float, default="max-min"
        Method to scale the cross-validation scores for the softmax weighting. Options are:
        - "max-min": scale by the range (max - min) of the scores
        - "std": scale by the standard deviation of the scores
        - "mad": scale by the median absolute deviation of the scores
        - "iqr": scale by the interquartile range of the scores
        - float: a custom scaling factor
    min_weight : float, default=0.0
        Minimum weight to be included in the final ensemble. Weights below this threshold
        will be set to zero.
    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting. If set to "
        raise", the error is raised. If numeric, the score is set to this value.

    Notes
    -----
    Model averaging uses cross-validation to estimate a probability distribution over a
    discrete space of models. The final prediction is a weighted average of the predictions
    from each model. The weights correspond to the estimated probability of each model
    being the "optimal" model, based on estimated out-of-sample performance. 

    One issue with the cross-validation estimator is that maximising the scores amplifies 
    noise in the estimation, resulting in a "winners curse" effect. When a large number of
    models are considered, this selection bias is more pronounced and can result in
    detrimental trading signals, simply by flickering between similar models across time 
    periods, even if those models have very similar out-of-sample performance. The dynamics
    caused by this model selection overfitting impacts the stability of the trading signal,
    not to mention transaction costs. 
    """
    def __init__(
        self,
        estimators, 
        scoring = make_scorer(r2_score, greater_is_better=True),
        cv = ExpandingKFoldPanelSplit(n_splits = 5),
        temperature = "max-min",
        min_weight = 0.0,
        error_score = np.nan,
    ):
        # Checks
        self._check_init_params(
            estimators, 
            scoring, 
            cv, 
            temperature, 
            min_weight,
            error_score
        )
        
        # Attributes
        self.estimators = estimators
        self.scoring = scoring
        self.cv = cv
        self.temperature = temperature
        self.min_weight = min_weight
        self.error_score = error_score

    def fit(self, X, y):
        # Checks
        self._check_fit_params(X, y)

        # Run a grid search for each estimator
        self.searches_ = {}
        self.best_estimators_ = {}
        self.best_params_ = {}
        self.cv_scores_ = {}
        
        for name, estimator, param_grid in self.estimators:
            gs = GridSearchCV(
                estimator = clone(estimator),
                param_grid = param_grid,
                scoring = self.scoring,
                cv = self.cv,
                refit = True,
                verbose = 0,
                error_score = self.error_score,
            ) 
            gs.fit(X, y)

            self.searches_[name] = gs
            self.best_estimators_[name] = gs.best_estimator_
            self.best_params_[name] = gs.best_params_
            self.cv_scores_[name] = gs.best_score_

        self.model_names_ = [name for name, _, _ in self.estimators]
        self.weights_ = self._compute_weights(self.cv_scores_, self.temperature, self.min_weight)
        
        # Store whether this is a single or multi-output model
        if isinstance(y, pd.DataFrame):
            self.multi_output_ = y.shape[1] > 1
            self.targets_ = y.columns.tolist()
        else:
            self.multi_output_ = False
            self.targets_ = None

        # For now, return feature importances of the model with the highest weight, if available
        # For comparability, get absolute value of the feature importances, and sum to one
        best_model_name = max(self.weights_, key=self.weights_.get)
        best_model = self.best_estimators_[best_model_name]
        if hasattr(best_model, "feature_importances_"):
            self.feature_importances_ = best_model.feature_importances_
            self.feature_importances_ = np.abs(self.feature_importances_)/np.sum(np.abs(self.feature_importances_))
        elif hasattr(best_model, "coef_"):
            self.feature_importances_ = best_model.coef_
            self.feature_importances_ = np.abs(self.feature_importances_)/np.sum(np.abs(self.feature_importances_))
        else:
            self.feature_importances_ = None
        return self
    
    def predict(self, X):
        # Checks
        self._check_predict_params(X)

        check_is_fitted(self, ["best_estimators_", "weights_"])

        predictions = [
            self.best_estimators_[name].predict(X)
            for name in self.model_names_
        ]
        # Convert to numpy arrays 
        predictions = [
            pred if isinstance(pred, np.ndarray) else pred.values
            for pred in predictions
        ]
        # Check if the model is single or multi-output
        predictions = np.stack(predictions, axis=-1)

        weights = np.array([
            self.weights_[name]
            for name in self.model_names_
        ])

        adjusted_predictions = predictions @ weights

        if self.multi_output_:
            return pd.DataFrame(
                data = adjusted_predictions,
                index = X.index,
                columns = self.targets_
            )
        else:
            return adjusted_predictions
    
    def _compute_weights(self, cv_scores, temperature, min_weight):
        all_scores = np.array([cv_scores[name] for name in self.model_names_])

        if temperature == "max-min":
            spread = np.max(all_scores) - np.min(all_scores)
        elif temperature == "std":
            spread = np.std(all_scores)
        elif temperature == "mad":
            spread = np.median(np.abs(all_scores - np.median(all_scores)))
        elif temperature == "iqr":
            spread = np.percentile(all_scores, 75) - np.percentile(all_scores, 25)
        else:
            # temperature is a float
            spread = float(temperature)

        scaled_scores = all_scores / spread # TODO: deal with zero later
        scaled_scores = scaled_scores - np.max(scaled_scores)
        weights = np.exp(scaled_scores)
        weights = weights / weights.sum()

        if isinstance(min_weight, str):
            if min_weight == "mean":
                min_weight = np.mean(weights)
            elif min_weight == "median":
                min_weight = np.median(weights)
            elif min_weight == "lq":
                min_weight = np.percentile(weights, 25)
            elif min_weight == "uq":
                min_weight = np.percentile(weights, 75)
            elif min_weight == "lb":
                min_weight = np.percentile(weights, 25) - 1.5 * (np.percentile(weights, 75) - np.percentile(weights, 25))
            elif min_weight == "ub":
                min_weight = np.percentile(weights, 75) + 1.5 * (np.percentile(weights, 75) - np.percentile(weights, 25))
        
        if min_weight > 0:
            adjusted_weights = np.where(weights < min_weight, 0.0, weights)
            if adjusted_weights.sum() == 0:
                adjusted_weights = weights
            else:
                adjusted_weights = adjusted_weights / adjusted_weights.sum()

        return dict(zip(self.model_names_, adjusted_weights))
    
    def _check_init_params(
        self,
        estimators,
        scoring,
        cv,
        temperature,
        min_weight,
        error_score
    ):
        # estimators
        if not isinstance(estimators, list):
            raise TypeError("estimators must be a list of (name, estimator, param_grid) tuples")
        for item in estimators:
            if not isinstance(item, tuple):
                raise TypeError(
                    "Each item in estimators must be a tuple of (name, estimator, param_grid)"
                )
            if len(item) != 3:
                raise ValueError(
                    "There must be three elements in each tuple: (name, estimator, param_grid). "
                    "Check {}.".format(item)
                )
            name, estimator, param_grid = item
            if not isinstance(name, str):
                raise TypeError(
                    "The first element of each tuple should be a string name for the "
                    "estimator. Got {} instead.".format(type(name))
                )
            if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
                raise TypeError(
                    "The second element of each tuple should be a scikit-learn compatible "
                    "estimator with fit and predict methods. Got {} instead.".format(type(estimator))
                )
            if not isinstance(param_grid, dict):
                raise TypeError(
                    "The third element of each tuple should be a dictionary of "
                    "hyperparameters for GridSearchCV. Got {} instead.".format(type(param_grid))
                )
            # check param_grid keys are valid for the estimator
            for param in param_grid.keys():
                if not hasattr(estimator, param):
                    raise ValueError(
                        "The hyperparameter '{}' is not valid for the estimator '{}'. "
                        "Check the estimator's documentation.".format(param, name)
                    )
                
        # scoring
        if not callable(scoring):
            raise TypeError(
                "scoring must be a callable function compatible with "
                "scikit-learn"
            )
        for name, estimator, param_grid in estimators:
            try:
                check_scoring(estimator, scoring=scoring)
            except Exception as e:
                raise ValueError(
                    "The scoring function is not valid for the estimator '{}'. "
                    "Check the estimator's documentation or the scorer.".format(name)
                ) from e

        # cv 
        if not isinstance(cv, BaseCrossValidator):
            raise TypeError(
                "cv must be a scikit-learn compatible cross-validation splitter. "
                "Check {} inherits from BaseCrossValidator.".format(cv)
            )
        
        # temperature
        if not (
            isinstance(temperature, str) or isinstance(temperature, numbers.Number)
        ):
            raise TypeError(
                "temperature must be a string or a float. Got {} instead.".format(type(temperature))
            )
        if isinstance(temperature, str) and temperature not in ["max-min", "std", "mad", "iqr"]:
            raise ValueError(
                "temperature must be one of 'max-min', 'std', 'mad', 'iqr' or a float. "
                "Got {} instead.".format(temperature)
            )
        
        # min_weight
        if not isinstance(min_weight, (str, numbers.Number)):
            raise TypeError(
                "min_weight must be a float or a str. Got {} instead.".format(type(min_weight))
            )
        if isinstance(min_weight, numbers.Number) and min_weight < 0:
            raise ValueError(
                "min_weight must be a non-negative float. Got {} instead.".format(min_weight)
            )
        if isinstance(min_weight, str) and min_weight not in ["mean", "median", "lq", "uq", "lb", "ub"]:
            raise ValueError(
                "min_weight must be one of 'mean', 'median', 'lq', 'uq', 'lb', 'ub' or a non-negative float. "
                "Got {} instead.".format(min_weight)
            )
        
        # error_score can be "raise", np.inf, np.nan, or a float
        if error_score != "raise":
            if not isinstance(error_score, numbers.Number):
                raise TypeError(
                    "error_score must be 'raise', np.inf, np.nan, or a float. "
                    "Got {} instead.".format(type(error_score))
                )
        
        
    def _check_fit_params(self, X, y):
        # X
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                "X must be a pandas DataFrame or a numpy ndarray. "
                "Got {} instead.".format(type(X))
            )
        if not isinstance(X, np.ndarray):
            if not isinstance(X.index, pd.MultiIndex):
                raise TypeError(
                    "X must have a pandas MultiIndex with (cid, date) as levels. "
                    "Got {} instead.".format(type(X.index))
                )
        # y
        if not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError(
                "y must be a pandas Series, DataFrame, or a numpy ndarray. "
                "Got {} instead.".format(type(y))
            )
        if isinstance(y, np.ndarray) and y.ndim >= 2:
            raise ValueError(
                "If y is a numpy array, it must be 1-dimensional. Got an array with shape {} instead.".format(y.shape)
            )
        
        # check X and y index match
        if not isinstance(X, np.ndarray) and not isinstance(y, np.ndarray):
            if not X.index.equals(y.index):
                raise ValueError(
                    "The index of X and y must match. "
                    "Got X.index: {} and y.index: {}.".format(X.index, y.index)
                )
         
    def _check_predict_params(self, X):
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                "X must be a pandas DataFrame or a numpy ndarray. "
                "Got {} instead.".format(type(X))
            )
        if not isinstance(X, np.ndarray):
            if not isinstance(X.index, pd.MultiIndex):
                raise TypeError(
                    "X must have a pandas MultiIndex with (cid, date) as levels. "
                    "Got {} instead.".format(type(X.index))
                )
        
if __name__ == "__main__":
    import macrosynergy.management as msm
    from macrosynergy.management.simulate import make_qdf

    from sklearn.linear_model import Ridge, Lasso

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

    train = msm.categories_df(
        df=dfd, xcats=xcats, cids=cids, val="value", blacklist=black, freq="M", lag=1
    ).dropna()

    # Dataset
    X_train = train.iloc[:, :-2]
    y_train = train.iloc[:,-2:]

    model = ModelAveragingRegressor(
        estimators = [
            ("ridge1", Ridge(alpha = 1), {}),
            ("ridge10", Ridge(alpha = 10), {}),
            ("ridge100", Ridge(alpha = 100), {}),
            ("ridge1000", Ridge(alpha = 1000), {}),
            ("lasso1", Lasso(alpha = 1), {}),
            ("lasso.1", Lasso(alpha = 0.1), {}),
            ("lasso.01", Lasso(alpha = 0.01), {}),
            ("lasso.001", Lasso(alpha = 0.001), {}),
        ],
        scoring = make_scorer(r2_score, greater_is_better=True),
        cv = ExpandingKFoldPanelSplit(n_splits = 5),
        temperature = "mad",
        min_weight = "lq"
    ).fit(X_train, y_train)

    model.predict(X_train)

    print(model.weights_)
    print(model.feature_importances_)


    