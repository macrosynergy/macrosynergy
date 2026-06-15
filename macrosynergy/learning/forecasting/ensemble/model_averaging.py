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

        return self
    
    def predict(self, X):
        # Checks
        self._check_predict_params(X)

        check_is_fitted(self, ["best_estimators_", "weights_"])

        predictions = np.column_stack([
            self.best_estimators_[name].predict(X)
            for name in self.model_names_
        ])

        weights = np.array([
            self.weights_[name]
            for name in self.model_names_
        ])

        return predictions @ weights
    
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

        if min_weight > 0:
            weights = np.maximum(weights, min_weight)
            weights = weights / weights.sum()

        return dict(zip(self.model_names_, weights))
    
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
            if not isinstance(estimator, RegressorMixin):
                raise TypeError(
                    "The second element of each tuple should be a scikit-learn compatible "
                    "regressor, inheriting from RegressorMixin. Check {}.".format(estimator)
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
        if not isinstance(min_weight, numbers.Number):
            raise TypeError(
                "min_weight must be a float. Got {} instead.".format(type(min_weight))
            )
        if min_weight < 0:
            raise ValueError(
                "min_weight must be a non-negative float. Got {} instead.".format(min_weight)
            )
        
        # error_score can be "raise", np.inf, np.nan, or a float
        if error_score != "raise":
            if not isinstance(error_score, numbers.Number):
                raise TypeError(
                    "error_score must be 'raise', np.inf, np.nan, or a float. "
                    "Got {} instead.".format(type(error_score))
                )
        
        
    def _check_fit_params(self, X, y):
        pass
         
    def _check_predict_params(self, X):
        pass
        
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
    X_train = train.drop(columns=["XR"])
    y_train = train["XR"]

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
        min_weight = 0.0
    ).fit(X_train, y_train)

    print(model.weights_)


    