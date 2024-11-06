import numpy as np
import pandas as pd

from macrosynergy.learning.forecasting.linear_model.lad_regressors import LADRegressor
from macrosynergy.learning.forecasting.weighted_regressors import (
    SignWeightedRegressor,
    TimeWeightedRegressor,
)


class SignWeightedLADRegressor(SignWeightedRegressor):
    def __init__(
        self,
        fit_intercept: bool = True,
        positive=False,
        alpha=0,
        shrinkage_type="l1",
        tol=None,
        maxiter=None,
    ):
        """
        LAD regressor with sign-weighted loss.

        Parameters
        ----------
        fit_intercept : bool, default=True
            Whether or not to add an intercept to the model.
        positive: bool, default=False
            Whether or not to enforce non-negativity of model weights,
            with exception to a model intercept.
        alpha: float, default=0
            Shrinkage hyperparameter.
        shrinkage_type: str, default="l1"
            Type of shrinkage regularization to perform.
        tol: float, default=None
            Tolerance for convergence of the learning algorithm (SLSQP).
            This is passed into the 'tol' parameter of the scipy.optimize.minimize
            function.
        maxiter: int, default=None
            Maximum number of iterations for the learning algorithm (SLSQP).
            This is passed into the 'maxiter' key of the options dictionary in the
            scipy.optimize.minimize function.


        Notes
        -----
        A dependent variable is modelled as a linear combination of the input features.
        The weights associated with each feature (and the intercept) are determined by
        finding the weights that minimise the weighted average absolute model residuals,
        where the weighted average is based on inverse frequency of the sign of the
        dependent variable.

        By weighting the contribution of different training samples based on the
        sign of the label, the model is encouraged to learn equally from both positive and
        negative return samples, irrespective of class imbalance. If there are more
        positive targets than negative targets in the training set, then the negative
        target samples are given a higher weight in the model training process.
        The opposite is true if there are more negative targets than positive targets.
        """
        super().__init__(
            model=LADRegressor(
                fit_intercept=fit_intercept,
                positive=positive,
                alpha=alpha,
                shrinkage_type=shrinkage_type,
                tol=tol,
                maxiter=maxiter,
            ),
        )
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.alpha = alpha
        self.shrinkage_type = shrinkage_type
        self.tol = tol
        self.maxiter = maxiter

    def set_params(self, **params):
        """
        Setter method to update the parameters of the `SignWeightedLADRegressor`.

        Parameters
        ----------
        **params : dict
            Dictionary of parameters to update.

        Returns
        -------
        self
            The `SignWeightedLADRegressor` instance with updated parameters.
        """
        super().set_params(**params)

        relevant_params = {"fit_intercept", "positive", "alpha", "shrinkage_type"}

        if relevant_params.intersection(params):
            self.model = LADRegressor(
                fit_intercept=self.fit_intercept,
                positive=self.positive,
                alpha=self.alpha,
                shrinkage_type=self.shrinkage_type,
            )

        return self


class TimeWeightedLADRegressor(TimeWeightedRegressor):
    def __init__(
        self,
        fit_intercept=True,
        positive=False,
        half_life=21 * 12,
        alpha=0,
        shrinkage_type="l1",
        tol=None,
        maxiter=None,
    ):
        """
        LAD regressor with time-weighted loss.

        Parameters
        ----------
        fit_intercept : bool, default=True
            Whether or not to add an intercept to the model.
        positive: bool, default=False
            Whether or not to enforce non-negativity of model weights,
            with exception to a model intercept.
        half_life: int, default=21 * 12
            Half-life of the exponential decay function used to weight the
            contribution of different training samples based on the time of
            the sample.
        alpha: float, default=0
            Shrinkage hyperparameter.
        shrinkage_type: str, default="l1"
            Type of shrinkage regularization to perform.
        tol: float, default=None
            Tolerance for convergence of the learning algorithm (SLSQP).
            This is passed into the 'tol' parameter of the scipy.optimize.minimize
            function.
        maxiter: int, default=None
            Maximum number of iterations for the learning algorithm (SLSQP).
            This is passed into the 'maxiter' key of the options dictionary in the
            scipy.optimize.minimize function.

        Notes
        -----
        A dependent variable is modelled as a linear combination of the input features.
        The weights associated with each feature (and the intercept) are determined by
        finding the weights that minimise the weighted average absolute model residuals,
        where the weighted average is based on an exponential decay specified by the
        half-life parameter, where more recent samples are given higher weight.

        By weighting the contribution of different training samples based on the
        timestamp, the model is encouraged to prioritise more recent samples in the
        model training process. The half-life denotes the number of time periods in units
        of the native dataset frequency for the weight attributed to the most recent sample
        (one) to decay by half.
        """
        super().__init__(
            model=LADRegressor(
                fit_intercept=fit_intercept,
                positive=positive,
                alpha=alpha,
                shrinkage_type=shrinkage_type,
                tol=tol,
                maxiter=maxiter,
            ),
            half_life=half_life,
        )
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.half_life = half_life
        self.alpha = alpha
        self.shrinkage_type = shrinkage_type
        self.tol = tol
        self.maxiter = maxiter

    def set_params(self, **params):
        """
        Setter method to update the parameters of the `TimeWeightedLADRegressor`.

        Parameters
        ----------
        **params : dict
            Dictionary of parameters to update.

        Returns
        -------
        self
            The `TimeWeightedLADRegressor` instance with updated parameters.
        """
        super().set_params(**params)

        relevant_params = {"fit_intercept", "positive", "alpha", "shrinkage_type"}

        if relevant_params.intersection(params):
            self.model = LADRegressor(
                fit_intercept=self.fit_intercept,
                positive=self.positive,
                alpha=self.alpha,
                shrinkage_type=self.shrinkage_type,
            )

        return self


if __name__ == "__main__":
    import macrosynergy.management as msm
    from macrosynergy.management.simulate import make_qdf

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

    X_train = train.drop(columns=["XR"])
    y_train = train["XR"]

    # Fit SWLAD - no regularization
    model = SignWeightedLADRegressor(
        fit_intercept=True, positive=False, alpha=0, shrinkage_type="l1"
    )
    model.fit(X_train, y_train)
    print(f"Intercept: {model.intercept_}, Coefficients: {model.coef_}")

    # Fit SWLAD - regularization
    model = SignWeightedLADRegressor(
        fit_intercept=True, positive=False, alpha=1, shrinkage_type="l1"
    )
    model.fit(X_train, y_train)
    print(f"Intercept: {model.intercept_}, Coefficients: {model.coef_}")

    # Fit TWLAD - no regularization
    model = TimeWeightedLADRegressor(
        fit_intercept=True, positive=False, half_life=36, alpha=0, shrinkage_type="l1"
    )
    model.fit(X_train, y_train)
    print(f"Intercept: {model.intercept_}, Coefficients: {model.coef_}")

    # Fit TWLAD - regularization
    model = TimeWeightedLADRegressor(
        fit_intercept=True, positive=False, half_life=36, alpha=1, shrinkage_type="l1"
    )
    model.fit(X_train, y_train)
    print(f"Intercept: {model.intercept_}, Coefficients: {model.coef_}")
