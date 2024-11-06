import numbers
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from macrosynergy.learning.forecasting.weighted_regressors import (
    SignWeightedRegressor,
    TimeWeightedRegressor,
)


class SignWeightedLinearRegression(SignWeightedRegressor):
    def __init__(
        self,
        fit_intercept=True,
        positive=False,
        alpha=0,
        shrinkage_type="l1",
    ):
        """
        Linear regression with sign-weighted L2 loss.

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


        Notes
        -----
        A dependent variable is modelled as a linear combination of the input features.
        The weights associated with each feature (and the intercept) are determined by
        finding the weights that minimise the weighted average squared model residuals,
        where the weighted average is based on inverse frequency of the sign of the
        dependent variable.

        By weighting the contribution of different training samples based on the
        sign of the label, the model is encouraged to learn equally from both positive and
        negative return samples, irrespective of class imbalance. If there are more
        positive targets than negative targets in the training set, then the negative
        target samples are given a higher weight in the model training process.
        The opposite is true if there are more negative targets than positive targets.
        """
        # Checks
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean.")
        if not isinstance(alpha, numbers.Number) or isinstance(alpha, bool):
            raise TypeError("alpha must be a number.")
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if not isinstance(shrinkage_type, str):
            raise TypeError("shrinkage_type must be a string.")
        if shrinkage_type not in ["l1", "l2"]:
            raise ValueError("Invalid shrinkage_type. Must be 'l1' or 'l2'.")

        if alpha == 0:
            super().__init__(
                model=LinearRegression(fit_intercept=fit_intercept, positive=positive),
            )
        elif shrinkage_type == "l1":
            super().__init__(
                model=Lasso(
                    fit_intercept=fit_intercept, positive=positive, alpha=alpha
                ),
            )
        elif shrinkage_type == "l2":
            super().__init__(
                model=Ridge(
                    fit_intercept=fit_intercept, positive=positive, alpha=alpha
                ),
            )

        self.fit_intercept = fit_intercept
        self.positive = positive
        self.alpha = alpha
        self.shrinkage_type = shrinkage_type

    def set_params(self, **params):
        """
        Setter method to update the parameters of the `SignWeightedLinearRegression`.

        Parameters
        ----------
        **params : dict
            Dictionary of parameters to update.

        Returns
        -------
        self
            The `SignWeightedLinearRegression` instance with updated parameters.
        """
        super().set_params(**params)

        relevant_params = {"fit_intercept", "positive", "alpha", "shrinkage_type"}

        if relevant_params.intersection(params):
            if self.alpha == 0:
                self.model = LinearRegression(
                    fit_intercept=self.fit_intercept,
                    positive=self.positive,
                )
            elif self.shrinkage_type == "l1":
                self.model = Lasso(
                    fit_intercept=self.fit_intercept,
                    positive=self.positive,
                    alpha=self.alpha,
                )
            elif self.shrinkage_type == "l2":
                self.model = Ridge(
                    fit_intercept=self.fit_intercept,
                    positive=self.positive,
                    alpha=self.alpha,
                )

        return self


class TimeWeightedLinearRegression(TimeWeightedRegressor):
    def __init__(
        self,
        fit_intercept=True,
        positive=False,
        half_life=21 * 12,
        alpha=0,
        shrinkage_type="l1",
    ):
        """
        Linear regression with time-weighted L2 loss.

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

        Notes
        -----
        A dependent variable is modelled as a linear combination of the input features.
        The weights associated with each feature (and the intercept) are determined by
        finding the weights that minimise the weighted average squared model residuals,
        where the weighted average is based on an exponential decay specified by the
        half-life parameter, where more recent samples are given higher weight.

        By weighting the contribution of different training samples based on the
        timestamp, the model is encouraged to prioritise more recent samples in the
        model training process. The half-life denotes the number of time periods in units
        of the native dataset frequency for the weight attributed to the most recent sample
        (one) to decay by half.
        """
        # Checks
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean.")
        if not isinstance(positive, bool):
            raise TypeError("positive must be a boolean.")
        if not isinstance(alpha, numbers.Number) or isinstance(alpha, bool):
            raise TypeError("alpha must be a number.")
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if not isinstance(shrinkage_type, str):
            raise TypeError("shrinkage_type must be a string.")
        if shrinkage_type not in ["l1", "l2"]:
            raise ValueError("Invalid shrinkage_type. Must be 'l1' or 'l2'.")
        if not isinstance(half_life, numbers.Number) or isinstance(half_life, bool):
            raise TypeError("half_life must be a number.")
        if half_life <= 0:
            raise ValueError("half_life must be positive.")

        if alpha == 0:
            model = LinearRegression(fit_intercept=fit_intercept, positive=positive)
        elif shrinkage_type == "l1":
            model = Lasso(fit_intercept=fit_intercept, positive=positive, alpha=alpha)
        elif shrinkage_type == "l2":
            model = Ridge(fit_intercept=fit_intercept, positive=positive, alpha=alpha)

        super().__init__(model=model, half_life=half_life)
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.alpha = alpha
        self.shrinkage_type = shrinkage_type

    def set_params(self, **params):
        """
        Setter method to update the parameters of the `TimeWeightedLinearRegression`.

        Parameters
        ----------
        **params : dict
            Dictionary of parameters to update.

        Returns
        -------
        self
            The `TimeWeightedLinearRegression` instance with updated parameters.
        """
        super().set_params(**params)

        relevant_params = {"fit_intercept", "positive", "alpha", "shrinkage_type"}

        if relevant_params.intersection(params):
            if self.alpha == 0:
                self.model = LinearRegression(
                    fit_intercept=self.fit_intercept,
                    positive=self.positive,
                )
            elif self.shrinkage_type == "l1":
                self.model = Lasso(
                    fit_intercept=self.fit_intercept,
                    positive=self.positive,
                    alpha=self.alpha,
                )
            elif self.shrinkage_type == "l2":
                self.model = Ridge(
                    fit_intercept=self.fit_intercept,
                    positive=self.positive,
                    alpha=self.alpha,
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

    # Fit SWLS - no regularization
    model = SignWeightedLinearRegression(
        fit_intercept=True, positive=False, alpha=0, shrinkage_type="l1"
    )
    model.fit(X_train, y_train)
    print(f"Intercept: {model.intercept_}, Coefficients: {model.coef_}")

    # Fit SWLS - regularization
    model = SignWeightedLinearRegression(
        fit_intercept=True, positive=False, alpha=0.01, shrinkage_type="l1"
    )
    model.fit(X_train, y_train)
    print(f"Intercept: {model.intercept_}, Coefficients: {model.coef_}")

    # Fit TWLS - no regularization
    model = TimeWeightedLinearRegression(
        fit_intercept=True, positive=False, half_life=36, alpha=0, shrinkage_type="l1"
    )
    model.fit(X_train, y_train)
    print(f"Intercept: {model.intercept_}, Coefficients: {model.coef_}")

    # Fit TWLS - regularization
    model = TimeWeightedLinearRegression(
        fit_intercept=True,
        positive=False,
        half_life=36,
        alpha=0.01,
        shrinkage_type="l1",
    )
    model.fit(X_train, y_train)
    print(f"Intercept: {model.intercept_}, Coefficients: {model.coef_}")
