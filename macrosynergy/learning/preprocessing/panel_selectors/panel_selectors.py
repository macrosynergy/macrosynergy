import numbers

import numpy as np
import pandas as pd
from sklearn.linear_model import Lars, lars_path

from macrosynergy.learning.preprocessing.panel_selectors.base_panel_selector import (
    BasePanelSelector,
)
from macrosynergy.learning.random_effects import RandomEffects


class LarsSelector(BasePanelSelector):
    def __init__(self, n_factors=10, fit_intercept=False):
        """
        Statistical feature selection using LARS.

        Parameters
        ----------
        n_factors : int, default=10
            Number of factors to select.
        fit_intercept : bool, default=False
            Whether to fit an intercept term in the LARS model.

        Notes
        -----
        The Least Angle Regression (LARS) algorithm was designed to fit high dimensional
        linear models. It is a means of estimating the covariates to include in the model,
        as well as associated coefficients. LARS can be considered to be a continuous
        equivalent to forward selection.

        The algorithm is described in detail in [1]_ and is implemented in the
        `scikit-learn` library [2]_. It works as follows:

        1. Set coefficients to zero.
        2. Find the covariate that has the highest correlation with the target variable.
        3. Increase the coefficient of this covariate in a stepwise fashion, recording
            the residual at each step. Stop when another covariate is as correlated with
            the residuals as the current one.
        4. Add this second covariate to the model and the compute the two-variable OLS
            solution.
        5. Increase the coefficients of the two covariates in a stepwise fashion towards
            the OLS solution, recording the residuals at each step. Stop when another
            covariate is as correlated with the residuals as the current ones.
        6. Add this third covariate to the model and compute the three-variable OLS
            solution.
        7. Iterate this process until the desired number of covariates have been selected.

        References
        ----------
        .. [1] Efron, B., Hastie, T., Johnstone, I. and Tibshirani, R., 2004.
            Least angle regression.
            https://arxiv.org/abs/math/0406456
        .. [2] https://scikit-learn.org/dev/modules/linear_model.html#least-angle-regression

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

        super().__init__()

    def determine_features(self, X, y):
        """
        Create feature mask based on the LARS algorithm.

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series or pandas.DataFrame
            The target vector.

        Returns
        -------
        mask : list
            Boolean mask of selected features.
        """
        lars = Lars(fit_intercept=self.fit_intercept, n_nonzero_coefs=self.n_factors)
        lars.fit(X.values, y.values.reshape(-1, 1))
        coefs = lars.coef_

        return [True if coef != 0 else False for coef in coefs]


class LassoSelector(BasePanelSelector):
    def __init__(self, n_factors=10, positive=False):
        """
        Statistical feature selection with LASSO-LARS.

        Parameters
        ----------
        n_factors : int
            Number of factors to select.
        positive : bool
            Whether to constrain the LASSO coefficients to be positive.

        Notes
        -----
        The Least Absolute Shrinkage and Selection Operator (LASSO) [1]_ is a linear model
        that estimates sparse coefficients. This means that some encouragement is given
        for the model to set some coefficients to zero. Hence, the LASSO can be said to
        perform feature selection. It transpires that the LARS algorithm [2]_
        (see `LarsSelector`) can be used to track the LASSO coefficients as the user-defined
        sparsity level is increased. Consequently, we use the LARS algorithm to
        compute the LASSO paths and select the desired number of factors. See [3]_ for
        the `scikit-learn` documentation on the LASSO-LARS model fit.

        References
        ----------
        .. [1] Tibshirani, R., 1996. Regression shrinkage and selection via the lasso.
            Journal of the Royal Statistical Society Series B: Statistical Methodology,
            58(1), pp.267-288.
            https://www.jstor.org/stable/2346178
        .. [2] Efron, B., Hastie, T., Johnstone, I. and Tibshirani, R., 2004.
            Least angle regression.
            https://arxiv.org/abs/math/0406456
        .. [3] https://scikit-learn.org/dev/modules/generated/sklearn.linear_model.LassoLars.html
        """
        # Checks
        if not isinstance(n_factors, int):
            raise TypeError("'n_factors' must be an integer.")
        if n_factors <= 0:
            raise ValueError("'n_factors' must be a positive integer.")
        if not isinstance(positive, bool):
            raise TypeError("'positive' must be a boolean.")

        # Attributes
        self.n_factors = n_factors
        self.positive = positive

        super().__init__()

    def determine_features(self, X, y):
        """
        Create feature mask based on the LASSO-LARS algorithm.

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series or pandas.DataFrame
            The target vector.

        Returns
        -------
        mask : np.ndarray
            Boolean mask of selected features.
        """
        # Obtain coefficient paths with dimensions (n_features, n_alphas)
        _, _, coefs_path = lars_path(
            X.values,
            y.values,
            positive=self.positive,
            method="lasso",
        )

        mask = coefs_path[:, min(self.n_factors, coefs_path.shape[1] - 1)] != 0

        return mask


class MapSelector(BasePanelSelector):
    def __init__(self, n_factors=None, significance_level=0.05, positive=False):
        """
        Univariate statistical feature selection using the Macrosynergy panel test.

        Parameters
        ----------
        n_factors : int, optional
            Number of factors to select.
        significance_level : float, default=0.05
            Significance level.
        positive : bool, default=False
            Whether to only keep features with positive estimated model coefficients.

        Notes
        -----
        The Macrosynergy panel test [1]_ is a univariate test that estimates the
        significance of a relationship between each feature and the target variable, over
        a panel. This test accounts for cross-sectional correlations. Often, different
        cross-sections in a panel are highly correlated - particularly in the case of
        dependent variable return data. This violates the assumption of independence
        in the usual z-test or t-test, from which the usual p-values are derived. As a
        consequence, probabilities of significance can be overstated.

        In the Macrosynergy panel test, a Wald test is used to compare the null hypothesis
        of an intercept + period-specific random effects model against the alternative
        hypothesis of an intercept + period-specific random effects model + the feature
        of interest. This works because the null-alternative hypotheses are nested models.
        The model in the null hypothesis accounts for the cross-sectional correlations
        that exist in each time period. Rejecting this model in favour of the alternative
        model indicates that the feature of interest is significant, accounting for those
        cross-sectional correlations.

        References
        ----------
        .. [1] Gholkar, Rushil and Sueppel, Ralph, 2023.
            Testing macro trading factors.
            https://macrosynergy.com/research/testing-macro-trading-factors/
        """
        # Checks
        if n_factors is not None:
            if not isinstance(n_factors, int):
                raise TypeError("The 'n_factors' parameter must be an integer.")
            if n_factors <= 0:
                raise ValueError(
                    "The 'n_factors' parameter must be a positive integer."
                )
        if not isinstance(significance_level, numbers.Number):
            raise TypeError("The significance_level must be a float.")
        if (significance_level < 0) or (significance_level > 1):
            raise ValueError("The significance_level must be in between 0 and 1.")
        if not isinstance(positive, (bool, np.bool_)):
            raise TypeError("The 'positive' parameter must be a boolean.")

        self.significance_level = significance_level
        self.positive = positive
        self.n_factors = n_factors

    def determine_features(self, X, y):
        """
        Create feature mask based on the Macrosynergy panel test.

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series or pandas.DataFrame
            The target vector.

        Returns
        -------
        mask : np.ndarray
            Boolean mask of selected features.
        """
        # Iterate through each feature and perform the panel test
        factor_pvals = []

        for col in self.feature_names_in_:
            ftr = X[col]
            re = RandomEffects(fit_intercept=True).fit(ftr, y)
            est = re.params[col]
            pval = re.pvals[col]
            factor_pvals.append(pval)

        if self.n_factors is not None:
            # Return a mask of factors with `n_factors` smallest p_values
            factor_indexes = np.argsort(factor_pvals)[: self.n_factors]
            mask = [
                True if idx in factor_indexes else False
                for idx in range(len(factor_pvals))
            ]

        else:
            if self.positive:
                # Return a mask of factors with positive estimated coefficients and
                # p_values < significance_level
                mask = [
                    True if ((est > 0) and (pval < self.significance_level)) else False
                    for est in factor_pvals
                ]
            else:
                # Return as mask of factors with p_values < significance_level
                mask = [
                    True if pval < self.significance_level else False
                    for pval in factor_pvals
                ]

        return np.array(mask)


if __name__ == "__main__":
    import macrosynergy.management as msm
    from macrosynergy.management.simulate import make_qdf

    # Randomly generate a panel
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
    train = train[
        train.index.get_level_values(1) >= pd.Timestamp(year=2005, month=8, day=1)
    ]

    X_train = train.drop(columns=["XR"])
    y_train = train["XR"]

    # LARS selector
    lars = LarsSelector(n_factors=2).fit(X_train, y_train)
    print(f"LARS 2-factors, no intercept: {lars.get_feature_names_out()}")
    lars = LarsSelector(n_factors=2, fit_intercept=True).fit(X_train, y_train)
    print(f"LARS 2-factors, with intercept: {lars.get_feature_names_out()}")

    print(lars.transform(X_train))

    # LASSO selector
    lasso = LassoSelector(n_factors=1, positive=True).fit(X_train, y_train)
    print(f"Lasso 1-factor, positive restriction: {lasso.get_feature_names_out()}")
    lasso = LassoSelector(n_factors=3, positive=False).fit(X_train, y_train)
    print(f"Lasso 3-factors, with intercept: {lasso.get_feature_names_out()}")

    print(lasso.transform(X_train))

    # Map selector
    map_selector = MapSelector(n_factors=2).fit(X_train, y_train)
    print(f"Map 2-factors: {map_selector.get_feature_names_out()}")
    map_selector = MapSelector(significance_level=0.2).fit(X_train, y_train)
    print(f"Map significance 0.2: {map_selector.get_feature_names_out()}")

    print(map_selector.transform(X_train))
