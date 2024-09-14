import numpy as np
import pandas as pd
import scipy.stats as stats 

import datetime
import warnings

from sklearn.linear_model import Lars, lasso_path, lars_path, enet_path
from sklearn.exceptions import NotFittedError

from statsmodels.api import add_constant
from linearmodels.panel import RandomEffects

from base_panel_selector import BasePanelSelector
from typing import Union

class LarsSelector(BasePanelSelector):
    def __init__(self, n_factors = 10, fit_intercept = False):
        """
        Statistical feature selection using LARS.  

        Parameters
        ----------
        n_factors : int, default=10
            Number of factors to select.
        fit_intercept : bool, default=False
            Whether to fit an intercept term in the LARS model.
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

    def determine_features(self, X, y):
        """
        Create feature mask based on the LARS algorithm.

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series or pandas.DataFrame
            The target vector.
        """
        lars = Lars(fit_intercept = self.fit_intercept, n_nonzero_coefs = self.n_factors)
        lars.fit(X.values, y.values.reshape(-1, 1))
        coefs = lars.coef_

        return [True if coef != 0 else False for coef in coefs]
    
class LassoSelector(BasePanelSelector):
    def __init__(self, n_factors = 10, positive = False, use_lars = False):
        """
        Statistical feature selection with the LASSO. 

        Parameters
        ----------
        n_factors : int
            Number of factors to select.
        positive : bool
            Whether to constrain the LASSO coefficients to be positive.
        use_lars : bool
            Whether to use the LARS algorithm to determine the coefficient paths for the LASSO.
        """
        # Checks
        if not isinstance(n_factors, int):
            raise TypeError("'n_factors' must be an integer.")
        if n_factors <= 0:
            raise ValueError("'n_factors' must be a positive integer.")
        if not isinstance(positive, bool):
            raise TypeError("'positive' must be a boolean.")
        if not isinstance(use_lars, bool):
            raise TypeError("'use_lars' must be a boolean.")
        
        # Attributes
        self.n_factors = n_factors
        self.positive = positive
        self.use_lars = use_lars

    def determine_features(self, X, y):
        """
        Create feature mask based on the LASSO algorithm.

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series or pandas.DataFrame
            The target vector.
        """
        # Obtain coefficient paths with dimensions (n_features, n_alphas)
        if self.use_lars:
            _, _, coefs_path = lars_path(X.values, y.values.reshape(-1, 1), positive = self.positive, method='lasso')
        else:
            _, _, coefs_path = lasso_path(X.values, y.values.reshape(-1, 1), positive = self.positive)

        mask = (coefs_path[:,self.n_factors] != 0)

        return mask
    
class ENetSelector(BasePanelSelector):
    def __init__(self, n_factors = 10, positive = False, l1_ratio = 0.5):
        """
        Statistical feature selection with Elastic Net. 

        Parameters
        ----------
        n_factors : int
            Number of factors to select.
        positive : bool
            Whether to constrain the Elastic Net coefficients to be positive.
        l1_ratio : float
            The Elastic Net mixing parameter.
        """
        # Checks
        if not isinstance(n_factors, int):
            raise TypeError("'n_factors' must be an integer.")
        if n_factors <= 0:
            raise ValueError("'n_factors' must be a positive integer.")
        if not isinstance(positive, bool):
            raise TypeError("'positive' must be a boolean.")
        if not isinstance(l1_ratio, (int, float)):
            raise TypeError("'l1_ratio' must be a float.")
        if (l1_ratio <= 0) or (l1_ratio > 1):
            raise ValueError("'l1_ratio' must be in the range (0, 1].")
        
        # Attributes
        self.n_factors = n_factors
        self.positive = positive
        self.l1_ratio = l1_ratio

    def determine_features(self, X, y):
        """
        Create feature mask based on the Elastic Net algorithm.

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        y : pandas.Series or pandas.DataFrame
            The target vector.
        """
        _, _, coefs_path = enet_path(X.values, y.values.reshape(-1, 1), positive = self.positive, l1_ratio = self.l1_ratio)

        mask = (coefs_path[:,self.n_factors] != 0)

        return mask
    
class MapSelector(BasePanelSelector):
    def __init__(self, significance_level = 0.05, positive = False):
        """
        Univariate statistical feature selection using the Macrosynergy panel test.

        Parameters
        ----------
        significance_level : float, default=0.05
            Significance level.
        positive : bool, default=False
            Whether to only keep features with positive estimated model coefficients.
        """
        if type(significance_level) != float:
            raise TypeError("The significance_level must be a float.")
        if (significance_level <= 0) or (significance_level >= 1):
            raise ValueError(
                "The significance_level must be in between 0 and 1."
            )
        if not isinstance(positive, (bool, np.bool_)):
            raise TypeError("The 'positive' parameter must be a boolean.")

        self.significance_level = significance_level
        self.positive = positive

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
        # Convert cross-sections to numeric codes for compatibility with RandomEffects
        unique_xss = sorted(X.index.get_level_values(0).unique())
        xs_codes = dict(zip(unique_xss, range(1, len(unique_xss) + 1)))

        X = X.rename(xs_codes, level=0, inplace=False).copy()
        y = y.rename(xs_codes, level=0, inplace=False).copy()

        # Iterate through each feature and perform the panel test
        mask = []
        for col in self.feature_names_in_:
            ftr = X[col]
            ftr = add_constant(ftr)
            # Swap levels so that random effects are placed on each time period,
            # as opposed to the cross-section
            re = RandomEffects(y.swaplevel(), ftr.swaplevel()).fit()
            est = re.params[col]
            zstat = est / re.std_errors[col]
            pval = 2 * (1 - stats.norm.cdf(zstat))
            if pval < self.significance_level:
                if self.positive:
                    mask.append(est > 0)
                else:
                    mask.append(True)
            else:
                mask.append(False)

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
    train = train[train.index.get_level_values(1) >= pd.Timestamp(year=2005,month=8,day=1)]

    X_train = train.drop(columns=["XR"])
    y_train = train["XR"]
    y_long_train = pd.melt(
        frame=y_train.reset_index(), id_vars=["cid", "real_date"], var_name="xcat"
    )

    # LARS selector
    lars = LarsSelector(n_factors = 5).fit(X_train, y_train)
    print(f"LARS 5-factors, no intercept: {lars.get_feature_names_out()}")
    lars = LarsSelector(n_factors = 5, fit_intercept = True).fit(X_train, y_train)
    print(f"LARS 5-factors, with intercept: {lars.get_feature_names_out()}")