import numpy as np
import pandas as pd
import scipy.stats as stats 

import datetime
import warnings

from sklearn.linear_model import Lars, lasso_path, lars_path, enet_path
from sklearn.exceptions import NotFittedError

from statsmodels.api import add_constant
from linearmodels.panel import RandomEffects

from macrosynergy.learning.preprocessing.selectors import BasePanelSelector
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
        if self.use_lars:
            _, _, coefs_path = lars_path(X.values, y.values.reshape(-1, 1), positive = self.positive, method='lasso') # Feature x alphas
        else:
            _, _, coefs_path = lasso_path(X.values, y.values.reshape(-1, 1), positive = self.positive)

        mask = (coefs_path[:,self.n_factors] != 0)

        return mask
    
class ENetSelector(BasePanelSelector):
    def __init__(self, n_factors = 10, positive = False):
        """
        Statistical feature selection with Elastic Net. 

        Parameters
        ----------
        n_factors : int
            Number of factors to select.
        positive : bool
            Whether to constrain the Elastic Net coefficients to be positive.
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
        _, _, coefs_path = enet_path(X.values, y.values.reshape(-1, 1), positive = self.positive)

        mask = (coefs_path[:,self.n_factors] != 0)

        return mask
    
class MapSelector(BasePanelSelector):
    def __init__(self, significance_level = 0.05, positive = False):
        """
        Univariate statistical feature selection using the Macrosynergy panel test.

        Parameters
        ----------
        significance_level : float, default=0.05
            Significance significance_level.
        positive : bool, default=False
            Whether to only keep features with positive estimated model coefficients.
        """
        if type(significance_level) != float:
            raise TypeError("The significance_level must be a float.")
        if (significance_level <= 0) or (significance_level > 1):
            raise ValueError(
                "The significance_level must be in between 0 (inclusive) and 1 (exclusive)."
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