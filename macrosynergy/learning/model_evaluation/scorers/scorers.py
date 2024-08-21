import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.base import RegressorMixin

def neg_mean_abs_corr(
    estimator,
    X_test,
    y_test,
    correlation_type = "pearson",
):
    """
    Negative mean absolute correlation between a time series of benchmark returns and a
    panel of computed hedged returns, with average taken over all cross-sections.

    Parameters
    ----------
    estimator : BaseRegressionSystem
        A fitted `scikit-learn` regression object with separate linear models for each 
        cross-section of returns, regressed against a time series of benchmark risk basket
        returns. It is expected to possess a `coefs_` dictionary attribute with keys
        corresponding to the cross-sections of returns and values corresponding to the
        estimated coefficients of the linear model for each cross-section.
    X_test : pd.DataFrame
        Risk-basket returns replicated for each cross-section of returns in `y_test`. 
    y_test : pd.Series 
        Panel of financial contract returns.
    correlation_type : str
        Type of correlation to compute between each hedged return
        series and the risk basket return series. Default is "pearson".
        Alternatives are "spearman" and "kendall". 

    Returns
    -------
    neg_mean_abs_corr : float
        Negative mean absolute correlation between benchmark risk basket returns and
        computed hedged returns.

    Notes
    -----
    For each cross-section :math:`c` in `X_test`, hedged returns are calculated by 
    subtracting :math:`X_{test, c} \\cdot \\text{coefs_}[c]` from each `y_{test, c}`.
    Following this, the negative mean absolute correlation over cross-sections can be
    calculated:

    ```{math}
    :label: neg_mean_abs_corr
    \\text{neg_mean_abs_corr} = - (1/C)\\sum_{c=1}^{C} \\left [ abs_corr_{c} \\right ]
    ```

    This function is a specialised scorer to evaluate the quality of a hedge within the
    `BetaEstimator` class in the `macrosynergy.learning` subpackage.
    """
    # Obtain key information 
    market_returns = X_test.iloc[:,0].copy() 
    contract_returns = y_test.copy()
    unique_cross_sections = X_test.index.get_level_values(0).unique()
    estimated_coefs = estimator.coefs_

    running_sum = 0
    xs_count = 0
    for cross_section in unique_cross_sections:
        # Check whether a model for this cross-section has been estimated
        if cross_section in estimated_coefs.keys():
            xs_count += 1
            # Get cross-section returns and matched risk basket returns
            contract_returns_c = contract_returns.xs(cross_section)
            market_returns_c = market_returns.xs(
                cross_section
            )
            hedged_returns_c = (
                contract_returns_c - estimated_coefs[cross_section] * market_returns_c
            )
            # Compute negative absolute market correlation
            if correlation_type == "pearson":
                abs_corr = abs(stats.pearsonr(hedged_returns_c, market_returns_c)[0])
            elif correlation_type == "spearman":
                abs_corr = abs(stats.spearmanr(hedged_returns_c, market_returns_c)[0])
            else:
                # Use Kendall
                abs_corr = abs(stats.kendalltau(hedged_returns_c, market_returns_c)[0])
            # Update running sum 
            running_sum += abs_corr
        else:
            # Then a model wasn't estimated for this cross-section 
            continue

    if xs_count == 0:
        return np.nan 
    else:
        return - running_sum / xs_count