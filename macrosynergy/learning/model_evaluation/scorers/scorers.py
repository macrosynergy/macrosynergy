from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.ensemble import VotingRegressor

from macrosynergy.learning.forecasting.model_systems import BaseRegressionSystem


def neg_mean_abs_corr(
    estimator,
    X_test,
    y_test,
    correlation_type="pearson",
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
    # Checks
    # estimator
    if isinstance(estimator, BaseRegressionSystem):
        if estimator.models_ is None:
            raise ValueError("estimator must be a fitted model.")
    elif isinstance(estimator, VotingRegressor):
        if not all(
            isinstance(est, BaseRegressionSystem) for est in estimator.estimators_
        ):
            raise TypeError(
                "estimator must be a VotingRegressor with BaseRegressionSystem estimators."
            )
        if not all(est.models_ is not None for est in estimator.estimators_):
            raise ValueError("estimator must be a VotingRegressor with fitted models.")
    else:
        raise TypeError(
            "estimator must be a BaseRegressionSystem or VotingRegressor object."
        )

    # X_test
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if X_test.ndim != 2:
        raise ValueError("X_test must be a 2-dimensional DataFrame.")
    if X_test.shape[1] != 1:
        raise ValueError("X_test must have only one column.")
    if not isinstance(X_test.index, pd.MultiIndex):
        raise ValueError("X_test must be multi-indexed.")
    if not X_test.index.get_level_values(0).dtype == "object":
        raise TypeError("The outer index of X_test must be strings.")
    if not X_test.index.get_level_values(1).dtype == "datetime64[ns]":
        raise TypeError("The inner index of X_test must be datetime.date.")
    if not X_test.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
        raise ValueError(
            "The input feature matrix column for neg_mean_abs_corr",
            " must be numeric.",
        )
    if X_test.isnull().values.any():
        raise ValueError(
            "The input feature matrix for neg_mean_abs_corr must not contain any "
            "missing values."
        )

    # y_test
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")
    if not isinstance(y_test.index, pd.MultiIndex):
        raise ValueError("y_test must be multi-indexed.")
    if not y_test.index.get_level_values(0).dtype == "object":
        raise TypeError("The outer index of y_test must be strings.")
    if not y_test.index.get_level_values(1).dtype == "datetime64[ns]":
        raise TypeError("The inner index of y_test must be datetime.date.")
    if not y_test.index.equals(X_test.index):
        raise ValueError("y_test and X_test must have the same index.")
    if not pd.api.types.is_numeric_dtype(y_test):
        raise ValueError(
            "The input target vector for neg_mean_abs_corr",
            " must be numeric.",
        )
    if y_test.isnull().values.any():
        raise ValueError(
            "The input target vector for neg_mean_abs_corr must not contain any "
            "missing values."
        )
    # Obtain key information
    market_returns = X_test.iloc[:, 0].copy()
    contract_returns = y_test.copy()
    unique_cross_sections = X_test.index.get_level_values(0).unique()
    
    # Handle voting regressor case later
    if isinstance(estimator, VotingRegressor):
        estimators = estimator.estimators_
        coefs_list = [est.coefs_ for est in estimators]
        sum_dict = defaultdict(lambda: [0, 0])

        for coefs in coefs_list:
            for key, value in coefs.items():
                sum_dict[key][0] += value
                sum_dict[key][1] += 1

        estimated_coefs = {key: sum / count for key, (sum, count) in sum_dict.items()}
    else:
        estimated_coefs = estimator.coefs_

    running_sum = 0
    xs_count = 0
    for cross_section in unique_cross_sections:
        # Check whether a model for this cross-section has been estimated
        if cross_section in estimated_coefs.keys():
            xs_count += 1
            # Get cross-section returns and matched risk basket returns
            contract_returns_c = contract_returns.xs(cross_section)
            market_returns_c = market_returns.xs(cross_section)
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
        return -running_sum / xs_count


if __name__ == "__main__":
    import macrosynergy.management as msm
    from macrosynergy.management.simulate import make_qdf
    from macrosynergy.learning import RidgeRegressionSystem

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "BMXR"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2012-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2012-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["BMXR"] = ["2012-01-01", "2020-12-31", 1, 2, 0.95, 1]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    Xy = msm.categories_df(
        df=dfd, xcats=xcats, cids=cids, freq="M", lag=1, xcat_aggs=["last", "sum"]
    ).dropna()
    X = Xy.iloc[:, :-1]
    y = Xy.iloc[:, -1]

    ridge = RidgeRegressionSystem()
    ridge.fit(X, y)
    print(
        "\nNegative mean absolute correlation: "
        f"{neg_mean_abs_corr(ridge, X, y, correlation_type='pearson')}"
    )
