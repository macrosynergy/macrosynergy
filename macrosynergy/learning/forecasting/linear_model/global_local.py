from typing import Union, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin


class GlobalLocalRegression(BaseEstimator, RegressorMixin):
    r"""
    Linear panel model with hierarchical shrinkage of country-specific (local)
    coefficients towards unknown global coefficients. Learning means that both
    country-specific and global coefficients are estimated from data.

    Parameters
    ----------
    local_lambda : float, default=1
        Regularization strength to pull local coefficients towards global coefficients.
        When `groups` is provided, local coefficients are instead pulled towards the
        coefficients of their group. When `shrink_to_global` is False and `groups` is
        None, local slopes are pulled towards zero and local intercepts are not
        penalized.
    global_lambda : float, default=1
        Regularization strength to pull global coefficients towards zero. Has no effect
        when `shrink_to_global` is False.
    positive : bool, default=False
        Whether to constrain all coefficients to be positive. Default is False.
    fit_intercept : bool, default=True
        Whether to fit an intercept term. Default is True.
    min_xs_samples : int, default=36
        Minimum number of samples required in each group for the group to be considered
        a contribution to the mean squared error component of the loss function.
    groups : dict, default=None
        Mapping of group names to list-likes of cross-sections. If provided, a
        three-level hierarchy is fit: local coefficients are shrunk towards the
        coefficients of their group, group coefficients are shrunk towards the global
        coefficients and the global coefficients are shrunk towards zero. Every
        cross-section in the training data must belong to exactly one group.
        Cross-sections that are absent from the training data are ignored, and groups
        without any training cross-sections are dropped. Default is None, meaning that
        local coefficients are shrunk directly towards the global coefficients.
    group_lambda : float, default=1
        Regularization strength to pull group coefficients towards global coefficients.
        When `shrink_to_global` is False, group slopes are instead pulled towards zero
        and group intercepts are not penalized. Has no effect when `groups` is None.
    shrink_to_global : bool, default=True
        Whether to fit global coefficients at the top of the hierarchy. If False, no
        global coefficients are fit and the top level of the hierarchy, which is the
        group coefficients when `groups` is provided and the local coefficients
        otherwise, has its slopes shrunk towards zero and its intercepts left
        unpenalized. Without `groups`, this amounts to a separate ridge regression per
        cross-section. `coef_` and `intercept_` are not set in this case.

    Notes
    -----
    A panel can be modelled from a global perspective, where time series of all countries
    are "pooled" or stacked together, meaning that samples from different countries are
    treated as independent. This is called a pooled regression. With one model fit on all
    countries' data, this is a high-bias, low-variance model.

    Alternatively, country-by-country regressions can be fit, with a separate model for
    each country. This is low-bias but high-variance, since each model sees less data.

    This implies that a balance can be found between these two extremes by balancing
    this bias-variance trade-off. Introduction of bias to the country-by-country models
    can lead to a potentially substantial reduction in variance. Mathematically, this fit
    is found by minimizing the sum of squared residuals for each country, with a term
    that penalizes deviation of country-specific coefficients from a global coefficient.
    The global coefficient is also penalized to prevent it from growing too large.

    The loss function is as follows:

    .. math::

            L(\{\beta_i\}_{i=1}^{C}, \beta) = \sum_{i = 1}^{C} \left [ \frac{1}{n_{i}}  \sum_{t=1}^{n_{i}} (y_{it} - x_{it}^{\intercal} \beta_{i})^2 \right ] + \lambda_{\text{local}} \sum_{i=1}^{C} ||\beta_i - \beta||_{2}^{2} + \lambda_{\text{global}} ||\beta||_{2}^{2}

    When `groups` is provided, an intermediate level of group coefficients
    :math:`\gamma_g` is added between the local and global coefficients. Local
    coefficients are pulled towards the coefficients of their group :math:`g(i)`, and
    group coefficients are pulled towards the global coefficients. The loss function
    is then:

    .. math::

            L(\{\beta_i\}_{i=1}^{C}, \{\gamma_g\}_{g=1}^{G}, \beta) = \sum_{i = 1}^{C} \left [ \frac{1}{n_{i}}  \sum_{t=1}^{n_{i}} (y_{it} - x_{it}^{\intercal} \beta_{i})^2 \right ] + \lambda_{\text{local}} \sum_{i=1}^{C} ||\beta_i - \gamma_{g(i)}||_{2}^{2} + \lambda_{\text{group}} \sum_{g=1}^{G} ||\gamma_g - \beta||_{2}^{2} + \lambda_{\text{global}} ||\beta||_{2}^{2}

    where the mean squared error of each cross-section is weighted when sample weights
    are passed to `fit`.

    When `shrink_to_global` is False, no global coefficients are fit and the top level
    of the hierarchy is shrunk towards zero instead. Writing :math:`\tilde{\beta}` for a
    coefficient vector without its intercept, the loss function without groups is:

    .. math::

            L(\{\beta_i\}_{i=1}^{C}) = \sum_{i = 1}^{C} \left [ \frac{1}{n_{i}}  \sum_{t=1}^{n_{i}} (y_{it} - x_{it}^{\intercal} \beta_{i})^2 \right ] + \lambda_{\text{local}} \sum_{i=1}^{C} ||\tilde{\beta}_i||_{2}^{2}

    and with groups:

    .. math::

            L(\{\beta_i\}_{i=1}^{C}, \{\gamma_g\}_{g=1}^{G}) = \sum_{i = 1}^{C} \left [ \frac{1}{n_{i}}  \sum_{t=1}^{n_{i}} (y_{it} - x_{it}^{\intercal} \beta_{i})^2 \right ] + \lambda_{\text{local}} \sum_{i=1}^{C} ||\beta_i - \gamma_{g(i)}||_{2}^{2} + \lambda_{\text{group}} \sum_{g=1}^{G} ||\tilde{\gamma}_g||_{2}^{2}
    """

    def __init__(
        self,
        local_lambda=1,
        global_lambda=1,
        group_lambda=1,
        positive=False,
        fit_intercept=True,
        min_xs_samples=36,
        groups=None,
        shrink_to_global=True,
    ):
        # Attributes
        self.local_lambda = local_lambda
        self.global_lambda = global_lambda
        self.group_lambda = group_lambda
        self.positive = positive
        self.fit_intercept = fit_intercept
        self.min_xs_samples = min_xs_samples
        self.groups = groups
        self.shrink_to_global = shrink_to_global

    def fit(self, X, y, sample_weight=None):
        """
        Fit the global-local model.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix, multi-indexed by `cid` and `real_date`.
        y : pd.DataFrame or pd.Series
            Target vector associated with each sample in X, multi-indexed by `cid` and
            `real_date`.
        sample_weight : np.ndarray, optional
            Sample weights for each sample in X. If provided, it should be a 1D
            array with the same length as the number of samples in X. If None, all
            samples are treated equally.

        Returns
        -------
        self
            Fitted estimator.
        """
        # Fit
        if self.fit_intercept:
            X = X.copy()
            X.insert(0, "intercept", 1)

        self.cids_ = sorted(X.index.get_level_values(0).unique())
        self.n_cids_ = len(self.cids_)
        self.n_features_ = X.shape[1]
        self._set_cid_summaries(X, y, sample_weight)

        # Each coefficient row is shrunk towards its parent row. Local rows point at
        # the global row, or at their group row when groups is provided.
        if self.shrink_to_global:
            if self.groups is None:
                n_rows = self.n_cids_ + 1
                parents = [n_rows - 1] * self.n_cids_
                lambdas = [self.local_lambda] * self.n_cids_
            else:
                self.cid_to_group_ = self._validate_groups()
                self.groups_ = sorted(set(self.cid_to_group_.values()))
                self.n_groups_ = len(self.groups_)
                n_rows = self.n_cids_ + self.n_groups_ + 1
                group_rows = {
                    group: self.n_cids_ + k for k, group in enumerate(self.groups_)
                }
                parents = [group_rows[self.cid_to_group_[cid]] for cid in self.cids_]
                parents += [n_rows - 1] * self.n_groups_
                lambdas = [self.local_lambda] * self.n_cids_
                lambdas += [self.group_lambda] * self.n_groups_

            # Row r of diff_matrix_ @ weights is beta_r - beta_parent(r). The global
            # row has no parent, so it is shrunk towards zero.
            self.diff_matrix_ = np.eye(n_rows)
            self.diff_matrix_[np.arange(n_rows - 1), parents] = -1
            lambdas = np.array(lambdas + [self.global_lambda], dtype=float)
            self.shrinkage_ = np.repeat(lambdas[:, None], self.n_features_, axis=1)
        else:
            # There is no global row. The top level, the local rows or the group rows
            # when groups is provided, has no parent and is shrunk towards zero.
            if self.groups is None:
                n_rows = self.n_cids_
                top_rows = np.arange(n_rows)
                self.diff_matrix_ = np.eye(n_rows)
                lambdas = np.array([self.local_lambda] * n_rows, dtype=float)
            else:
                self.cid_to_group_ = self._validate_groups()
                self.groups_ = sorted(set(self.cid_to_group_.values()))
                self.n_groups_ = len(self.groups_)
                n_rows = self.n_cids_ + self.n_groups_
                top_rows = np.arange(self.n_cids_, n_rows)
                group_rows = {
                    group: self.n_cids_ + k for k, group in enumerate(self.groups_)
                }
                parents = [group_rows[self.cid_to_group_[cid]] for cid in self.cids_]
                self.diff_matrix_ = np.eye(n_rows)
                self.diff_matrix_[np.arange(self.n_cids_), parents] = -1
                lambdas = np.array(
                    [self.local_lambda] * self.n_cids_
                    + [self.group_lambda] * self.n_groups_,
                    dtype=float,
                )

            # As in ridge regression, intercepts are not shrunk towards zero
            self.shrinkage_ = np.repeat(lambdas[:, None], self.n_features_, axis=1)
            if self.fit_intercept:
                self.shrinkage_[top_rows, 0] = 0

        coefs = self._calculate_map_coefs(
            n_rows=n_rows,
            loss_fn=self.loss,
            jac_fn=self.loss_derivative,
        )

        self._assign_coef_and_intercept_attrs(coefs)

        return self

    def predict(self, X):
        """
        Predict the target values for the given input data.

        Parameters
        ----------
        X : pd.DataFrame
            Input features for prediction.

        Returns
        -------
        np.ndarray
            Predicted target values.
        """
        if self.fit_intercept:
            X = X.copy()
            X.insert(0, "intercept", 1)

        group_indices = X.index.get_level_values(0)
        X_values = X.values
        preds = np.zeros(len(X))

        for group in self.cids_:
            mask = group_indices == group
            if mask.any():
                beta = self.cid_coefs_.get(group, np.zeros(self.n_features_))
                preds[mask] = X_values[mask] @ beta

        return preds

    def loss(self, weights):
        """
        Loss function for the global-local regression model.

        Parameters
        ----------
        weights : np.ndarray
            Flattened array of weights, reshaping to rows of n_features_ elements.
            The first n_cids_ rows are the local coefficients for each country in
            cids_ order. When groups is provided, the next n_groups_ rows are the
            group coefficients in groups_ order. When shrink_to_global is True, the
            last row is the global coefficients.
        """
        weights = weights.reshape(-1, self.n_features_)
        diffs = self.diff_matrix_ @ weights

        return self._mse(weights) + np.sum(self.shrinkage_ * diffs**2)

    def loss_derivative(self, weights):
        """
        Derivative of the loss function with respect to the weights.

        Parameters
        ----------
        weights : np.ndarray
            Flattened array of weights, reshaping to rows of n_features_ elements.
            The first n_cids_ rows are the local coefficients for each country in
            cids_ order. When groups is provided, the next n_groups_ rows are the
            group coefficients in groups_ order. When shrink_to_global is True, the
            last row is the global coefficients.
        """
        weights = weights.reshape(-1, self.n_features_)
        diffs = self.diff_matrix_ @ weights

        # The transpose sends each penalty gradient back to the child row (+) and to
        # its parent row (-)
        grads = self.diff_matrix_.T @ (2 * self.shrinkage_ * diffs)
        grads[: self.n_cids_] += self._mse_gradient(weights)

        return grads.flatten()

    def _mse(self, weights):
        # Each cross-section's weighted MSE expands to yty - 2 b'Xty + b'XtX b
        local = weights[: self.n_cids_]
        XtX_b = np.einsum("cij,cj->ci", self.XtX_cid_, local)

        return (
            self.yty_cid_.sum()
            - 2 * np.sum(local * self.Xty_cid_)
            + np.sum(local * XtX_b)
        )

    def _mse_gradient(self, weights):
        local = weights[: self.n_cids_]

        return 2 * (np.einsum("cij,cj->ci", self.XtX_cid_, local) - self.Xty_cid_)

    def _set_cid_summaries(self, X, y, sample_weight):
        """
        Store the weighted second moments of each cross-section's training data.

        Sets `XtX_cid_`, `Xty_cid_` and `yty_cid_`, with rows in `cids_` order. Each
        moment is normalised by the cross-section's total sample weight, so that the
        mean squared error of a cross-section can be evaluated from them alone. `y`
        and `sample_weight` are aligned with `X` by row position.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix, including the intercept column when fitted.
        y : pd.DataFrame or pd.Series
            Target vector with one value per row of X.
        sample_weight : np.ndarray or None
            Sample weights with one value per row of X, or None for equal weights.
        """
        X_values = X.to_numpy(dtype=float)
        y_values = np.asarray(y, dtype=float).reshape(len(y))
        if sample_weight is None:
            w_values = np.ones(len(y_values))
        else:
            w_values = np.asarray(sample_weight, dtype=float)

        # One stable sort makes each cross-section's rows a contiguous block
        cid_codes = pd.Categorical(
            X.index.get_level_values(0), categories=self.cids_
        ).codes
        order = np.argsort(cid_codes, kind="stable")
        starts = np.searchsorted(cid_codes[order], np.arange(self.n_cids_ + 1))
        X_values = X_values[order]
        y_values = y_values[order]
        w_values = w_values[order]

        self.XtX_cid_ = np.zeros((self.n_cids_, self.n_features_, self.n_features_))
        self.Xty_cid_ = np.zeros((self.n_cids_, self.n_features_))
        self.yty_cid_ = np.zeros(self.n_cids_)
        for i in range(self.n_cids_):
            block = slice(starts[i], starts[i + 1])
            X_i, y_i, w_i = X_values[block], y_values[block], w_values[block]

            total_weight = w_i.sum()

            self.XtX_cid_[i] = (X_i * w_i[:, None]).T @ X_i / total_weight
            self.Xty_cid_[i] = X_i.T @ (w_i * y_i) / total_weight
            self.yty_cid_[i] = w_i @ y_i**2 / total_weight

    def _validate_groups(self):
        """
        Validate `groups` against the training cross-sections.

        Returns
        -------
        dict
            Mapping of each training cross-section to the name of its group.
        """
        if not isinstance(self.groups, dict):
            raise TypeError(
                f"`groups` must be a dictionary mapping group names to lists of "
                f"cross-sections, got {type(self.groups).__name__}."
            )

        # Each group must map to a list-like of cids, not a single string
        for group, group_cids in self.groups.items():
            if isinstance(group_cids, str):
                raise TypeError(
                    f"Each group in `groups` must map to a list-like of "
                    f"cross-sections, got the string '{group_cids}' for group {group}."
                )

        # Each listed cid may belong to one group only
        listed_cid_to_group = {}
        for group, group_cids in self.groups.items():
            for cid in group_cids:
                if cid in listed_cid_to_group and listed_cid_to_group[cid] != group:
                    raise ValueError(
                        f"Cross-section {cid} is assigned to more than one group: "
                        f"{listed_cid_to_group[cid]} and {group}."
                    )
                listed_cid_to_group[cid] = group

        # Every training cid must belong to a group
        missing_cids = [cid for cid in self.cids_ if cid not in listed_cid_to_group]
        if len(missing_cids) > 0:
            raise ValueError(
                f"The following cross-sections in the training data are not assigned "
                f"to any group: {missing_cids}."
            )

        return {cid: listed_cid_to_group[cid] for cid in self.cids_}

    def _assign_coef_and_intercept_attrs(self, coefs):
        self.cid_coefs_ = {cid: coefs[i] for i, cid in enumerate(self.cids_)}

        # Without a global row there are no global coefficients to report
        if self.shrink_to_global:
            if self.fit_intercept:
                self.coef_ = coefs[-1, 1:]
                self.intercept_ = coefs[-1, 0]
            else:
                self.coef_ = coefs[-1]
                self.intercept_ = None

        if self.groups is not None:
            self.group_coefs_ = {
                group: coefs[self.n_cids_ + k]
                for k, group in enumerate(self.groups_)
            }

    def _define_bounds(self, n_rows: int) -> Union[List[Tuple], None]:
        bounds = None
        if self.positive:
            row_bounds = [(0, None)] * self.n_features_
            if self.fit_intercept:
                row_bounds[0] = (None, None)  # no restriction on intercepts
            bounds = row_bounds * n_rows

        return bounds

    def _calculate_map_coefs(self, n_rows: int, loss_fn, jac_fn):
        result = minimize(
            fun=loss_fn,
            jac=jac_fn,
            x0=np.zeros(n_rows * self.n_features_),  # initialize with zeros
            method="L-BFGS-B",
            bounds=self._define_bounds(n_rows),
        )

        coefs = result.x.reshape(n_rows, self.n_features_)

        return coefs


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

    # Regressor
    X_train = train.drop(columns=["XR"])
    y_train = np.sign(train["XR"])

    glr = GlobalLocalRegression().fit(X_train, y_train)
    print(glr.predict(X_train))
