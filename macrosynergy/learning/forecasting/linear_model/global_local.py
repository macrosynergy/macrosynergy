import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, RegressorMixin


class GlobalLocalRegression(BaseEstimator, RegressorMixin):
    r"""
    Linear panel model with hierarchical shrinkage of country-specific (local)
    coefficients towards unknown group and/or global coefficients. Learning
    means that country-specific, group, and global coefficients are estimated
    from data.

    Parameters
    ----------
    local_lambda : float, default=1
        Regularization strength to pull local coefficients towards global coefficients.
        When groups is provided, local coefficients are instead pulled towards the
        coefficients of their group. When shrink_to_global is False and groups is
        None, local slopes are pulled towards zero and local intercepts are not
        penalized.
    global_lambda : float, default=1
        Regularization strength to pull global slopes towards zero. The global
        intercept is not penalized. Has no effect when shrink_to_global is False.
    positive : bool, default=False
        Whether to constrain all coefficients to be positive. Default is False.
    fit_intercept : bool, default=True
        Whether to fit an intercept term. Default is True.
    groups : Union[Dict, str], default=None
        Mapping of group names to list-likes of cross-sections. If "cluster", the
        groups are instead formed at fit time by hierarchical agglomerative clustering
        of the correlations between the returns of the cross-sections, as described in
        the Notes. Default is None, meaning that local coefficients are shrunk directly
        towards the global coefficients.
    group_lambda : float, default=1
        Regularization strength to pull group coefficients towards global coefficients.
        When shrink_to_global=False, group slopes are instead pulled towards zero
        and group intercepts are not penalized. Has no effect when groups is None.
    shrink_to_global : bool, default=True
        Whether to fit global coefficients at the top of the hierarchy. Default is True.
    n_clusters : int, default=2
        Number of groups to form when groups is "cluster".
    linkage : str, default="average"
        Linkage method used to merge clusters when groups is "cluster". One of
        "average", "complete" or "single". Default is "average".
    min_overlap : int, default=36
        Minimum number of dates two cross-sections must share for the correlation of
        their returns to be estimated when groups is "cluster". Pairs with less
        overlap are treated as uncorrelated. Default is 36.
    relative_returns : bool, default=False
        Whether to cluster on returns relative to the cross-sectional mean of each
        date, rather than raw returns, when groups is "cluster". Default is False.

    Notes
    -----
    A panel can be modelled from a global perspective, where time series of all
    countries are "pooled" or stacked together, meaning that samples from different
    countries are treated as independent. This is called a pooled regression. With
    one model fit on all countries' data, this is a high-bias, low-variance model.

    A panel can also be modelled from a group perspective, where time series from
    similar countries are "pooled" together and a model fit for each group. Relative
    to the global perspective, this reduces bias but increases variance.

    Alternatively, country-by-country regressions can be fit, with a separate model for
    each country. This is low-bias but high-variance, since each model sees less data.

    A balance can be found between these three cases by balancing this bias-variance
    trade-off. Introduction of bias to the country-by-country models
    can lead to a potentially substantial reduction in variance. Mathematically, this
    fit is found by minimizing the sum of squared residuals for each country, with a
    term that penalizes deviation of country-specific coefficients from group
    coefficients, which in turn penalizes deviation from global coefficients. The
    global slopes are also penalized to prevent them from growing too large.

    The loss function is as follows:

    .. math::

            L(\{\beta_i\}_{i=1}^{C}, \{\gamma_g\}_{g=1}^{G}, \beta) =
            \sum_{i = 1}^{C} \left [ \frac{1}{n_{i}}  \sum_{t=1}^{n_{i}} (y_{it} -
            x_{it}^{\intercal} \beta_{i})^2 \right ] +
            \lambda_{\text{local}} \sum_{i=1}^{C} ||\beta_i - \gamma_{g(i)}||_{2}^{2} +
            \lambda_{\text{group}} \sum_{g=1}^{G} ||\gamma_g - \beta||_{2}^{2} +
            \lambda_{\text{global}} ||\tilde{\beta}||_{2}^{2}

    where :math:`\tilde{\beta}` is :math:`\beta` without its intercept.

    When groups is None, local coefficients are shrunk directly towards the set of
    global coefficients, whose slopes are themselves shrunk towards 0.

    When shrink_to_global is False, no coefficients are shrunk towards global
    coefficients. Instead, the top level of the hierarchy, the local coefficients or
    the group coefficients when groups is provided, has its slopes shrunk towards 0.

    Intercepts at the top of the hierarchy are never shrunk towards 0. This matches
    sklearn's Ridge behaviour.

    When groups="cluster", the training targets are pivoted to one time series
    per cross-section and the Pearson correlation :math:`\rho_{ij}` of each pair is
    computed over their common dates. Pairs sharing fewer than min_overlap dates, or
    with no variance, are treated as uncorrelated. The correlations are turned into
    distances :math:`d_{ij} = \sqrt{0.5 (1 - \rho_{ij})}` and the cross-sections
    are merged by agglomerative hierarchical clustering with the chosen linkage
    until at most n_clusters groups remain.

    When relative_returns is True, each target is first replaced by its deviation
    from the equally weighted mean of the cross-sections observed on the same date,
    so moves common to the whole panel do not drive the clustering. Demeaning
    induces a correlation of about :math:`-1/(C-1)` between otherwise unrelated
    cross-sections, which matters when there are few of them.
    """

    def __init__(
        self,
        local_lambda=1,
        global_lambda=1,
        group_lambda=1,
        positive=False,
        fit_intercept=True,
        groups=None,
        shrink_to_global=True,
        n_clusters=2,
        linkage="average",
        min_overlap=36,
        relative_returns=False,
    ):
        # Attributes
        self.local_lambda = local_lambda
        self.global_lambda = global_lambda
        self.group_lambda = group_lambda
        self.positive = positive
        self.fit_intercept = fit_intercept
        self.groups = groups
        self.shrink_to_global = shrink_to_global
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.min_overlap = min_overlap
        self.relative_returns = relative_returns

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

        self._set_cid_sufficient_statistics(X, y, sample_weight)
        self._set_groups(X, y)

        parents, lambdas = self._coef_hierarchy()

        # Row r of diff_matrix_ @ weights is beta_r - beta_parent(r)
        n_rows = parents.shape[0]
        children = np.flatnonzero(parents >= 0)
        self.diff_matrix_ = np.eye(n_rows)
        self.diff_matrix_[children, parents[children]] = -1

        # shrinkage_[r, j] is the penalty strength on feature j of row r
        self.shrinkage_ = np.repeat(lambdas[:, None], self.n_features_, axis=1)
        if self.fit_intercept:
            self.shrinkage_[parents < 0, 0] = 0  # don't shrink top level intercept

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

        Notes
        -----
        Every penalty term has the form lambda * (row - parent row)^2. A row without a
        parent is compared with zero. All terms are evaluated at once from the following
        two arrays built in fit:

        * diff_matrix_, of shape (n_rows, n_rows). Row r has +1 in column r and -1 in
          the column of the parent of r, so row r of diff_matrix_ @ weights is
          beta_r - beta_parent(r).
        * shrinkage_, of shape (n_rows, n_features_), the shape of
          diff_matrix_ @ weights. Entry (r, j) is the penalty strength on feature j of
          row r.

        The penalty is then np.sum(shrinkage_ * diffs**2).
        """
        return _loss(
            weights=weights,
            diff_matrix=self.diff_matrix_,
            shrinkage=self.shrinkage_,
            xtx=self.XtX_cid_,
            xty=self.Xty_cid_,
            yty=self.yty_cid_,
        )

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

        Notes
        -----
        The penalty term lambda * (beta_r - beta_p)^2 of row r with parent row p has
        derivative 2 * lambda * (beta_r - beta_p) with respect to beta_r, and the
        negative of that with respect to beta_p. Row r of 2 * shrinkage_ * diff_matrix_
        @ weights holds this derivative for row r. Multiplying by diff_matrix_.T adds
        it to row r, through the +1 in column r, and subtracts it from row p, through
        the -1 in column p.
        """
        return _loss_derivative(
            weights=weights,
            diff_matrix=self.diff_matrix_,
            shrinkage=self.shrinkage_,
            xtx=self.XtX_cid_,
            xty=self.Xty_cid_,
        )

    def _set_cid_sufficient_statistics(self, X, y, sample_weight):
        """
        For each cid compute the following statistics:
            * X^T W X / s
            * X^T W y / s
            * y^T W y / s
        where s is the total sample weight of the cross-section.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix, including the intercept column when fitted.
        y : pd.DataFrame or pd.Series
            Target vector with one value per row of X.
        sample_weight : np.ndarray or None
            Sample weights with one value per row of X, or None for equal weights.
        """
        suff_stats = _compute_sufficient_statistics(
            x=X,
            y=y,
            sample_weight=sample_weight,
            cids=self.cids_,
        )

        self.XtX_cid_, self.Xty_cid_, self.yty_cid_ = suff_stats

    def _set_groups(self, X, y):
        """
        Assign each training cross-section to a group, from `groups` or by clustering.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix, multi-indexed by `cid` and `real_date`.
        y : pd.DataFrame or pd.Series
            Target vector with one value per row of X.
        """
        if self.groups is None:
            return

        if isinstance(self.groups, str):
            self.cid_to_group_ = self._cluster_groups(X, y)
        else:
            self.cid_to_group_ = _validate_groups(self.groups, self.cids_)

        self.groups_ = sorted(set(self.cid_to_group_.values()))
        self.n_groups_ = len(self.groups_)

    def _coef_hierarchy(self):
        """
        Assign parent row and shrinkage strength of each coefficient row. Rows are
        ordered as local rows, then group rows when groups is provided, then the
        global row when shrink_to_global is True. Each row is shrunk towards its
        parent row. A parent of -1 marks a root, whose slopes are shrunk towards zero.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Integer parent row index and float shrinkage strength of each row.
        """
        return _coef_hierarchy(
            cids=self.cids_,
            local_lambda=self.local_lambda,
            group_lambda=self.group_lambda,
            global_lambda=self.global_lambda,
            groups=getattr(self, "groups_", None),
            cid_to_group=getattr(self, "cid_to_group_", None),
            shrink_to_global=self.shrink_to_global,
        )

    def _cluster_groups(self, X, y):
        """
        Form groups by hierarchical agglomerative clustering of the training targets.
        Cross-sections are clustered on the pairwise correlations of their targets,
        or of their targets relative to the cross-sectional mean of each date when
        relative_returns is True.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix, multi-indexed by cid and real_date.
        y : pd.DataFrame or pd.Series
            Target vector with one value per row of X.

        Returns
        -------
        dict
            Mapping of each training cross-section to the integer label of its group.
        """
        if self.groups != "cluster":
            raise TypeError(
                f"`groups` must be a dictionary mapping group names to lists of "
                f"cross-sections or the string 'cluster', got '{self.groups}'."
            )
        if not isinstance(self.n_clusters, (int, np.integer)):
            raise TypeError("n_clusters must be an integer")
        if not isinstance(self.min_overlap, (int, np.integer)):
            raise TypeError("min_overlap must be an integer")
        if not isinstance(self.relative_returns, (bool, np.bool_)):
            raise TypeError("relative_returns must be a boolean")
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")

        valid_linkages = ["average", "complete", "single"]
        if self.linkage not in valid_linkages:
            raise ValueError(
                f"`linkage` must be one of {valid_linkages}, got {self.linkage!r}."
            )
        if self.min_overlap < 2:
            raise ValueError(
                f"`min_overlap` must be at least 2, got {self.min_overlap}."
            )

        if self.n_clusters > self.n_cids_:
            warnings.warn(
                f"`n_clusters` ({self.n_clusters}) is larger than the number of "
                f"training cross-sections ({self.n_cids_}). Falling back to a single cluster.",
                UserWarning,
            )
            n_clusters = 1
        else:
            n_clusters = self.n_clusters

        y_vals = np.asarray(y, dtype=float).reshape(len(y))
        rets = pd.Series(y_vals, index=X.index)

        clusters, cid_to_group = _cluster_returns(
            rets=rets,
            cids=self.cids_,
            n_clusters=n_clusters,
            min_overlap=self.min_overlap,
            linkage=self.linkage,
            relative_returns=self.relative_returns,
        )

        self.clusters_ = clusters

        return cid_to_group

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
                group: coefs[self.n_cids_ + k] for k, group in enumerate(self.groups_)
            }

    def _calculate_map_coefs(self, n_rows: int, loss_fn, jac_fn):
        bounds = _define_bounds(
            n_rows=n_rows,
            positive=self.positive,
            fit_intercept=self.fit_intercept,
            n_features=self.n_features_,
        )

        result = minimize(
            fun=loss_fn,
            jac=jac_fn,
            x0=np.zeros(n_rows * self.n_features_),  # initialize with zeros
            method="L-BFGS-B",
            bounds=bounds,
        )

        coefs = result.x.reshape(n_rows, self.n_features_)

        return coefs


def _mse(
    weights: np.ndarray,
    xtx: np.ndarray,
    xty: np.ndarray,
    yty: np.ndarray,
) -> float:
    n_cids = xtx.shape[0]
    local = weights[:n_cids]
    xtx_b = np.einsum("cij,cj->ci", xtx, local)

    return yty.sum() - 2 * np.sum(local * xty) + np.sum(local * xtx_b)


def _mse_gradient(weights: np.ndarray, xtx: np.ndarray, xty: np.ndarray) -> np.ndarray:
    n_cids = xtx.shape[0]
    local = weights[:n_cids]

    grad = 2 * (np.einsum("cij,cj->ci", xtx, local) - xty)

    return grad


def _loss(
    weights: np.ndarray,
    diff_matrix: np.ndarray,
    shrinkage: np.ndarray,
    xtx: np.ndarray,
    xty: np.ndarray,
    yty: np.ndarray,
) -> float:
    n_features = xtx.shape[1]

    weights = weights.reshape(-1, n_features)
    diffs = diff_matrix @ weights

    mse = _mse(weights=weights, xtx=xtx, xty=xty, yty=yty)

    return mse + np.sum(shrinkage * diffs**2)


def _loss_derivative(
    weights: np.ndarray,
    diff_matrix: np.ndarray,
    shrinkage: np.ndarray,
    xtx: np.ndarray,
    xty: np.ndarray,
) -> np.ndarray:
    n_cids = xtx.shape[0]
    n_features = xtx.shape[1]

    weights = weights.reshape(-1, n_features)
    diffs = diff_matrix @ weights

    mse_grad = _mse_gradient(weights, xtx, xty)

    grads = diff_matrix.T @ (2 * shrinkage * diffs)
    grads[:n_cids] += mse_grad

    return grads.flatten()


def _compute_sufficient_statistics(
    x: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame, np.ndarray],
    cids: List[str],
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_cids = len(cids)
    n_features = x.shape[1]

    x_vals = np.asarray(x, dtype=float)
    y_vals = np.asarray(y, dtype=float).reshape(len(y))
    w_vals = (
        np.ones(y_vals.shape[0])
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )

    # One stable sort makes each cross-section's rows a contiguous block
    cid_codes = pd.Categorical(x.index.get_level_values(0), categories=cids).codes
    order = np.argsort(cid_codes, kind="stable")
    starts = np.searchsorted(cid_codes[order], np.arange(n_cids + 1))

    x_vals = x_vals[order]
    y_vals = y_vals[order]
    w_vals = w_vals[order]

    xtx = np.zeros((n_cids, n_features, n_features))
    xty = np.zeros((n_cids, n_features))
    yty = np.zeros(n_cids)

    for i in range(n_cids):
        start, end = starts[i], starts[i + 1]

        X_i = x_vals[start:end]
        y_i = y_vals[start:end]
        w_i = w_vals[start:end]

        total_weight = w_i.sum()

        xtx[i] = (X_i * w_i[:, None]).T @ X_i / total_weight
        xty[i] = X_i.T @ (w_i * y_i) / total_weight
        yty[i] = w_i @ y_i**2 / total_weight

    return xtx, xty, yty


def _cluster_returns(
    rets: pd.Series,
    cids: List[str],
    n_clusters: int,
    min_overlap: int,
    linkage: str,
    relative_returns: bool = False,
) -> Tuple[Dict[int, List[str]], Dict[str, int]]:
    labels = [1]
    if len(cids) != 1:
        rets = rets.unstack(level=0).reindex(columns=cids)
        if relative_returns:
            rets = rets.sub(rets.mean(axis=1), axis=0)
        corr = rets.corr(min_periods=min_overlap).to_numpy()

        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1)

        dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
        np.fill_diagonal(dist, 0)

        linkage_matrix = hierarchy.linkage(
            y=squareform(dist, checks=False), method=linkage
        )
        labels = hierarchy.fcluster(
            Z=linkage_matrix,
            t=n_clusters,
            criterion="maxclust",
        )

    # Label groups
    relabel = {}
    for label in labels:
        relabel.setdefault(label, len(relabel))
    cid_to_group = {cid: relabel[label] for cid, label in zip(cids, labels)}

    clusters = {
        group: [cid for cid in cids if cid_to_group[cid] == group]
        for group in range(len(relabel))
    }

    return clusters, cid_to_group


def _coef_hierarchy(
    cids: List[str],
    local_lambda: float,
    group_lambda: float,
    global_lambda: float,
    groups: Optional[List[int]],
    cid_to_group: Optional[Dict[str, int]] = None,
    shrink_to_global: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    n_cids = len(cids)

    # Local rows start at the top of the hierarchy, as roots with parent -1
    parents = [-1] * n_cids
    lambdas = [local_lambda] * n_cids

    # Each local row points at its group row, and the group rows become the roots
    if groups is not None:
        group_rows = {group: n_cids + k for k, group in enumerate(groups)}
        parents = [group_rows[cid_to_group[cid]] for cid in cids]
        parents += [-1] * len(groups)
        lambdas += [group_lambda] * len(groups)

    # The current roots point at one global row, which becomes the only root
    if shrink_to_global:
        global_row = len(parents)
        parents = [global_row if p == -1 else p for p in parents] + [-1]
        lambdas += [global_lambda]

    return np.array(parents), np.array(lambdas, dtype=float)


def _define_bounds(
    n_rows: int,
    positive: bool,
    fit_intercept: bool,
    n_features: int,
) -> Union[List[Tuple], None]:
    bounds = None
    if positive:
        row_bounds = [(0, None)] * n_features
        if fit_intercept:
            row_bounds[0] = (None, None)  # no restriction on intercepts
        bounds = row_bounds * n_rows

    return bounds


def _validate_groups(
    groups: Dict[str, List[str]], cids: List[str]
) -> Dict[str, List[str]]:
    if not isinstance(groups, dict):
        raise TypeError(
            "groups must be a dictionary mapping group names to lists of "
            "cross-sections"
        )

    # Each listed cid may belong to one group only
    listed_cid_to_group = {}
    for group, group_cids in groups.items():
        for cid in group_cids:
            if cid in listed_cid_to_group and listed_cid_to_group[cid] != group:
                raise ValueError(
                    f"Cross-section {cid} is assigned to more than one group: "
                    f"{listed_cid_to_group[cid]} and {group}."
                )
            listed_cid_to_group[cid] = group

    # Every training cid must belong to a group
    missing_cids = [cid for cid in cids if cid not in listed_cid_to_group]
    if len(missing_cids) > 0:
        raise ValueError(
            f"The following cross-sections in the training data are not assigned "
            f"to any group: {missing_cids}."
        )

    return {cid: listed_cid_to_group[cid] for cid in cids}


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
