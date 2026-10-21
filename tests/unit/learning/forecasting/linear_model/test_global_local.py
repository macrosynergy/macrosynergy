"""
Unit tests for GlobalLocalRegression, including the grouped (three-level) mode.

Every expected value is computed by an independent reference built in this module,
written directly from the model's loss formulas rather than from the class:

    two-level (groups=None):
        L = sum_i fit_i(beta_i)
            + local_lambda  * sum_i ||beta_i - beta||^2
            + global_lambda * ||beta||^2

    grouped:
        L = sum_i fit_i(beta_i)
            + local_lambda  * sum_i ||beta_i - gamma_{g(i)}||^2
            + group_lambda  * sum_g ||gamma_g - beta||^2
            + global_lambda * ||beta||^2

    fit_i(b) = sum_t w_it * (y_it - x_it' b)^2 / sum_t w_it    (w_it = 1 if unweighted)

shrink_to_global=False drops the global row and shrinks the top level of the hierarchy
towards zero, where s(b) is b without its intercept (b itself without fit_intercept):

    no groups:
        L = sum_i fit_i(beta_i) + local_lambda * sum_i ||s(beta_i)||^2

    grouped:
        L = sum_i fit_i(beta_i)
            + local_lambda  * sum_i ||beta_i - gamma_{g(i)}||^2
            + group_lambda  * sum_g ||s(gamma_g)||^2

With fit_intercept=True column 0 of every row is the intercept and is penalised like
any other column, except in the s(.) penalties above. The flat parameter vector holds
the local rows (sorted cid order), then the group rows (sorted group-name order,
grouped mode only), then the global row (shrink_to_global=True only).

For positive=False every objective is quadratic, so the minimiser solves a linear
system, which the reference solves exactly. The reference is validated against the
existing two-level class before it is used to judge the grouped mode.
"""

import inspect
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.base import clone
from sklearn.linear_model import Ridge

from macrosynergy.learning import GlobalLocalRegression

# The class calls the `minimize` it imported into its own module.
MINIMIZE_TARGET = "macrosynergy.learning.forecasting.linear_model.global_local.minimize"

# --------------------------------------------------------------------------------------
# Synthetic panel
# --------------------------------------------------------------------------------------

FEATURES = ["CRY", "GROWTH", "INFL"]

# Unequal panel lengths per cross-section.
CID_LENGTHS = {"AUD": 40, "BRL": 25, "CAD": 60, "JPY": 35, "KRW": 50, "NOK": 30}

# Group names and members are deliberately listed out of sorted order, and the groups
# interleave in sorted-cid order, so that layout mistakes cannot cancel out.
GROUPS = {
    "commodity": ["NOK", "AUD", "CAD"],
    "asia": ["KRW", "JPY"],
    "latam": ["BRL"],
}
SORTED_GROUPS = ["asia", "commodity", "latam"]
CID_TO_GROUP = {
    "AUD": "commodity",
    "BRL": "latam",
    "CAD": "commodity",
    "JPY": "asia",
    "KRW": "asia",
    "NOK": "commodity",
}

# Generating coefficients, intercept first. The global intercept is clearly negative
# and the GROWTH slope is negative in every group, so that positive=True binds.
GLOBAL_COEFS = np.array([-1.0, 1.0, -0.5, 0.25])
GROUP_OFFSETS = {
    "asia": np.array([0.3, -0.4, 0.2, 0.1]),
    "commodity": np.array([-0.2, 0.5, 0.0, -0.3]),
    "latam": np.array([0.4, 0.1, -0.3, 0.5]),
}

# Distinct penalty strengths, so that a swapped or misapplied penalty shows up.
LOCAL_LAMBDA = 0.5
GLOBAL_LAMBDA = 0.1
GROUP_LAMBDA = 0.3

# Every (fit_intercept, weighted) combination.
VARIANTS = [(True, False), (True, True), (False, False), (False, True)]

# --------------------------------------------------------------------------------------
# Tolerances
# --------------------------------------------------------------------------------------

# The class minimises with L-BFGS-B at scipy's default settings, which stop on a
# relative reduction of the loss of about 2e-9 rather than at the exact optimum.
# Measured on this panel (8 data seeds, all four VARIANTS):
# - existing two-level class vs the exact reference: max abs coefficient error <= 7e-5
#   at the penalties used here, <= 1.5e-4 over a range of other penalties;
# - L-BFGS-B (default settings, zeros start, analytic gradient) on the grouped
#   reference objective vs its exact solution: <= 7e-5 here, <= 7.5e-4 for small,
#   ill-conditioned penalties (0.05, 0.01, 0.02).
# FIT_ATOL leaves a >10x margin over the errors seen at these settings, while the
# errors it must catch are much larger: a swapped lambda, a doubled group penalty, a
# dropped sample weight or a two-level instead of a grouped fit each move some
# coefficient by 0.05-0.19 on this panel.
FIT_ATOL = 1e-3

# Max |projected gradient| of the exact objective at an L-BFGS-B solution with
# positive=True: <= 7e-5 at these settings (<= 1.5e-3 at the ill-conditioned ones).
KKT_ATOL = 1e-3

# Limits use a penalty of 1e3. On this panel the exact limit gaps at 1e3 are ~5e-4
# (group rows onto the global row) and ~3e-4 (local rows onto group rows); with
# L-BFGS-B's early stop on the stiffer problem the measured gaps were <= 1.5e-3 and
# <= 4.2e-4 over 10 data seeds. At the moderate penalties used elsewhere the same
# gaps are >= 0.11, so LIMIT_ATOL separates "in the limit" from "not" by ~10x either
# way. Larger penalties do not help: the gap shrinks but L-BFGS-B's error grows.
LIMIT_LAMBDA = 1e3
LIMIT_ATOL = 1e-2

# Both objectives are quadratic, so central differences have no truncation error and
# only rounding error of order eps * |L| / FD_STEP (~1e-9 here).
FD_STEP = 1e-5
FD_TOL = 1e-6


def make_panel(seed=42):
    """
    Build a small unbalanced panel with a known group structure.

    Returns
    -------
    X : pd.DataFrame
        Features, multi-indexed by `cid` and `real_date`, sorted by cid.
    y : pd.Series
        Targets with the same index as X.
    sample_weight : np.ndarray
        Strictly positive sample weights, one per row of X.
    """
    rng = np.random.default_rng(seed)
    X_blocks = []
    y_blocks = []
    for cid in sorted(CID_LENGTHS):
        n = CID_LENGTHS[cid]
        dates = pd.bdate_range(end="2023-12-29", periods=n)
        index = pd.MultiIndex.from_product([[cid], dates], names=["cid", "real_date"])
        ftrs = rng.normal(size=(n, len(FEATURES)))
        coefs = (
            GLOBAL_COEFS
            + GROUP_OFFSETS[CID_TO_GROUP[cid]]
            + rng.normal(scale=0.1, size=len(GLOBAL_COEFS))
        )
        target = coefs[0] + ftrs @ coefs[1:] + rng.normal(scale=0.5, size=n)
        X_blocks.append(pd.DataFrame(ftrs, index=index, columns=FEATURES))
        y_blocks.append(pd.Series(target, index=index, name="XR"))

    X = pd.concat(X_blocks)
    y = pd.concat(y_blocks)
    sample_weight = rng.uniform(0.5, 2.0, size=len(y))
    return X, y, sample_weight


# --------------------------------------------------------------------------------------
# Independent reference
# --------------------------------------------------------------------------------------


class Reference:
    """
    Independent reference for the two-level and grouped global-local objectives, with
    or without the global row.

    Parameters
    ----------
    X, y, sample_weight
        Panel data as passed to `fit`.
    fit_intercept : bool
        Whether column 0 of each row is an intercept.
    groups : dict or None
        Group name -> cids. None gives the two-level model.
    local_lambda, global_lambda, group_lambda : float
        Penalty strengths. `group_lambda` is ignored when `groups` is None, and
        `global_lambda` when `shrink_to_global` is False.
    shrink_to_global : bool
        Whether there is a global row. Without it, the top level is shrunk towards
        zero, slopes only.
    """

    def __init__(
        self,
        X,
        y,
        sample_weight=None,
        fit_intercept=True,
        groups=None,
        local_lambda=LOCAL_LAMBDA,
        global_lambda=GLOBAL_LAMBDA,
        group_lambda=GROUP_LAMBDA,
        shrink_to_global=True,
    ):
        self.local_lambda = local_lambda
        self.global_lambda = global_lambda
        self.group_lambda = group_lambda
        self.fit_intercept = fit_intercept
        self.shrink_to_global = shrink_to_global

        cid_index = X.index.get_level_values(0)
        self.cids = sorted(cid_index.unique())
        self.blocks = []
        for cid in self.cids:
            mask = np.asarray(cid_index == cid)
            X_i = X.values[mask]
            if fit_intercept:
                X_i = np.column_stack([np.ones(len(X_i)), X_i])
            y_i = y.values[mask]
            if sample_weight is None:
                w_i = np.ones(len(y_i))
            else:
                w_i = np.asarray(sample_weight)[mask]
            self.blocks.append((X_i, y_i, w_i))

        self.n_cids = len(self.cids)
        self.n_features = self.blocks[0][0].shape[1]

        if groups is None:
            self.group_names = []
            self.cid_group = None
        else:
            present = set(self.cids)
            self.group_names = sorted(
                name for name, members in groups.items() if present & set(members)
            )
            self.cid_group = [
                next(
                    k for k, name in enumerate(self.group_names) if cid in groups[name]
                )
                for cid in self.cids
            ]
        self.n_groups = len(self.group_names)
        self.n_rows = self.n_cids + self.n_groups + (1 if shrink_to_global else 0)

        # 1 on slope columns, 0 on the intercept column.
        self.slope_mask = np.ones(self.n_features)
        if fit_intercept:
            self.slope_mask[0] = 0

    def parent_row(self, i):
        """Row that local row `i` is shrunk towards, or None if towards zero."""
        if self.cid_group is None:
            return self.n_rows - 1 if self.shrink_to_global else None
        return self.n_cids + self.cid_group[i]

    def slopes(self, row):
        """`row` without its intercept: s(row) in the formulas."""
        return row[1:] if self.fit_intercept else row

    def objective(self, theta):
        """Loss at the flat parameter vector `theta`, straight from the formula."""
        B = theta.reshape(self.n_rows, self.n_features)
        value = 0.0
        for i, (X_i, y_i, w_i) in enumerate(self.blocks):
            value += np.sum(w_i * (y_i - X_i @ B[i]) ** 2) / np.sum(w_i)
        for i in range(self.n_cids):
            parent = self.parent_row(i)
            if parent is None:
                value += self.local_lambda * np.sum(self.slopes(B[i]) ** 2)
            else:
                value += self.local_lambda * np.sum((B[i] - B[parent]) ** 2)
        for k in range(self.n_groups):
            r = self.n_cids + k
            if self.shrink_to_global:
                value += self.group_lambda * np.sum((B[r] - B[-1]) ** 2)
            else:
                value += self.group_lambda * np.sum(self.slopes(B[r]) ** 2)
        if self.shrink_to_global:
            value += self.global_lambda * np.sum(B[-1] ** 2)
        return value

    def gradient(self, theta):
        """Analytic gradient of `objective`, row-wise as in the model formulas."""
        B = theta.reshape(self.n_rows, self.n_features)
        grad = np.zeros_like(B)
        for i, (X_i, y_i, w_i) in enumerate(self.blocks):
            grad[i] += 2 * X_i.T @ (w_i * (X_i @ B[i] - y_i)) / np.sum(w_i)
            parent = self.parent_row(i)
            if parent is None:
                grad[i] += 2 * self.local_lambda * self.slope_mask * B[i]
            else:
                grad[i] += 2 * self.local_lambda * (B[i] - B[parent])
                grad[parent] += 2 * self.local_lambda * (B[parent] - B[i])
        for k in range(self.n_groups):
            r = self.n_cids + k
            if self.shrink_to_global:
                grad[r] += 2 * self.group_lambda * (B[r] - B[-1])
                grad[-1] += 2 * self.group_lambda * (B[-1] - B[r])
            else:
                grad[r] += 2 * self.group_lambda * self.slope_mask * B[r]
        if self.shrink_to_global:
            grad[-1] += 2 * self.global_lambda * B[-1]
        return grad.ravel()

    def solve(self):
        """
        Exact unconstrained minimiser.

        The objective equals theta' H theta - 2 r' theta + const, with
        H = blockdiag(A_i, 0, ..., 0) + kron(P_slope, S) + kron(P_intercept, I - S).
        A_i = X_i' W_i X_i / sum(w_i) is the weighted Gram matrix of local row i, and
        S = diag(slope_mask). P_slope and P_intercept are the penalty matrices over rows
        for the slope columns and the intercept column: each penalised difference
        lambda * ||row_u - row_v||^2 adds lambda * (e_u - e_v)(e_u - e_v)' to both, the
        global penalty adds global_lambda * e_last e_last' to both, and each slopes-only
        penalty lambda * ||s(row_u)||^2 adds lambda * e_u e_u' to P_slope only. The
        minimiser solves H theta = r.
        """
        n_rows, p = self.n_rows, self.n_features
        P_slope = np.zeros((n_rows, n_rows))
        P_intercept = np.zeros((n_rows, n_rows))

        def add_difference_penalty(u, v, lam):
            for P in (P_slope, P_intercept):
                P[u, u] += lam
                P[v, v] += lam
                P[u, v] -= lam
                P[v, u] -= lam

        for i in range(self.n_cids):
            parent = self.parent_row(i)
            if parent is None:
                P_slope[i, i] += self.local_lambda
            else:
                add_difference_penalty(i, parent, self.local_lambda)
        for k in range(self.n_groups):
            r = self.n_cids + k
            if self.shrink_to_global:
                add_difference_penalty(r, n_rows - 1, self.group_lambda)
            else:
                P_slope[r, r] += self.group_lambda
        if self.shrink_to_global:
            P_slope[-1, -1] += self.global_lambda
            P_intercept[-1, -1] += self.global_lambda

        H = np.kron(P_slope, np.diag(self.slope_mask)) + np.kron(
            P_intercept, np.diag(1 - self.slope_mask)
        )
        r = np.zeros(n_rows * p)
        for i, (X_i, y_i, w_i) in enumerate(self.blocks):
            block = slice(i * p, (i + 1) * p)
            H[block, block] += X_i.T @ (w_i[:, None] * X_i) / np.sum(w_i)
            r[block] = X_i.T @ (w_i * y_i) / np.sum(w_i)
        return np.linalg.solve(H, r)

    def split(self, theta):
        """
        Split a flat vector into ({cid: row}, {group: row}, global row). The global
        row is None when shrink_to_global is False.
        """
        B = theta.reshape(self.n_rows, self.n_features)
        cid_rows = {cid: B[i] for i, cid in enumerate(self.cids)}
        group_rows = {
            name: B[self.n_cids + k] for k, name in enumerate(self.group_names)
        }
        global_row = B[-1] if self.shrink_to_global else None
        return cid_rows, group_rows, global_row

    def theta_from_model(self, model):
        """Flat vector in the reference layout, read from a fitted model."""
        rows = [model.cid_coefs_[cid] for cid in self.cids]
        rows += [model.group_coefs_[name] for name in self.group_names]
        if self.shrink_to_global:
            if self.fit_intercept:
                rows.append(np.concatenate([[model.intercept_], model.coef_]))
            else:
                rows.append(model.coef_)
        return np.concatenate(rows)

    def lower_bounds(self, positive):
        """Lower bound per entry of the flat vector (-inf where unconstrained)."""
        row = np.full(self.n_features, 0.0 if positive else -np.inf)
        if self.fit_intercept:
            row[0] = -np.inf
        return np.tile(row, self.n_rows)


def finite_difference_gradient(fun, theta, step=FD_STEP):
    """Central finite differences of `fun` at `theta`."""
    grad = np.zeros_like(theta)
    for j in range(len(theta)):
        e = np.zeros_like(theta)
        e[j] = step
        grad[j] = (fun(theta + e) - fun(theta - e)) / (2 * step)
    return grad


def projected_gradient(theta, grad, lower):
    """
    Projected gradient for lower bounds, as L-BFGS-B measures it. A positive gradient
    on a bounded entry points a descent step towards the bound, so it only counts up
    to the distance left to the bound; an entry on its bound with a positive gradient
    contributes zero. At a constrained minimiser (KKT point) every entry is zero.
    """
    pg = grad.copy()
    bounded = np.isfinite(lower)
    pushes_down = bounded & (grad > 0)
    pg[pushes_down] = np.minimum(
        theta[pushes_down] - lower[pushes_down], grad[pushes_down]
    )
    return pg


def expected_bounds(n_rows, n_features, positive, fit_intercept):
    """
    Bounds the spec prescribes for `n_rows` coefficient rows: None (unconstrained)
    without `positive`; with it, a list where every slope column >= 0 and the
    intercept column (column 0, when fitted) is free.
    """
    if not positive:
        return None
    row = [(0, None)] * n_features
    if fit_intercept:
        row[0] = (None, None)
    return row * n_rows


def variant_label(fit_intercept, weighted):
    return f"fit_intercept={fit_intercept}, weighted={weighted}"


def grouped_model(**overrides):
    """Grouped model at the module's penalty settings."""
    params = {
        "local_lambda": LOCAL_LAMBDA,
        "global_lambda": GLOBAL_LAMBDA,
        "groups": GROUPS,
        "group_lambda": GROUP_LAMBDA,
    }
    params.update(overrides)
    return GlobalLocalRegression(**params)


def no_global_model(**overrides):
    """Model without a global row at the module's penalty settings, ungrouped."""
    params = {
        "local_lambda": LOCAL_LAMBDA,
        "global_lambda": GLOBAL_LAMBDA,
        "group_lambda": GROUP_LAMBDA,
        "shrink_to_global": False,
    }
    params.update(overrides)
    return GlobalLocalRegression(**params)


def groups_label(groups, fit_intercept, weighted):
    return f"groups={groups is not None}, {variant_label(fit_intercept, weighted)}"


class PanelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.sample_weight = make_panel()

    def weights(self, weighted):
        return self.sample_weight if weighted else None

    def assert_matches_reference(self, model, ref, theta, atol, label=""):
        """Compare every fitted coefficient attribute with the reference vector."""
        cid_rows, group_rows, global_row = ref.split(theta)
        self.assertEqual(list(model.cids_), ref.cids)
        for cid in ref.cids:
            np.testing.assert_allclose(
                model.cid_coefs_[cid],
                cid_rows[cid],
                rtol=0,
                atol=atol,
                err_msg=f"{label} cid_coefs_[{cid}]",
            )
        if ref.n_groups:
            self.assertEqual(list(model.groups_), ref.group_names)
            for name in ref.group_names:
                np.testing.assert_allclose(
                    model.group_coefs_[name],
                    group_rows[name],
                    rtol=0,
                    atol=atol,
                    err_msg=f"{label} group_coefs_[{name}]",
                )
        if global_row is None:
            for name in ["coef_", "intercept_"]:
                self.assertFalse(hasattr(model, name), msg=f"{label} {name}")
        elif ref.fit_intercept:
            np.testing.assert_allclose(
                model.coef_, global_row[1:], rtol=0, atol=atol, err_msg=f"{label} coef_"
            )
            np.testing.assert_allclose(
                model.intercept_,
                global_row[0],
                rtol=0,
                atol=atol,
                err_msg=f"{label} intercept_",
            )
        else:
            np.testing.assert_allclose(
                model.coef_, global_row, rtol=0, atol=atol, err_msg=f"{label} coef_"
            )
            self.assertIsNone(model.intercept_, msg=label)


class TestReference(PanelTestCase):
    """
    Validate the reference against the existing two-level class (groups=None), and
    check the grouped reference for internal consistency. Needs no grouped code.
    """

    def test_two_level_objective_matches_existing_loss(self):
        rng = np.random.default_rng(1)
        for fit_intercept, weighted in VARIANTS:
            label = variant_label(fit_intercept, weighted)
            sw = self.weights(weighted)
            ref = Reference(self.X, self.y, sw, fit_intercept)
            model = GlobalLocalRegression(
                local_lambda=LOCAL_LAMBDA,
                global_lambda=GLOBAL_LAMBDA,
                fit_intercept=fit_intercept,
            ).fit(self.X, self.y, sample_weight=sw)
            theta = rng.normal(size=ref.n_rows * ref.n_features)
            np.testing.assert_allclose(
                model.loss(theta), ref.objective(theta), rtol=1e-12, err_msg=label
            )
            np.testing.assert_allclose(
                model.loss_derivative(theta),
                ref.gradient(theta),
                rtol=1e-10,
                atol=1e-12,
                err_msg=label,
            )

    def test_two_level_fit_matches_closed_form(self):
        for fit_intercept, weighted in VARIANTS:
            label = variant_label(fit_intercept, weighted)
            sw = self.weights(weighted)
            ref = Reference(self.X, self.y, sw, fit_intercept)
            model = GlobalLocalRegression(
                local_lambda=LOCAL_LAMBDA,
                global_lambda=GLOBAL_LAMBDA,
                fit_intercept=fit_intercept,
            ).fit(self.X, self.y, sample_weight=sw)
            self.assert_matches_reference(model, ref, ref.solve(), FIT_ATOL, label)

    def test_grouped_reference_is_self_consistent(self):
        # The analytic reference gradient matches finite differences of the reference
        # objective, and vanishes at the closed-form solution.
        rng = np.random.default_rng(2)
        for fit_intercept, weighted in VARIANTS:
            label = variant_label(fit_intercept, weighted)
            ref = Reference(
                self.X, self.y, self.weights(weighted), fit_intercept, GROUPS
            )
            self.assertEqual(ref.group_names, SORTED_GROUPS)
            theta = rng.normal(size=ref.n_rows * ref.n_features)
            np.testing.assert_allclose(
                ref.gradient(theta),
                finite_difference_gradient(ref.objective, theta),
                rtol=FD_TOL,
                atol=FD_TOL,
                err_msg=label,
            )
            np.testing.assert_allclose(
                ref.gradient(ref.solve()), 0, atol=1e-10, err_msg=label
            )


class TestOptimiserCall(PanelTestCase):
    """
    Every path makes one `minimize` call with exactly the spec's settings: L-BFGS-B,
    zeros start, the path's own loss and analytic gradient, the spec's bounds, and no
    other arguments. The real `minimize` still runs (wraps=), so fits are unchanged.
    """

    def capture_minimize_call(self, model):
        """Fit `model` and return the arguments of its `minimize` call, by name."""
        with mock.patch(MINIMIZE_TARGET, wraps=minimize) as spy:
            model.fit(self.X, self.y)
        self.assertEqual(spy.call_count, 1)
        args, kwargs = spy.call_args
        # Bind to scipy's signature so positional and keyword passing compare alike.
        return inspect.signature(minimize).bind(*args, **kwargs).arguments

    def assert_minimize_call(self, call, fun, jac, n_rows, positive, fit_intercept):
        label = f"positive={positive}, fit_intercept={fit_intercept}"
        n_features = len(FEATURES) + 1 if fit_intercept else len(FEATURES)
        self.assertEqual(set(call), {"fun", "x0", "method", "bounds", "jac"}, msg=label)
        self.assertEqual(call["fun"], fun, msg=label)
        self.assertEqual(call["jac"], jac, msg=label)
        self.assertEqual(call["method"], "L-BFGS-B", msg=label)
        x0 = np.asarray(call["x0"])
        self.assertEqual(x0.shape, (n_rows * n_features,), msg=label)
        np.testing.assert_array_equal(x0, 0, err_msg=label)
        bounds = call["bounds"]
        if bounds is not None:
            bounds = [tuple(b) for b in bounds]
        self.assertEqual(
            bounds,
            expected_bounds(n_rows, n_features, positive, fit_intercept),
            msg=label,
        )

    def test_minimize_call_without_groups(self):
        # n_cids local rows + global row.
        n_rows = len(CID_LENGTHS) + 1
        for positive in [False, True]:
            for fit_intercept in [True, False]:
                model = GlobalLocalRegression(
                    local_lambda=LOCAL_LAMBDA,
                    global_lambda=GLOBAL_LAMBDA,
                    positive=positive,
                    fit_intercept=fit_intercept,
                )
                call = self.capture_minimize_call(model)
                self.assert_minimize_call(
                    call,
                    model.loss,
                    model.loss_derivative,
                    n_rows,
                    positive,
                    fit_intercept,
                )

    def test_minimize_call_with_groups(self):
        # n_cids local rows + n_groups group rows + global row.
        n_rows = len(CID_LENGTHS) + len(SORTED_GROUPS) + 1
        for positive in [False, True]:
            for fit_intercept in [True, False]:
                model = grouped_model(positive=positive, fit_intercept=fit_intercept)
                call = self.capture_minimize_call(model)
                self.assert_minimize_call(
                    call,
                    model.loss,
                    model.loss_derivative,
                    n_rows,
                    positive,
                    fit_intercept,
                )

    def test_minimize_call_without_global(self):
        # n_cids local rows, plus n_groups group rows when grouped. No global row.
        for groups, n_rows in [
            (None, len(CID_LENGTHS)),
            (GROUPS, len(CID_LENGTHS) + len(SORTED_GROUPS)),
        ]:
            for positive in [False, True]:
                for fit_intercept in [True, False]:
                    model = no_global_model(
                        groups=groups, positive=positive, fit_intercept=fit_intercept
                    )
                    call = self.capture_minimize_call(model)
                    self.assert_minimize_call(
                        call,
                        model.loss,
                        model.loss_derivative,
                        n_rows,
                        positive,
                        fit_intercept,
                    )


class TestGroupedGlobalLocalRegression(PanelTestCase):
    """Grouped mode held to the reference and to the interface specification."""

    # ----------------------------------------------------------------------------------
    # Fit, loss and gradient
    # ----------------------------------------------------------------------------------

    def test_grouped_fit_matches_closed_form(self):
        for fit_intercept, weighted in VARIANTS:
            label = variant_label(fit_intercept, weighted)
            sw = self.weights(weighted)
            ref = Reference(self.X, self.y, sw, fit_intercept, GROUPS)
            model = grouped_model(fit_intercept=fit_intercept).fit(
                self.X, self.y, sample_weight=sw
            )
            self.assert_matches_reference(model, ref, ref.solve(), FIT_ATOL, label)

    def test_grouped_loss_matches_reference_objective(self):
        # Pins the parameter-vector layout: local rows (sorted cids), group rows
        # (sorted group names), global row.
        rng = np.random.default_rng(3)
        for fit_intercept, weighted in VARIANTS:
            label = variant_label(fit_intercept, weighted)
            sw = self.weights(weighted)
            ref = Reference(self.X, self.y, sw, fit_intercept, GROUPS)
            model = grouped_model(fit_intercept=fit_intercept).fit(
                self.X, self.y, sample_weight=sw
            )
            theta = rng.normal(size=ref.n_rows * ref.n_features)
            value = model.loss(theta)
            self.assertEqual(np.ndim(value), 0, msg=label)
            np.testing.assert_allclose(
                value, ref.objective(theta), rtol=1e-12, err_msg=label
            )

    def test_grouped_loss_derivative_matches_finite_differences(self):
        rng = np.random.default_rng(4)
        for fit_intercept, weighted in VARIANTS:
            label = variant_label(fit_intercept, weighted)
            sw = self.weights(weighted)
            model = grouped_model(fit_intercept=fit_intercept).fit(
                self.X, self.y, sample_weight=sw
            )
            n_params = (model.n_cids_ + model.n_groups_ + 1) * model.n_features_
            theta = rng.normal(size=n_params)
            grad = model.loss_derivative(theta)
            self.assertEqual(np.shape(grad), (n_params,), msg=label)
            np.testing.assert_allclose(
                grad,
                finite_difference_gradient(model.loss, theta),
                rtol=FD_TOL,
                atol=FD_TOL,
                err_msg=label,
            )

    # ----------------------------------------------------------------------------------
    # Limits
    # ----------------------------------------------------------------------------------

    def test_large_group_lambda_approaches_two_level_fit(self):
        for fit_intercept, weighted in VARIANTS:
            label = variant_label(fit_intercept, weighted)
            sw = self.weights(weighted)

            # Guard against a vacuous test: at the moderate penalty the grouped and
            # two-level solutions are clearly apart.
            grouped_ref = Reference(self.X, self.y, sw, fit_intercept, GROUPS)
            two_level_ref = Reference(self.X, self.y, sw, fit_intercept)
            moderate, _, _ = grouped_ref.split(grouped_ref.solve())
            two_level, _, _ = two_level_ref.split(two_level_ref.solve())
            self.assertGreater(
                max(np.max(np.abs(moderate[c] - two_level[c])) for c in moderate),
                10 * LIMIT_ATOL,
                msg=label,
            )

            grouped = grouped_model(
                fit_intercept=fit_intercept, group_lambda=LIMIT_LAMBDA
            ).fit(self.X, self.y, sample_weight=sw)
            plain = GlobalLocalRegression(
                local_lambda=LOCAL_LAMBDA,
                global_lambda=GLOBAL_LAMBDA,
                fit_intercept=fit_intercept,
            ).fit(self.X, self.y, sample_weight=sw)

            for cid in plain.cids_:
                np.testing.assert_allclose(
                    grouped.cid_coefs_[cid],
                    plain.cid_coefs_[cid],
                    rtol=0,
                    atol=LIMIT_ATOL,
                    err_msg=f"{label} {cid}",
                )
            np.testing.assert_allclose(
                grouped.coef_, plain.coef_, rtol=0, atol=LIMIT_ATOL, err_msg=label
            )
            if fit_intercept:
                np.testing.assert_allclose(
                    grouped.intercept_,
                    plain.intercept_,
                    rtol=0,
                    atol=LIMIT_ATOL,
                    err_msg=label,
                )
                global_row = np.concatenate([[grouped.intercept_], grouped.coef_])
            else:
                global_row = grouped.coef_
            # Group rows collapse onto the global row.
            for name in grouped.groups_:
                np.testing.assert_allclose(
                    grouped.group_coefs_[name],
                    global_row,
                    rtol=0,
                    atol=LIMIT_ATOL,
                    err_msg=f"{label} {name}",
                )

    def test_large_local_lambda_collapses_cids_onto_groups(self):
        for fit_intercept, weighted in VARIANTS:
            label = variant_label(fit_intercept, weighted)
            sw = self.weights(weighted)

            # Guard against a vacuous test: at the moderate penalty local rows are
            # clearly apart from their group rows.
            ref = Reference(self.X, self.y, sw, fit_intercept, GROUPS)
            cid_rows, group_rows, _ = ref.split(ref.solve())
            self.assertGreater(
                max(
                    np.max(np.abs(cid_rows[c] - group_rows[CID_TO_GROUP[c]]))
                    for c in cid_rows
                ),
                10 * LIMIT_ATOL,
                msg=label,
            )

            model = grouped_model(
                fit_intercept=fit_intercept, local_lambda=LIMIT_LAMBDA
            ).fit(self.X, self.y, sample_weight=sw)
            for cid in model.cids_:
                np.testing.assert_allclose(
                    model.cid_coefs_[cid],
                    model.group_coefs_[model.cid_to_group_[cid]],
                    rtol=0,
                    atol=LIMIT_ATOL,
                    err_msg=f"{label} {cid}",
                )

    # ----------------------------------------------------------------------------------
    # groups=None path
    # ----------------------------------------------------------------------------------

    def test_group_lambda_has_no_effect_without_groups(self):
        for fit_intercept, weighted in VARIANTS:
            label = variant_label(fit_intercept, weighted)
            sw = self.weights(weighted)
            fits = [
                GlobalLocalRegression(
                    local_lambda=LOCAL_LAMBDA,
                    global_lambda=GLOBAL_LAMBDA,
                    fit_intercept=fit_intercept,
                    groups=None,
                    group_lambda=group_lambda,
                ).fit(self.X, self.y, sample_weight=sw)
                for group_lambda in [0.0, GROUP_LAMBDA, 1e6]
            ]
            first = fits[0]
            for other in fits[1:]:
                self.assertEqual(other.cids_, first.cids_, msg=label)
                for cid in first.cids_:
                    np.testing.assert_array_equal(
                        other.cid_coefs_[cid], first.cid_coefs_[cid], err_msg=label
                    )
                np.testing.assert_array_equal(other.coef_, first.coef_, err_msg=label)
                self.assertEqual(other.intercept_, first.intercept_, msg=label)
            # The grouped-only attributes are not set in this path.
            for model in fits:
                for attr in ["groups_", "n_groups_", "cid_to_group_", "group_coefs_"]:
                    self.assertFalse(hasattr(model, attr), msg=f"{label} {attr}")

    # ----------------------------------------------------------------------------------
    # positive=True
    # ----------------------------------------------------------------------------------

    def test_positive_constrains_slopes_of_every_row(self):
        for fit_intercept, weighted in VARIANTS:
            label = variant_label(fit_intercept, weighted)
            sw = self.weights(weighted)
            ref = Reference(self.X, self.y, sw, fit_intercept, GROUPS)

            # Guard: unconstrained, some slope is clearly negative, so the bounds bind.
            unconstrained = ref.solve().reshape(ref.n_rows, ref.n_features)
            unconstrained_slopes = (
                unconstrained[:, 1:] if fit_intercept else unconstrained
            )
            self.assertLess(unconstrained_slopes.min(), -0.1, msg=label)

            model = grouped_model(fit_intercept=fit_intercept, positive=True).fit(
                self.X, self.y, sample_weight=sw
            )
            theta = ref.theta_from_model(model)
            rows = theta.reshape(ref.n_rows, ref.n_features)

            # Every slope column of every row (local, group, global) is >= 0.
            slopes = rows[:, 1:] if fit_intercept else rows
            self.assertTrue(np.all(slopes >= 0), msg=f"{label}: {slopes.min()}")

            # Intercepts are unconstrained: here every one of them is negative.
            if fit_intercept:
                self.assertLess(model.intercept_, 0, msg=label)
                self.assertTrue(np.all(rows[:, 0] < 0), msg=label)

            # Optimality: KKT conditions of the reference objective under these bounds.
            pg = projected_gradient(
                theta, ref.gradient(theta), ref.lower_bounds(positive=True)
            )
            self.assertLessEqual(np.max(np.abs(pg)), KKT_ATOL, msg=label)

    # ----------------------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------------------

    def test_groups_not_a_dict_raises_type_error(self):
        for groups in [list(GROUPS.values()), "commodity"]:
            # No validation at construction (sklearn convention), only in fit.
            model = grouped_model(groups=groups)
            with self.assertRaises(TypeError):
                model.fit(self.X, self.y)

    def test_str_group_value_raises_type_error(self):
        # A str is not a list-like of cids, even though it is iterable.
        groups = {
            "commodity": ["NOK", "AUD", "CAD"],
            "asia": ["KRW", "JPY"],
            "latam": "BRL",
        }
        with self.assertRaises(TypeError):
            grouped_model(groups=groups).fit(self.X, self.y)

    def test_cid_in_two_groups_raises_value_error(self):
        groups = {
            "commodity": ["NOK", "AUD", "CAD", "JPY"],
            "asia": ["KRW", "JPY"],
            "latam": ["BRL"],
        }
        with self.assertRaises(ValueError):
            grouped_model(groups=groups).fit(self.X, self.y)

    def test_training_cid_in_no_group_raises_value_error_naming_it(self):
        groups = {"commodity": ["NOK", "AUD", "CAD"], "asia": ["JPY"]}
        with self.assertRaises(ValueError) as ctx:
            grouped_model(groups=groups).fit(self.X, self.y)
        message = str(ctx.exception)
        for cid in ["BRL", "KRW"]:
            self.assertIn(cid, message)

    def test_cids_absent_from_data_are_ignored(self):
        groups = {
            "commodity": ["NOK", "AUD", "CAD", "NZD"],
            "asia": ["KRW", "JPY", "THB"],
            "latam": ["BRL", "MXN"],
            # Only absent cids: dropped from the fitted model.
            "europe": ["GBP", "EUR"],
        }
        model = grouped_model(groups=groups).fit(self.X, self.y)
        self.assertEqual(model.groups_, SORTED_GROUPS)
        self.assertEqual(model.n_groups_, len(SORTED_GROUPS))
        self.assertEqual(set(model.group_coefs_), set(SORTED_GROUPS))
        self.assertNotIn("europe", model.group_coefs_)
        self.assertEqual(model.cid_to_group_, CID_TO_GROUP)
        # Same model as without the absent cids.
        ref = Reference(self.X, self.y, None, True, GROUPS)
        self.assert_matches_reference(model, ref, ref.solve(), FIT_ATOL)

    # ----------------------------------------------------------------------------------
    # Fitted attributes
    # ----------------------------------------------------------------------------------

    def test_fitted_attributes(self):
        n_x = len(FEATURES)
        for fit_intercept, weighted in VARIANTS:
            label = variant_label(fit_intercept, weighted)
            sw = self.weights(weighted)
            model = grouped_model(fit_intercept=fit_intercept).fit(
                self.X, self.y, sample_weight=sw
            )
            n_features = n_x + 1 if fit_intercept else n_x

            # Attributes shared with the groups=None path.
            self.assertEqual(model.cids_, sorted(CID_LENGTHS), msg=label)
            self.assertEqual(model.n_cids_, len(CID_LENGTHS), msg=label)
            self.assertEqual(model.n_features_, n_features, msg=label)
            self.assertEqual(set(model.cid_coefs_), set(CID_LENGTHS), msg=label)
            for cid in CID_LENGTHS:
                self.assertEqual(model.cid_coefs_[cid].shape, (n_features,), msg=label)
            n_cids = len(CID_LENGTHS)
            self.assertEqual(
                model.XtX_cid_.shape, (n_cids, n_features, n_features), msg=label
            )
            self.assertEqual(model.Xty_cid_.shape, (n_cids, n_features), msg=label)
            self.assertEqual(model.yty_cid_.shape, (n_cids,), msg=label)
            # The training data itself is not kept on the estimator.
            for name in ["X_cid_", "y_cid_", "Xy_cid_weights_"]:
                self.assertFalse(hasattr(model, name), msg=f"{label} {name}")
            self.assertEqual(np.shape(model.coef_), (n_x,), msg=label)
            if fit_intercept:
                self.assertEqual(np.ndim(model.intercept_), 0, msg=label)
            else:
                self.assertIsNone(model.intercept_, msg=label)

            # Grouped-only attributes.
            self.assertIsInstance(model.groups_, list, msg=label)
            self.assertEqual(model.groups_, SORTED_GROUPS, msg=label)
            self.assertEqual(model.n_groups_, len(SORTED_GROUPS), msg=label)
            self.assertEqual(model.cid_to_group_, CID_TO_GROUP, msg=label)
            self.assertEqual(set(model.group_coefs_), set(SORTED_GROUPS), msg=label)
            for name in SORTED_GROUPS:
                self.assertEqual(
                    model.group_coefs_[name].shape, (n_features,), msg=label
                )

    # ----------------------------------------------------------------------------------
    # sklearn compatibility
    # ----------------------------------------------------------------------------------

    def test_get_params_and_clone(self):
        model = grouped_model()
        # Stored verbatim, no copy.
        self.assertIs(model.groups, GROUPS)
        params = model.get_params()
        self.assertIs(params["groups"], GROUPS)
        self.assertEqual(params["group_lambda"], GROUP_LAMBDA)

        model.fit(self.X, self.y)
        cloned = clone(model)
        self.assertIsNot(cloned, model)
        cloned_params = cloned.get_params()
        self.assertEqual(cloned_params["groups"], GROUPS)
        self.assertEqual(cloned_params["group_lambda"], GROUP_LAMBDA)
        self.assertFalse(hasattr(cloned, "group_coefs_"))
        cloned.fit(self.X, self.y)
        self.assertEqual(cloned.groups_, SORTED_GROUPS)

    # ----------------------------------------------------------------------------------
    # predict
    # ----------------------------------------------------------------------------------

    def test_predict_uses_each_cids_local_row(self):
        # Rows of a cid unseen in training are predicted as 0.
        unseen_dates = pd.bdate_range(end="2023-12-29", periods=5)
        unseen = pd.DataFrame(
            np.ones((5, len(FEATURES))),
            index=pd.MultiIndex.from_product(
                [["ZAR"], unseen_dates], names=["cid", "real_date"]
            ),
            columns=FEATURES,
        )
        X_new = pd.concat([self.X, unseen])
        cid_index = X_new.index.get_level_values(0)

        for fit_intercept in [True, False]:
            model = grouped_model(fit_intercept=fit_intercept).fit(self.X, self.y)
            preds = model.predict(X_new)
            self.assertEqual(np.shape(preds), (len(X_new),))
            for cid in sorted(CID_LENGTHS):
                mask = np.asarray(cid_index == cid)
                X_cid = X_new.values[mask]
                if fit_intercept:
                    X_cid = np.column_stack([np.ones(len(X_cid)), X_cid])
                np.testing.assert_allclose(
                    preds[mask], X_cid @ model.cid_coefs_[cid], rtol=1e-12, atol=1e-12
                )
            np.testing.assert_array_equal(preds[np.asarray(cid_index == "ZAR")], 0)


class TestNoGlobalRegression(PanelTestCase):
    """
    shrink_to_global=False, with and without groups, held to the reference and to the
    interface specification.
    """

    # ----------------------------------------------------------------------------------
    # Fit, loss and gradient
    # ----------------------------------------------------------------------------------

    def test_reference_is_self_consistent(self):
        # The analytic reference gradient matches finite differences of the reference
        # objective, and vanishes at the closed-form solution.
        rng = np.random.default_rng(6)
        for groups in [None, GROUPS]:
            for fit_intercept, weighted in VARIANTS:
                label = groups_label(groups, fit_intercept, weighted)
                ref = Reference(
                    self.X,
                    self.y,
                    self.weights(weighted),
                    fit_intercept,
                    groups,
                    shrink_to_global=False,
                )
                theta = rng.normal(size=ref.n_rows * ref.n_features)
                np.testing.assert_allclose(
                    ref.gradient(theta),
                    finite_difference_gradient(ref.objective, theta),
                    rtol=FD_TOL,
                    atol=FD_TOL,
                    err_msg=label,
                )
                np.testing.assert_allclose(
                    ref.gradient(ref.solve()), 0, atol=1e-10, err_msg=label
                )

    def test_fit_without_groups_matches_per_cid_ridge(self):
        # Independent of the reference: each cid is its own ridge regression with an
        # unpenalised intercept. Scaling a cid's loss by sum(w_i) gives sklearn's
        # objective with alpha = local_lambda * sum(w_i).
        cid_index = self.X.index.get_level_values(0)
        for fit_intercept, weighted in VARIANTS:
            label = variant_label(fit_intercept, weighted)
            sw = self.weights(weighted)
            model = no_global_model(fit_intercept=fit_intercept).fit(
                self.X, self.y, sample_weight=sw
            )
            for cid in sorted(CID_LENGTHS):
                mask = np.asarray(cid_index == cid)
                w_i = np.ones(mask.sum()) if sw is None else sw[mask]
                ridge = Ridge(
                    alpha=LOCAL_LAMBDA * w_i.sum(), fit_intercept=fit_intercept
                ).fit(self.X.values[mask], self.y.values[mask], sample_weight=w_i)
                if fit_intercept:
                    expected = np.concatenate([[ridge.intercept_], ridge.coef_])
                else:
                    expected = ridge.coef_
                np.testing.assert_allclose(
                    model.cid_coefs_[cid],
                    expected,
                    rtol=0,
                    atol=FIT_ATOL,
                    err_msg=f"{label} {cid}",
                )

    def test_fit_matches_closed_form(self):
        for groups in [None, GROUPS]:
            for fit_intercept, weighted in VARIANTS:
                label = groups_label(groups, fit_intercept, weighted)
                sw = self.weights(weighted)
                ref = Reference(
                    self.X, self.y, sw, fit_intercept, groups, shrink_to_global=False
                )
                model = no_global_model(
                    groups=groups, fit_intercept=fit_intercept
                ).fit(self.X, self.y, sample_weight=sw)
                self.assert_matches_reference(model, ref, ref.solve(), FIT_ATOL, label)

    def test_loss_matches_reference_objective(self):
        # Pins the parameter-vector layout: local rows (sorted cids), then group rows
        # (sorted group names) when grouped, and no global row.
        rng = np.random.default_rng(7)
        for groups in [None, GROUPS]:
            for fit_intercept, weighted in VARIANTS:
                label = groups_label(groups, fit_intercept, weighted)
                sw = self.weights(weighted)
                ref = Reference(
                    self.X, self.y, sw, fit_intercept, groups, shrink_to_global=False
                )
                model = no_global_model(
                    groups=groups, fit_intercept=fit_intercept
                ).fit(self.X, self.y, sample_weight=sw)
                theta = rng.normal(size=ref.n_rows * ref.n_features)
                value = model.loss(theta)
                self.assertEqual(np.ndim(value), 0, msg=label)
                np.testing.assert_allclose(
                    value, ref.objective(theta), rtol=1e-12, err_msg=label
                )

    def test_loss_derivative_matches_finite_differences(self):
        rng = np.random.default_rng(8)
        for groups in [None, GROUPS]:
            for fit_intercept, weighted in VARIANTS:
                label = groups_label(groups, fit_intercept, weighted)
                sw = self.weights(weighted)
                model = no_global_model(
                    groups=groups, fit_intercept=fit_intercept
                ).fit(self.X, self.y, sample_weight=sw)
                n_rows = model.n_cids_ + (0 if groups is None else model.n_groups_)
                n_params = n_rows * model.n_features_
                theta = rng.normal(size=n_params)
                grad = model.loss_derivative(theta)
                self.assertEqual(np.shape(grad), (n_params,), msg=label)
                np.testing.assert_allclose(
                    grad,
                    finite_difference_gradient(model.loss, theta),
                    rtol=FD_TOL,
                    atol=FD_TOL,
                    err_msg=label,
                )

    def test_global_lambda_has_no_effect(self):
        for groups in [None, GROUPS]:
            for fit_intercept, weighted in VARIANTS:
                label = groups_label(groups, fit_intercept, weighted)
                sw = self.weights(weighted)
                fits = [
                    no_global_model(
                        groups=groups,
                        fit_intercept=fit_intercept,
                        global_lambda=global_lambda,
                    ).fit(self.X, self.y, sample_weight=sw)
                    for global_lambda in [0.0, GLOBAL_LAMBDA, 1e6]
                ]
                first = fits[0]
                for other in fits[1:]:
                    for cid in first.cids_:
                        np.testing.assert_array_equal(
                            other.cid_coefs_[cid], first.cid_coefs_[cid], err_msg=label
                        )
                    if groups is not None:
                        for name in first.groups_:
                            np.testing.assert_array_equal(
                                other.group_coefs_[name],
                                first.group_coefs_[name],
                                err_msg=label,
                            )

    # ----------------------------------------------------------------------------------
    # positive=True
    # ----------------------------------------------------------------------------------

    def test_positive_constrains_slopes_of_every_row(self):
        for groups in [None, GROUPS]:
            for fit_intercept, weighted in VARIANTS:
                label = groups_label(groups, fit_intercept, weighted)
                sw = self.weights(weighted)
                ref = Reference(
                    self.X, self.y, sw, fit_intercept, groups, shrink_to_global=False
                )

                # Guard: unconstrained, some slope is clearly negative, so the bounds
                # bind.
                unconstrained = ref.solve().reshape(ref.n_rows, ref.n_features)
                unconstrained_slopes = (
                    unconstrained[:, 1:] if fit_intercept else unconstrained
                )
                self.assertLess(unconstrained_slopes.min(), -0.1, msg=label)

                model = no_global_model(
                    groups=groups, fit_intercept=fit_intercept, positive=True
                ).fit(self.X, self.y, sample_weight=sw)
                theta = ref.theta_from_model(model)
                rows = theta.reshape(ref.n_rows, ref.n_features)

                # Every slope column of every row (local, and group if grouped) is >= 0.
                slopes = rows[:, 1:] if fit_intercept else rows
                self.assertTrue(np.all(slopes >= 0), msg=f"{label}: {slopes.min()}")

                # Intercepts are unconstrained: here every one of them is negative.
                if fit_intercept:
                    self.assertTrue(np.all(rows[:, 0] < 0), msg=label)

                # Optimality: KKT conditions of the reference objective under these
                # bounds.
                pg = projected_gradient(
                    theta, ref.gradient(theta), ref.lower_bounds(positive=True)
                )
                self.assertLessEqual(np.max(np.abs(pg)), KKT_ATOL, msg=label)

    # ----------------------------------------------------------------------------------
    # Fitted attributes and sklearn compatibility
    # ----------------------------------------------------------------------------------

    def test_fitted_attributes(self):
        n_x = len(FEATURES)
        for groups in [None, GROUPS]:
            for fit_intercept, weighted in VARIANTS:
                label = groups_label(groups, fit_intercept, weighted)
                model = no_global_model(
                    groups=groups, fit_intercept=fit_intercept
                ).fit(self.X, self.y, sample_weight=self.weights(weighted))
                n_features = n_x + 1 if fit_intercept else n_x

                self.assertEqual(model.cids_, sorted(CID_LENGTHS), msg=label)
                self.assertEqual(set(model.cid_coefs_), set(CID_LENGTHS), msg=label)
                for cid in CID_LENGTHS:
                    self.assertEqual(
                        model.cid_coefs_[cid].shape, (n_features,), msg=label
                    )

                # There is no global row to report.
                for name in ["coef_", "intercept_"]:
                    self.assertFalse(hasattr(model, name), msg=f"{label} {name}")

                grouped_attrs = ["groups_", "n_groups_", "cid_to_group_", "group_coefs_"]
                if groups is None:
                    for name in grouped_attrs:
                        self.assertFalse(hasattr(model, name), msg=f"{label} {name}")
                else:
                    self.assertEqual(model.groups_, SORTED_GROUPS, msg=label)
                    self.assertEqual(model.cid_to_group_, CID_TO_GROUP, msg=label)
                    self.assertEqual(
                        set(model.group_coefs_), set(SORTED_GROUPS), msg=label
                    )
                    for name in SORTED_GROUPS:
                        self.assertEqual(
                            model.group_coefs_[name].shape, (n_features,), msg=label
                        )

    def test_get_params_and_clone(self):
        self.assertTrue(GlobalLocalRegression().get_params()["shrink_to_global"])

        model = no_global_model(groups=GROUPS)
        self.assertFalse(model.get_params()["shrink_to_global"])
        model.fit(self.X, self.y)
        cloned = clone(model)
        self.assertFalse(cloned.get_params()["shrink_to_global"])
        self.assertFalse(hasattr(cloned, "cid_coefs_"))
        cloned.fit(self.X, self.y)
        for cid in model.cids_:
            np.testing.assert_array_equal(cloned.cid_coefs_[cid], model.cid_coefs_[cid])


class TestCidSummaries(PanelTestCase):
    """
    The per-cross-section weighted moments that `fit` stores and that the loss is
    evaluated from: their definition, independence of row order, and target shapes.
    """

    SUMMARIES = ["XtX_cid_", "Xty_cid_", "yty_cid_"]

    def fitted_rows(self, model):
        """Every fitted coefficient row: local, group (if any), then global."""
        rows = [model.cid_coefs_[cid] for cid in model.cids_]
        if model.groups is not None:
            rows += [model.group_coefs_[name] for name in model.groups_]
        rows.append(model.coef_)
        if model.fit_intercept:
            rows.append([model.intercept_])
        return np.concatenate(rows)

    def test_summaries_match_definitions(self):
        for fit_intercept, weighted in VARIANTS:
            label = variant_label(fit_intercept, weighted)
            sw = self.weights(weighted)
            ref = Reference(self.X, self.y, sw, fit_intercept)
            model = GlobalLocalRegression(fit_intercept=fit_intercept).fit(
                self.X, self.y, sample_weight=sw
            )
            self.assertEqual(model.cids_, ref.cids, msg=label)
            for i, (X_i, y_i, w_i) in enumerate(ref.blocks):
                cid_label = f"{label} {ref.cids[i]}"
                np.testing.assert_allclose(
                    model.XtX_cid_[i],
                    X_i.T @ (w_i[:, None] * X_i) / np.sum(w_i),
                    rtol=1e-12,
                    err_msg=cid_label,
                )
                np.testing.assert_allclose(
                    model.Xty_cid_[i],
                    X_i.T @ (w_i * y_i) / np.sum(w_i),
                    rtol=1e-12,
                    err_msg=cid_label,
                )
                np.testing.assert_allclose(
                    model.yty_cid_[i],
                    np.sum(w_i * y_i**2) / np.sum(w_i),
                    rtol=1e-12,
                    err_msg=cid_label,
                )

    def test_row_order_does_not_change_fit(self):
        # Interleave cids and unsort dates; X, y and weights stay aligned by row.
        perm = np.random.default_rng(5).permutation(len(self.y))
        X_perm, y_perm = self.X.iloc[perm], self.y.iloc[perm]
        for groups in [None, GROUPS]:
            for positive in [False, True]:
                for fit_intercept, weighted in VARIANTS:
                    label = (
                        f"groups={groups is not None}, positive={positive}, "
                        f"{variant_label(fit_intercept, weighted)}"
                    )
                    sw = self.weights(weighted)
                    params = dict(
                        local_lambda=LOCAL_LAMBDA,
                        global_lambda=GLOBAL_LAMBDA,
                        group_lambda=GROUP_LAMBDA,
                        groups=groups,
                        positive=positive,
                        fit_intercept=fit_intercept,
                    )
                    model = GlobalLocalRegression(**params).fit(
                        self.X, self.y, sample_weight=sw
                    )
                    model_perm = GlobalLocalRegression(**params).fit(
                        X_perm,
                        y_perm,
                        sample_weight=None if sw is None else sw[perm],
                    )
                    self.assertEqual(model_perm.cids_, model.cids_, msg=label)
                    for name in self.SUMMARIES:
                        np.testing.assert_allclose(
                            getattr(model_perm, name),
                            getattr(model, name),
                            rtol=1e-12,
                            atol=1e-14,
                            err_msg=f"{label} {name}",
                        )
                    np.testing.assert_allclose(
                        self.fitted_rows(model_perm),
                        self.fitted_rows(model),
                        rtol=0,
                        atol=1e-10,
                        err_msg=label,
                    )

    def test_single_column_dataframe_target(self):
        y_frame = self.y.to_frame()
        for fit_intercept, weighted in VARIANTS:
            label = variant_label(fit_intercept, weighted)
            sw = self.weights(weighted)
            model = GlobalLocalRegression(fit_intercept=fit_intercept).fit(
                self.X, self.y, sample_weight=sw
            )
            model_frame = GlobalLocalRegression(fit_intercept=fit_intercept).fit(
                self.X, y_frame, sample_weight=sw
            )
            for name in self.SUMMARIES:
                np.testing.assert_array_equal(
                    getattr(model_frame, name),
                    getattr(model, name),
                    err_msg=f"{label} {name}",
                )
            np.testing.assert_array_equal(
                self.fitted_rows(model_frame), self.fitted_rows(model), err_msg=label
            )

        # More than one target column is rejected.
        y_two = pd.concat([self.y, self.y], axis=1)
        with self.assertRaises(ValueError):
            GlobalLocalRegression().fit(self.X, y_two)


# --------------------------------------------------------------------------------------
# Correlation-clustered groups (groups="cluster")
# --------------------------------------------------------------------------------------

# Block of each cid. Every cid's target is its block's common factor plus small noise,
# and the two factors are independent. The blocks interleave in sorted-cid order, so
# labels that do not follow first appearance along the sorted cids show up.
CLUSTER_BLOCKS = {"AUD": 0, "CAD": 1, "EUR": 0, "GBP": 1, "JPY": 0, "USD": 1}
CLUSTER_CIDS = sorted(CLUSTER_BLOCKS)
CLUSTER_FEATURES = ["CRY", "GROWTH"]
CLUSTER_N_DATES = 60

# Group 0 is the block holding cids_[0] ("AUD"); members are listed in cids_ order.
EXPECTED_CLUSTERS = {0: ["AUD", "EUR", "JPY"], 1: ["CAD", "GBP", "USD"]}
EXPECTED_CID_TO_GROUP = {
    "AUD": 0,
    "CAD": 1,
    "EUR": 0,
    "GBP": 1,
    "JPY": 0,
    "USD": 1,
}

# The same partition as a user-supplied dict with string group names.
BLOCK_GROUPS = {"bloc_b": ["USD", "CAD", "GBP"], "bloc_a": ["JPY", "AUD", "EUR"]}

LINKAGES = ["average", "complete", "single", "ward"]

# The key invariant asks for identical coefficients up to floating-point tolerance.
INVARIANT_ATOL = 1e-10

# Linkage matrices recomputed from the contract's recipe agree up to rounding in the
# distances; merge order and cluster sizes agree exactly on this panel.
LINKAGE_RTOL = 1e-10
LINKAGE_ATOL = 1e-12


def make_clustered_panel(seed=0, n_dates=CLUSTER_N_DATES):
    """
    Build a small balanced panel whose targets follow CLUSTER_BLOCKS.

    Returns
    -------
    X : pd.DataFrame
        Features, multi-indexed by `cid` and `real_date`, sorted by cid. "CRY" is a
        noisy copy of the cid's block factor, "GROWTH" is pure noise.
    y : pd.Series
        Targets with the same index as X: block factor + N(0, 0.2^2) noise.
    sample_weight : np.ndarray
        Strictly positive sample weights, one per row of X.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2023-12-29", periods=n_dates)
    factors = rng.normal(size=(n_dates, 2))
    X_blocks = []
    y_blocks = []
    for cid in CLUSTER_CIDS:
        factor = factors[:, CLUSTER_BLOCKS[cid]]
        index = pd.MultiIndex.from_product([[cid], dates], names=["cid", "real_date"])
        ftrs = np.column_stack(
            [factor + rng.normal(scale=0.5, size=n_dates), rng.normal(size=n_dates)]
        )
        target = factor + rng.normal(scale=0.2, size=n_dates)
        X_blocks.append(pd.DataFrame(ftrs, index=index, columns=CLUSTER_FEATURES))
        y_blocks.append(pd.Series(target, index=index, name="XR"))

    X = pd.concat(X_blocks)
    y = pd.concat(y_blocks)
    sample_weight = rng.uniform(0.5, 2.0, size=len(y))
    return X, y, sample_weight


def reference_correlation(X, y):
    """
    Contract steps 1-2: the values of `y` indexed by X's index, pivoted to dates x
    cids (sorted), pairwise Pearson correlation with pandas defaults, NaN set to 0
    and a unit diagonal.
    """
    cids = sorted(X.index.get_level_values(0).unique())
    values = np.asarray(y, dtype=float).reshape(len(y))
    wide = pd.Series(values, index=X.index).unstack(level=0).reindex(columns=cids)
    corr = wide.corr().to_numpy(copy=True)
    corr[np.isnan(corr)] = 0
    np.fill_diagonal(corr, 1)
    return corr


def reference_linkage_matrix(X, y, method):
    """
    Contract steps 1-4: linkage of the condensed distances
    d_ij = sqrt(0.5 * (1 - rho_ij)), floored at 0, with a zero diagonal.
    """
    corr = reference_correlation(X, y)
    dist = np.sqrt(np.maximum(0.5 * (1 - corr), 0))
    np.fill_diagonal(dist, 0)
    return hierarchy.linkage(squareform(dist, checks=False), method=method)


def clustered_model(**overrides):
    """Model with groups="cluster" at the module's penalty settings."""
    params = {
        "local_lambda": LOCAL_LAMBDA,
        "global_lambda": GLOBAL_LAMBDA,
        "group_lambda": GROUP_LAMBDA,
        "groups": "cluster",
    }
    params.update(overrides)
    return GlobalLocalRegression(**params)


def fit_recording_user_warnings(model, X, y, sample_weight=None):
    """Fit `model` and return the UserWarnings emitted during the fit."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y, sample_weight=sample_weight)
    return [w for w in caught if issubclass(w.category, UserWarning)]


def select_cids(X, y, cids):
    """Rows of X and y (aligned by position) whose cid is in `cids`."""
    mask = np.asarray(X.index.get_level_values(0).isin(cids))
    return X[mask], y[mask]


class TestClusteredGlobalLocalRegression(unittest.TestCase):
    """
    groups="cluster": groups formed by hierarchical clustering of the cids' target
    correlations, held to the interface contract.
    """

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.sample_weight = make_clustered_panel()

    def assert_clusters(self, model, expected_clusters, label=""):
        """
        The fitted grouping is `expected_clusters`, consistently across `clusters_`,
        `cid_to_group_`, `groups_`, `n_groups_` and `group_coefs_`, with every group
        label a Python int and cids_[0] in group 0.
        """
        self.assertEqual(model.clusters_, expected_clusters, msg=label)
        self.assertEqual(
            model.cid_to_group_,
            {cid: g for g, members in expected_clusters.items() for cid in members},
            msg=label,
        )
        self.assertEqual(model.groups_, list(range(len(expected_clusters))), msg=label)
        self.assertEqual(model.n_groups_, len(expected_clusters), msg=label)
        self.assertEqual(set(model.group_coefs_), set(expected_clusters), msg=label)
        self.assertEqual(model.cid_to_group_[model.cids_[0]], 0, msg=label)
        labels = (
            list(model.clusters_)
            + list(model.groups_)
            + list(model.cid_to_group_.values())
            + list(model.group_coefs_)
        )
        for g in labels:
            self.assertIs(type(g), int, msg=f"{label} label {g!r}")

    def assert_finite_coefficients(self, model, label=""):
        for cid in model.cids_:
            self.assertTrue(np.all(np.isfinite(model.cid_coefs_[cid])), msg=label)
        for g in model.groups_:
            self.assertTrue(np.all(np.isfinite(model.group_coefs_[g])), msg=label)
        if model.shrink_to_global:
            self.assertTrue(np.all(np.isfinite(model.coef_)), msg=label)
            if model.fit_intercept:
                self.assertTrue(np.isfinite(model.intercept_), msg=label)

    def assert_linkage_matrix(self, model, expected, label=""):
        self.assertIsInstance(model.linkage_matrix_, np.ndarray, msg=label)
        np.testing.assert_allclose(
            model.linkage_matrix_,
            expected,
            rtol=LINKAGE_RTOL,
            atol=LINKAGE_ATOL,
            err_msg=label,
        )

    # ----------------------------------------------------------------------------------
    # Panel guard
    # ----------------------------------------------------------------------------------

    def test_panel_has_block_structure(self):
        # Guard against a vacuous fixture: targets are strongly correlated within a
        # block and nearly uncorrelated across blocks.
        corr = reference_correlation(self.X, self.y)
        blocks = np.array([CLUSTER_BLOCKS[cid] for cid in CLUSTER_CIDS])
        same_block = np.equal.outer(blocks, blocks)
        off_diagonal = ~np.eye(len(blocks), dtype=bool)
        self.assertGreater(corr[same_block & off_diagonal].min(), 0.9)
        self.assertLess(np.abs(corr[~same_block]).max(), 0.3)

    # ----------------------------------------------------------------------------------
    # Clustering
    # ----------------------------------------------------------------------------------

    def test_recovers_blocks(self):
        model = clustered_model(n_clusters=2).fit(self.X, self.y)
        self.assertEqual(model.cids_, CLUSTER_CIDS)
        self.assertEqual(model.cid_to_group_, EXPECTED_CID_TO_GROUP)
        self.assertEqual(model.groups_, [0, 1])
        self.assertEqual(model.n_groups_, 2)
        self.assert_clusters(model, EXPECTED_CLUSTERS)

    def test_default_parameters_recover_blocks(self):
        # Defaults: n_clusters=2, linkage="average".
        model = GlobalLocalRegression(groups="cluster").fit(self.X, self.y)
        self.assert_clusters(model, EXPECTED_CLUSTERS)
        self.assert_linkage_matrix(
            model, reference_linkage_matrix(self.X, self.y, "average")
        )

    def test_linkage_matrix(self):
        model = clustered_model().fit(self.X, self.y)
        n_cids = len(CLUSTER_CIDS)
        self.assertEqual(model.linkage_matrix_.shape, (n_cids - 1, 4))
        self.assert_linkage_matrix(
            model, reference_linkage_matrix(self.X, self.y, "average")
        )

    def test_each_linkage_recovers_blocks(self):
        for method in LINKAGES:
            label = f"linkage={method}"
            model = clustered_model(linkage=method).fit(self.X, self.y)
            self.assert_clusters(model, EXPECTED_CLUSTERS, label)
            self.assert_linkage_matrix(
                model, reference_linkage_matrix(self.X, self.y, method), label
            )

    def test_numpy_integer_n_clusters_is_accepted(self):
        for n_clusters in [np.int64(2), np.int32(2)]:
            label = f"n_clusters={n_clusters!r}"
            model = clustered_model(n_clusters=n_clusters).fit(self.X, self.y)
            self.assert_clusters(model, EXPECTED_CLUSTERS, label)

    def test_n_clusters_above_n_cids_is_capped_with_warning(self):
        singletons = {k: [cid] for k, cid in enumerate(CLUSTER_CIDS)}
        for n_clusters in [len(CLUSTER_CIDS) + 1, 100]:
            label = f"n_clusters={n_clusters}"
            model = clustered_model(n_clusters=n_clusters)
            with self.assertWarns(UserWarning, msg=label):
                model.fit(self.X, self.y)
            self.assert_clusters(model, singletons, label)
            self.assertEqual(
                model.linkage_matrix_.shape, (len(CLUSTER_CIDS) - 1, 4), msg=label
            )

    def test_n_clusters_equal_to_n_cids_gives_singletons_without_warning(self):
        model = clustered_model(n_clusters=len(CLUSTER_CIDS))
        caught = fit_recording_user_warnings(model, self.X, self.y)
        self.assertEqual(caught, [])
        self.assert_clusters(model, {k: [cid] for k, cid in enumerate(CLUSTER_CIDS)})

    def test_one_cluster_holds_every_cid(self):
        model = clustered_model(n_clusters=1)
        caught = fit_recording_user_warnings(model, self.X, self.y)
        self.assertEqual(caught, [])
        self.assert_clusters(model, {0: CLUSTER_CIDS})
        self.assertEqual(model.linkage_matrix_.shape, (len(CLUSTER_CIDS) - 1, 4))

    def test_single_cid_panel(self):
        X_one, y_one = select_cids(self.X, self.y, ["GBP"])

        # n_clusters (default 2) exceeds the one training cid: capped with a warning.
        model = clustered_model()
        with self.assertWarns(UserWarning):
            model.fit(X_one, y_one)
        self.assertEqual(model.cids_, ["GBP"])
        self.assert_clusters(model, {0: ["GBP"]})
        self.assertIsNone(model.linkage_matrix_)
        self.assert_finite_coefficients(model)

        # n_clusters=1 needs no cap.
        model = clustered_model(n_clusters=1)
        caught = fit_recording_user_warnings(model, X_one, y_one)
        self.assertEqual(caught, [])
        self.assert_clusters(model, {0: ["GBP"]})
        self.assertIsNone(model.linkage_matrix_)
        self.assert_finite_coefficients(model)

    def test_unbalanced_panel_with_cids_sharing_no_dates(self):
        # AUD keeps only the first half of the dates and CAD only the second half.
        cid_index = self.X.index.get_level_values(0)
        date_index = self.X.index.get_level_values(1)
        split = sorted(date_index.unique())[CLUSTER_N_DATES // 2]
        drop = ((cid_index == "AUD") & (date_index >= split)) | (
            (cid_index == "CAD") & (date_index < split)
        )
        keep = np.asarray(~drop)
        X_unb, y_unb = self.X[keep], self.y[keep]
        aud_dates = set(X_unb.xs("AUD", level=0).index)
        cad_dates = set(X_unb.xs("CAD", level=0).index)
        self.assertTrue(aud_dates)
        self.assertTrue(cad_dates)
        self.assertEqual(aud_dates & cad_dates, set())

        for shrink_to_global in [True, False]:
            label = f"shrink_to_global={shrink_to_global}"
            model = clustered_model(shrink_to_global=shrink_to_global).fit(
                X_unb, y_unb
            )
            self.assert_finite_coefficients(model, label)
            self.assert_linkage_matrix(
                model, reference_linkage_matrix(X_unb, y_unb, "average"), label
            )
            self.assert_clusters(model, EXPECTED_CLUSTERS, label)

    def test_zero_variance_targets_and_fewer_clusters_than_requested(self):
        # Every cid has a constant target, so every correlation is NaN -> 0 and every
        # distance is sqrt(0.5). With all merge heights tied, fcluster returns fewer
        # clusters than requested, which is accepted silently.
        X_four, _ = select_cids(self.X, self.y, ["AUD", "CAD", "EUR", "GBP"])
        constants = {"AUD": 1.0, "CAD": 2.0, "EUR": -3.0, "GBP": 0.5}
        y_const = pd.Series(
            [constants[cid] for cid in X_four.index.get_level_values(0)],
            index=X_four.index,
            name="XR",
        )
        expected_linkage = reference_linkage_matrix(X_four, y_const, "average")
        np.testing.assert_allclose(expected_linkage[:, 2], np.sqrt(0.5))
        # Guard: the tie really yields fewer clusters than the 3 requested.
        raw_labels = hierarchy.fcluster(expected_linkage, t=3, criterion="maxclust")
        self.assertEqual(len(np.unique(raw_labels)), 1)

        model = clustered_model(n_clusters=3)
        caught = fit_recording_user_warnings(model, X_four, y_const)
        self.assertEqual(caught, [])
        self.assert_clusters(model, {0: ["AUD", "CAD", "EUR", "GBP"]})
        self.assert_linkage_matrix(model, expected_linkage)
        self.assert_finite_coefficients(model)

    def test_sample_weight_does_not_change_clusters(self):
        base = clustered_model().fit(self.X, self.y)
        # Extreme weights, which would move any weighted correlation.
        extreme = np.where(np.arange(len(self.y)) % 3 == 0, 1e3, 1e-3)
        for sw in [self.sample_weight, extreme]:
            weighted = clustered_model().fit(self.X, self.y, sample_weight=sw)
            self.assertEqual(weighted.clusters_, base.clusters_)
            np.testing.assert_array_equal(
                weighted.linkage_matrix_, base.linkage_matrix_
            )

    def test_target_is_aligned_with_X_by_row_position(self):
        # The returns panel takes y's values in X's row order and indexes them by X's
        # index, whatever y's own index or the row order.
        base = clustered_model().fit(self.X, self.y)
        perm = np.random.default_rng(9).permutation(len(self.y))
        X_perm, y_perm = self.X.iloc[perm], self.y.iloc[perm]
        # Values aligned with X_perm by position, index labels scrambled.
        y_scrambled = pd.Series(y_perm.values, index=self.X.index, name="XR")
        cases = {
            "single-column frame": (self.X, self.y.to_frame()),
            "range index": (self.X, self.y.reset_index(drop=True)),
            "permuted rows": (X_perm, y_perm),
            "permuted rows, range-index frame": (
                X_perm,
                y_perm.reset_index(drop=True).to_frame(),
            ),
            "permuted rows, scrambled y index": (X_perm, y_scrambled),
        }
        for label, (X_case, y_case) in cases.items():
            model = clustered_model().fit(X_case, y_case)
            self.assert_clusters(model, EXPECTED_CLUSTERS, label)
            self.assert_linkage_matrix(model, base.linkage_matrix_, label)

    # ----------------------------------------------------------------------------------
    # Key invariant: same fit as the equivalent dict
    # ----------------------------------------------------------------------------------

    def test_coefficients_match_equivalent_dict_groups(self):
        for shrink_to_global in [True, False]:
            for fit_intercept, weighted in VARIANTS:
                label = (
                    f"shrink_to_global={shrink_to_global}, "
                    f"{variant_label(fit_intercept, weighted)}"
                )
                sw = self.sample_weight if weighted else None
                params = {
                    "fit_intercept": fit_intercept,
                    "shrink_to_global": shrink_to_global,
                }
                clustered = clustered_model(**params).fit(
                    self.X, self.y, sample_weight=sw
                )
                # Guard: the clustering is non-trivial.
                self.assertEqual(clustered.clusters_, EXPECTED_CLUSTERS, msg=label)
                explicit = clustered_model(groups=clustered.clusters_, **params).fit(
                    self.X, self.y, sample_weight=sw
                )

                self.assertEqual(clustered.cids_, explicit.cids_, msg=label)
                self.assertEqual(
                    clustered.cid_to_group_, explicit.cid_to_group_, msg=label
                )
                self.assertEqual(clustered.groups_, explicit.groups_, msg=label)
                self.assertEqual(clustered.n_groups_, explicit.n_groups_, msg=label)
                for cid in explicit.cids_:
                    np.testing.assert_allclose(
                        clustered.cid_coefs_[cid],
                        explicit.cid_coefs_[cid],
                        rtol=0,
                        atol=INVARIANT_ATOL,
                        err_msg=f"{label} cid_coefs_[{cid}]",
                    )
                self.assertEqual(
                    set(clustered.group_coefs_), set(explicit.group_coefs_), msg=label
                )
                for g in explicit.groups_:
                    np.testing.assert_allclose(
                        clustered.group_coefs_[g],
                        explicit.group_coefs_[g],
                        rtol=0,
                        atol=INVARIANT_ATOL,
                        err_msg=f"{label} group_coefs_[{g}]",
                    )
                if shrink_to_global:
                    np.testing.assert_allclose(
                        clustered.coef_,
                        explicit.coef_,
                        rtol=0,
                        atol=INVARIANT_ATOL,
                        err_msg=f"{label} coef_",
                    )
                    if fit_intercept:
                        np.testing.assert_allclose(
                            clustered.intercept_,
                            explicit.intercept_,
                            rtol=0,
                            atol=INVARIANT_ATOL,
                            err_msg=f"{label} intercept_",
                        )
                    else:
                        self.assertIsNone(clustered.intercept_, msg=label)
                else:
                    for name in ["coef_", "intercept_"]:
                        self.assertFalse(hasattr(clustered, name), msg=f"{label} {name}")

    # ----------------------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------------------

    def test_other_groups_string_raises_type_error_mentioning_cluster(self):
        for groups in ["kmeans", "clusters", "Cluster", ""]:
            # No validation at construction (sklearn convention), only in fit.
            model = clustered_model(groups=groups)
            with self.assertRaises(TypeError, msg=repr(groups)) as ctx:
                model.fit(self.X, self.y)
            self.assertIn("cluster", str(ctx.exception), msg=repr(groups))

    def test_invalid_linkage_raises_value_error(self):
        # Valid scipy methods outside the four names are rejected too.
        for method in ["centroid", "median", "weighted", "Average", "ward ", ""]:
            model = clustered_model(linkage=method)
            with self.assertRaises(ValueError, msg=repr(method)):
                model.fit(self.X, self.y)

    def test_n_clusters_below_one_raises_value_error(self):
        for n_clusters in [0, -1, np.int64(0)]:
            model = clustered_model(n_clusters=n_clusters)
            with self.assertRaises(ValueError, msg=repr(n_clusters)):
                model.fit(self.X, self.y)

    def test_non_integer_n_clusters_raises_type_error(self):
        for n_clusters in [2.5, "2", True, 2.0, np.float64(2.0), np.True_, None]:
            model = clustered_model(n_clusters=n_clusters)
            with self.assertRaises(TypeError, msg=repr(n_clusters)):
                model.fit(self.X, self.y)

    def test_cluster_parameters_ignored_without_cluster_groups(self):
        # Neither validated nor used unless groups == "cluster".
        invalid = [(0, "average"), ("2", "bogus"), (2.5, "centroid"), (True, "ward")]
        for groups in [None, BLOCK_GROUPS]:
            baseline = clustered_model(groups=groups).fit(self.X, self.y)
            for n_clusters, method in invalid:
                label = f"groups={groups}, n_clusters={n_clusters!r}, linkage={method}"
                model = clustered_model(
                    groups=groups, n_clusters=n_clusters, linkage=method
                ).fit(self.X, self.y)
                for name in ["clusters_", "linkage_matrix_"]:
                    self.assertFalse(hasattr(model, name), msg=f"{label} {name}")
                for cid in baseline.cids_:
                    np.testing.assert_array_equal(
                        model.cid_coefs_[cid], baseline.cid_coefs_[cid], err_msg=label
                    )
                np.testing.assert_array_equal(
                    model.coef_, baseline.coef_, err_msg=label
                )
                if groups is not None:
                    self.assertEqual(model.groups_, baseline.groups_, msg=label)

    # ----------------------------------------------------------------------------------
    # sklearn compatibility
    # ----------------------------------------------------------------------------------

    def test_new_parameters_follow_shrink_to_global(self):
        signature = inspect.signature(GlobalLocalRegression.__init__)
        names = list(signature.parameters)
        self.assertEqual(names[-3:], ["shrink_to_global", "n_clusters", "linkage"])
        self.assertEqual(signature.parameters["n_clusters"].default, 2)
        self.assertEqual(signature.parameters["linkage"].default, "average")

    def test_get_params_and_clone(self):
        params = GlobalLocalRegression().get_params()
        self.assertEqual(params["n_clusters"], 2)
        self.assertEqual(params["linkage"], "average")

        # Stored unchanged, without validation at construction.
        odd = GlobalLocalRegression(n_clusters="2", linkage="bogus")
        self.assertEqual(odd.n_clusters, "2")
        self.assertEqual(odd.linkage, "bogus")

        model = clustered_model(n_clusters=3, linkage="complete")
        self.assertEqual(model.n_clusters, 3)
        self.assertEqual(model.linkage, "complete")
        params = model.get_params()
        self.assertEqual(params["n_clusters"], 3)
        self.assertEqual(params["linkage"], "complete")
        self.assertEqual(params["groups"], "cluster")

        model.fit(self.X, self.y)
        cloned = clone(model)
        self.assertIsNot(cloned, model)
        cloned_params = cloned.get_params()
        self.assertEqual(cloned_params["n_clusters"], 3)
        self.assertEqual(cloned_params["linkage"], "complete")
        self.assertEqual(cloned_params["groups"], "cluster")
        for name in ["clusters_", "linkage_matrix_"]:
            self.assertFalse(hasattr(cloned, name), msg=name)
        cloned.fit(self.X, self.y)
        self.assertEqual(cloned.clusters_, model.clusters_)
        np.testing.assert_array_equal(cloned.linkage_matrix_, model.linkage_matrix_)

    # ----------------------------------------------------------------------------------
    # predict
    # ----------------------------------------------------------------------------------

    def test_predict_after_clustered_fit(self):
        for shrink_to_global in [True, False]:
            label = f"shrink_to_global={shrink_to_global}"
            model = clustered_model(shrink_to_global=shrink_to_global).fit(
                self.X, self.y
            )
            preds = model.predict(self.X)
            self.assertEqual(np.shape(preds), (len(self.X),), msg=label)
            self.assertTrue(np.all(np.isfinite(preds)), msg=label)

    # ----------------------------------------------------------------------------------
    # Duplicate (cid, real_date) rows
    # ----------------------------------------------------------------------------------

    def test_duplicate_rows_raise_value_error_naming_the_pair(self):
        # Repeated rows are appended at the end, so X and y stay aligned by position.
        position = 3
        X = pd.concat([self.X, self.X.iloc[[position]]])
        y = pd.concat([self.y, self.y.iloc[[position]]])
        cid, date = self.X.index[position]
        for shrink_to_global in [True, False]:
            label = f"shrink_to_global={shrink_to_global}"
            model = clustered_model(shrink_to_global=shrink_to_global)
            with self.assertRaises(ValueError, msg=label) as ctx:
                model.fit(X, y)
            message = str(ctx.exception)
            for part in ["cluster", "unique", "duplicate", f"{cid} {date:%Y-%m-%d}"]:
                self.assertIn(part, message, msg=label)
            # Not pandas' raw error from the pivot
            self.assertNotIn("cannot reshape", message, msg=label)

    def test_duplicate_count_is_of_distinct_pairs(self):
        # A pair that occurs three times is one duplicated pair, not two extra rows.
        cases = [([3, 3], 1), ([3, 3, 100], 2), ([3, 100, 100, 3], 2)]
        for extra_positions, expected_count in cases:
            label = f"extra rows at {extra_positions}"
            X = pd.concat([self.X, self.X.iloc[extra_positions]])
            y = pd.concat([self.y, self.y.iloc[extra_positions]])
            with self.assertRaises(ValueError, msg=label) as ctx:
                clustered_model().fit(X, y)
            self.assertRegex(
                str(ctx.exception), rf"more than once: {expected_count}\.", msg=label
            )

    def test_duplicate_listing_shows_first_five_pairs_in_index_order(self):
        # Seven pairs over several cids, repeated in reverse order, so that index order
        # differs from the order in which the repeated rows appear.
        positions = [2, 10, 65, 70, 130, 190, 250]
        X = pd.concat([self.X, self.X.iloc[positions[::-1]]])
        y = pd.concat([self.y, self.y.iloc[positions[::-1]]])
        pairs = [f"{cid} {date:%Y-%m-%d}" for cid, date in self.X.index[positions]]
        with self.assertRaises(ValueError) as ctx:
            clustered_model().fit(X, y)
        message = str(ctx.exception)
        self.assertRegex(message, r"more than once: 7\.")
        for pair in pairs[5:]:
            self.assertNotIn(pair, message)
        offsets = [message.find(pair) for pair in pairs[:5]]
        self.assertNotIn(-1, offsets)
        self.assertEqual(offsets, sorted(offsets))

    def test_duplicate_rows_still_fit_with_dict_or_no_groups(self):
        # Only groups="cluster" needs unique pairs; the dict and None paths are unchanged.
        extra_positions = [3, 3, 100]
        X = pd.concat([self.X, self.X.iloc[extra_positions]])
        y = pd.concat([self.y, self.y.iloc[extra_positions]])
        for shrink_to_global in [True, False]:
            for groups in [EXPECTED_CLUSTERS, BLOCK_GROUPS, None]:
                label = f"groups={groups}, shrink_to_global={shrink_to_global}"
                model = clustered_model(
                    groups=groups, shrink_to_global=shrink_to_global
                ).fit(X, y)
                self.assertEqual(model.cids_, CLUSTER_CIDS, msg=label)
                if groups is None:
                    for cid in model.cids_:
                        coefs = model.cid_coefs_[cid]
                        self.assertTrue(np.all(np.isfinite(coefs)), msg=label)
                else:
                    self.assert_finite_coefficients(model, label)


if __name__ == "__main__":
    unittest.main()
