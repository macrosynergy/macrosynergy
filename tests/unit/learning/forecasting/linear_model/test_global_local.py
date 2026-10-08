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
from unittest import mock

import numpy as np
import pandas as pd
from scipy.optimize import minimize
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


if __name__ == "__main__":
    unittest.main()
