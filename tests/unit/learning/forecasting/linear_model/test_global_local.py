import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge

from macrosynergy.learning import GlobalLocalRegression
from macrosynergy.learning.forecasting.linear_model.global_local import (
    _cluster_returns,
    _coef_hierarchy,
    _compute_sufficient_statistics,
    _define_bounds,
    _loss,
    _loss_derivative,
    _mse,
    _mse_gradient,
    _validate_groups,
)

CIDS = ["AUD", "CAD", "GBP", "USD"]
SLOPES = {
    "AUD": [1.0, -0.5],
    "CAD": [0.8, -0.3],
    "GBP": [-0.4, 0.6],
    "USD": [-0.6, 0.9],
}
INTERCEPTS = {"AUD": 0.5, "CAD": 0.3, "GBP": -0.4, "USD": -0.2}
GROUPS = {"G1": ["AUD", "CAD"], "G2": ["GBP", "USD"]}


def make_panel(cids=CIDS, n_dates=120, noise=0.1, seed=0):
    """Simulate y = intercept + X @ slopes + noise separately for each cross-section."""
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    index = pd.MultiIndex.from_product([cids, dates], names=["cid", "real_date"])
    row_cids = index.get_level_values(0)

    X = pd.DataFrame(
        data=rng.normal(size=(len(index), 2)),
        index=index,
        columns=["CRY", "GROWTH"],
    )

    slopes = np.array([SLOPES[cid] for cid in row_cids])
    intercepts = np.array([INTERCEPTS[cid] for cid in row_cids])
    noise = noise * rng.normal(size=len(index))
    y = pd.Series(intercepts + np.sum(X.values * slopes, axis=1) + noise, index=index)

    return X, y


def make_block_returns(n_dates=100, seed=0):
    """AUD and GBP follow one common factor, CAD and USD another."""
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    factors = rng.normal(size=(2, n_dates))
    cid_factor = {"AUD": 0, "CAD": 1, "GBP": 0, "USD": 1}

    rets = pd.DataFrame(
        {
            cid: factors[f] + 0.3 * rng.normal(size=n_dates)
            for cid, f in cid_factor.items()
        },
        index=dates,
    )

    return rets.rename_axis(index="real_date", columns="cid").unstack()


def penalty_arrays(parents, n_features=2, strength=0.5):
    """diff_matrix and shrinkage that shrink each row towards its parent row."""
    diff_matrix = np.eye(len(parents))
    for row, parent in enumerate(parents):
        if parent >= 0:
            diff_matrix[row, parent] = -1

    shrinkage = np.full((len(parents), n_features), strength)

    return diff_matrix, shrinkage


@pytest.fixture
def panel():
    return make_panel()


@pytest.fixture
def suff_stats(panel):
    X, y = panel
    return _compute_sufficient_statistics(X, y, cids=CIDS)


class TestMSE:
    def test_equals_sum_of_cid_mean_squared_residuals(self, panel, suff_stats):
        X, y = panel

        weights = np.random.default_rng(1).normal(size=(len(CIDS) + 1, 2))

        result = _mse(weights, *suff_stats)
        expected = sum(
            np.mean((y.xs(cid).values - X.xs(cid).values @ weights[i]) ** 2)
            for i, cid in enumerate(CIDS)
        )

        assert result == pytest.approx(expected)


class TestMSEGradient:
    def test_matches_finite_differences(self, suff_stats):
        xtx, xty, yty = suff_stats
        weights = np.random.default_rng(1).normal(size=(len(CIDS), 2))

        result = _mse_gradient(weights, xtx, xty).flatten()
        expected = approx_fprime(
            weights.flatten(),
            lambda w: _mse(w.reshape(weights.shape), xtx, xty, yty),
            1e-6,
        )

        np.testing.assert_allclose(result, expected, atol=1e-4)


class TestLoss:
    def test_zero_shrinkage_equals_mse(self, suff_stats):
        xtx, xty, yty = suff_stats
        diff_matrix, _ = penalty_arrays([4, 4, 4, 4, -1])
        weights = np.random.default_rng(1).normal(size=(5, 2))

        loss = _loss(
            weights.flatten(),
            diff_matrix=diff_matrix,
            shrinkage=np.zeros((5, 2)),
            xtx=xtx,
            xty=xty,
            yty=yty,
        )

        assert loss == pytest.approx(_mse(weights, xtx, xty, yty))

    def test_penalty_shrinks_rows_towards_parents(self):
        # two local rows shrunk towards a global row, which is shrunk towards zero
        diff_matrix, _ = penalty_arrays([2, 2, -1], n_features=1)
        shrinkage = np.array([[0.5], [0.5], [0.1]])
        weights = np.array([1.0, 3.0, 2.0])

        # data is 0s so the loss is the penalty alone
        loss = _loss(
            weights,
            diff_matrix=diff_matrix,
            shrinkage=shrinkage,
            xtx=np.zeros((2, 1, 1)),
            xty=np.zeros((2, 1)),
            yty=np.zeros(2),
        )

        expected = 0.5 * (1 - 2) ** 2 + 0.5 * (3 - 2) ** 2 + 0.1 * 2**2
        assert loss == pytest.approx(expected)


class TestLossDerivative:
    @pytest.mark.parametrize(
        "parents",
        [
            [-1, -1, -1, -1],
            [4, 4, 4, 4, -1],
            [4, 4, 5, 5, 6, 6, -1],
        ],
        ids=["local", "local-global", "local-group-global"],
    )
    def test_matches_finite_differences(self, suff_stats, parents):
        xtx, xty, yty = suff_stats
        diff_matrix, shrinkage = penalty_arrays(parents)
        weights = np.random.default_rng(1).normal(size=len(parents) * 2)

        grad = _loss_derivative(
            weights=weights,
            diff_matrix=diff_matrix,
            shrinkage=shrinkage,
            xtx=xtx,
            xty=xty,
        )

        numerical = approx_fprime(
            weights,
            lambda w: _loss(
                w,
                diff_matrix=diff_matrix,
                shrinkage=shrinkage,
                xtx=xtx,
                xty=xty,
                yty=yty,
            ),
            1e-6,
        )

        np.testing.assert_allclose(grad, numerical, atol=1e-4)


class TestComputeSufficientStatistics:
    @pytest.mark.parametrize("weighted", [False, True])
    def test_matches_per_cid_weighted_means(self, panel, weighted):
        X, y = panel
        w = np.random.default_rng(1).uniform(0.5, 2, size=len(y)) if weighted else None

        xtx, xty, yty = _compute_sufficient_statistics(X, y, cids=CIDS, sample_weight=w)

        w = np.ones(len(y)) if w is None else w
        row_cids = X.index.get_level_values(0)
        for i, cid in enumerate(CIDS):
            mask = row_cids == cid
            X_i, y_i, w_i = X.values[mask], y.values[mask], w[mask]

            np.testing.assert_allclose(xtx[i], X_i.T @ np.diag(w_i) @ X_i / w_i.sum())
            np.testing.assert_allclose(xty[i], X_i.T @ (w_i * y_i) / w_i.sum())
            assert yty[i] == pytest.approx(w_i @ y_i**2 / w_i.sum())

    def test_row_order_does_not_matter(self, panel):
        X, y = panel
        shuffled = np.random.default_rng(1).permutation(len(y))

        expected = _compute_sufficient_statistics(X, y, cids=CIDS)
        result = _compute_sufficient_statistics(
            X.iloc[shuffled], y.iloc[shuffled], cids=CIDS
        )

        for res, exp in zip(result, expected):
            np.testing.assert_allclose(res, exp)


class TestClusterReturns:
    @pytest.mark.parametrize("linkage", ["average", "complete", "single"])
    def test_recovers_correlated_blocks(self, linkage):
        clusters, cid_to_group = _cluster_returns(
            rets=make_block_returns(),
            cids=CIDS,
            n_clusters=2,
            min_overlap=36,
            linkage=linkage,
        )

        assert clusters == {0: ["AUD", "GBP"], 1: ["CAD", "USD"]}
        assert cid_to_group == {"AUD": 0, "CAD": 1, "GBP": 0, "USD": 1}

    def test_one_cluster_holds_all_cids(self):
        clusters, cid_to_group = _cluster_returns(
            rets=make_block_returns(),
            cids=CIDS,
            n_clusters=1,
            min_overlap=36,
            linkage="average",
        )

        assert clusters == {0: CIDS}
        assert set(cid_to_group.values()) == {0}

    def test_single_cid_forms_one_cluster(self):
        rets = make_block_returns().loc[["AUD"]]

        clusters, cid_to_group = _cluster_returns(
            rets=rets, cids=["AUD"], n_clusters=2, min_overlap=36, linkage="average"
        )

        assert clusters == {0: ["AUD"]}
        assert cid_to_group == {"AUD": 0}


class TestCoefHierarchy:
    @pytest.mark.parametrize(
        "groups, shrink_to_global, scale, expected_parents, expected_lambdas",
        [
            (None, True, False, [3, 3, 3, -1], [1, 1, 1, 3]),
            (None, False, False, [-1, -1, -1], [1, 1, 1]),
            (["G1", "G2"], True, False, [3, 4, 3, 5, 5, -1], [1, 1, 1, 2, 2, 3]),
            (["G1", "G2"], False, False, [3, 4, 3, -1, -1], [1, 1, 1, 2, 2]),
            (None, True, True, [3, 3, 3, -1], [1, 1, 1, 3]),
            (["G1", "G2"], True, True, [3, 4, 3, 5, 5, -1], [1, 1, 1, 1, 1, 3]),
            (["G1", "G2"], False, True, [3, 4, 3, -1, -1], [1, 1, 1, 1, 1]),
        ],
        ids=[
            "local-global",
            "local",
            "local-group-global",
            "local-group",
            "local-global-scaled",
            "local-group-global-scaled",
            "local-group-scaled",
        ],
    )
    def test_parents_and_lambdas(
        self, groups, shrink_to_global, scale, expected_parents, expected_lambdas
    ):
        parents, lambdas = _coef_hierarchy(
            cids=["AUD", "CAD", "GBP"],
            local_lambda=1,
            group_lambda=2,
            global_lambda=3,
            groups=groups,
            cid_to_group={"AUD": "G1", "CAD": "G2", "GBP": "G1"},
            shrink_to_global=shrink_to_global,
            scale_group_lambda=scale,
        )

        np.testing.assert_array_equal(parents, expected_parents)
        np.testing.assert_array_equal(lambdas, expected_lambdas)


class TestDefineBounds:
    def test_unconstrained_returns_none(self):
        bounds = _define_bounds(
            n_rows=2, positive=False, fit_intercept=True, n_features=3
        )

        assert bounds is None

    def test_positive_leaves_intercepts_unbounded(self):
        bounds = _define_bounds(
            n_rows=2, positive=True, fit_intercept=True, n_features=3
        )

        assert bounds == [(None, None), (0, None), (0, None)] * 2

    def test_positive_without_intercept_bounds_every_coefficient(self):
        bounds = _define_bounds(
            n_rows=2, positive=True, fit_intercept=False, n_features=3
        )

        assert bounds == [(0, None)] * 6


class TestValidateGroups:
    def test_maps_each_training_cid_to_its_group(self):
        # JPY is not in the training data and is ignored
        groups = {"G1": ["AUD", "CAD", "JPY"], "G2": ["GBP"]}

        cid_to_group = _validate_groups(groups, ["AUD", "CAD", "GBP"])

        assert cid_to_group == {"AUD": "G1", "CAD": "G1", "GBP": "G2"}

    def test_non_dict_raises(self):
        with pytest.raises(TypeError):
            _validate_groups([["AUD", "CAD"]], ["AUD", "CAD"])

    def test_cid_in_two_groups_raises(self):
        groups = {"G1": ["AUD", "CAD"], "G2": ["CAD"]}

        with pytest.raises(ValueError, match="more than one group"):
            _validate_groups(groups, ["AUD", "CAD"])

    def test_unassigned_training_cid_raises(self):
        with pytest.raises(ValueError, match="not assigned"):
            _validate_groups({"G1": ["AUD"]}, ["AUD", "CAD"])


class TestGlobalLocalRegression:
    def test_fit_sets_coefficient_attributes(self, panel):
        X, y = panel
        model = GlobalLocalRegression()

        assert model.fit(X, y) is model
        assert model.cids_ == CIDS
        assert all(coefs.shape == (3,) for coefs in model.cid_coefs_.values())
        assert model.coef_.shape == (2,)
        assert np.isscalar(model.intercept_)

    def test_fit_without_intercept(self, panel):
        X, y = panel

        model = GlobalLocalRegression(fit_intercept=False).fit(X, y)

        assert all(coefs.shape == (2,) for coefs in model.cid_coefs_.values())
        assert model.coef_.shape == (2,)
        assert model.intercept_ is None

    def test_no_global_coefficients_without_shrink_to_global(self, panel):
        X, y = panel

        model = GlobalLocalRegression(shrink_to_global=False).fit(X, y)

        assert not hasattr(model, "coef_")
        assert not hasattr(model, "intercept_")

    def test_zero_local_lambda_gives_cid_by_cid_ols(self, panel):
        X, y = panel

        model = GlobalLocalRegression(local_lambda=0, shrink_to_global=False)
        model.fit(X, y)

        for cid in CIDS:
            ols = LinearRegression().fit(X.xs(cid), y.xs(cid))
            np.testing.assert_allclose(
                model.cid_coefs_[cid], [ols.intercept_, *ols.coef_], atol=1e-5
            )

    def test_single_cid_matches_ridge(self):
        X, y = make_panel(cids=["AUD"])
        local_lambda = 0.5

        model = GlobalLocalRegression(
            local_lambda=local_lambda, shrink_to_global=False
        ).fit(X, y)

        # Ridge minimises the summed rather than the mean squared residuals
        ridge = Ridge(alpha=local_lambda * len(y)).fit(X, y)
        np.testing.assert_allclose(
            model.cid_coefs_["AUD"], [ridge.intercept_, *ridge.coef_], atol=1e-5
        )

    def test_large_local_lambda_pools_cids(self, panel):
        X, y = panel

        model = GlobalLocalRegression(local_lambda=1e3, global_lambda=0).fit(X, y)

        pooled = LinearRegression().fit(X, y)
        np.testing.assert_allclose(model.coef_, pooled.coef_, atol=1e-3)
        assert model.intercept_ == pytest.approx(pooled.intercept_, abs=1e-3)
        for cid in CIDS:
            np.testing.assert_allclose(
                model.cid_coefs_[cid], [model.intercept_, *model.coef_], atol=1e-2
            )

    def test_top_level_intercepts_are_not_shrunk(self, panel):
        X, y = panel

        model = GlobalLocalRegression(local_lambda=1e3, shrink_to_global=False)
        model.fit(X, y)

        cid_means = y.groupby(level=0).mean()
        for cid in CIDS:
            intercept, *slopes = model.cid_coefs_[cid]
            assert intercept == pytest.approx(cid_means[cid], abs=1e-2)
            np.testing.assert_allclose(slopes, 0, atol=1e-2)

    def test_large_local_lambda_pools_cids_within_groups(self, panel):
        X, y = panel

        model = GlobalLocalRegression(
            local_lambda=1e3, group_lambda=0, groups=GROUPS, shrink_to_global=False
        ).fit(X, y)

        assert model.groups_ == ["G1", "G2"]
        for group, group_cids in GROUPS.items():
            pooled = LinearRegression().fit(X.loc[group_cids], y.loc[group_cids])
            group_coefs = model.group_coefs_[group]
            np.testing.assert_allclose(
                group_coefs, [pooled.intercept_, *pooled.coef_], atol=1e-2
            )
            for cid in group_cids:
                np.testing.assert_allclose(
                    model.cid_coefs_[cid], group_coefs, atol=1e-2
                )

    @pytest.mark.parametrize("shrink_to_global", [True, False])
    def test_scale_group_lambda_divides_by_number_of_groups(
        self, panel, shrink_to_global
    ):
        X, y = panel
        params = {"groups": GROUPS, "shrink_to_global": shrink_to_global}
        scaled = GlobalLocalRegression(
            group_lambda=2 * len(GROUPS), scale_group_lambda=True, **params
        ).fit(X, y)
        unscaled = GlobalLocalRegression(group_lambda=2, **params).fit(X, y)
        for group in GROUPS:
            np.testing.assert_allclose(
                scaled.group_coefs_[group], unscaled.group_coefs_[group], atol=1e-6
            )

    def test_positive_constrains_slopes_but_not_intercepts(self, panel):
            X, y = panel

            model = GlobalLocalRegression(positive=True).fit(X, y)

            assert np.all(model.coef_ >= 0)
            assert all(np.all(coefs[1:] >= 0) for coefs in model.cid_coefs_.values())
            assert model.cid_coefs_["GBP"][0] < 0

    def test_predict_uses_cid_coefficients(self, panel):
        X, y = panel
        model = GlobalLocalRegression().fit(X, y)

        preds = model.predict(X)

        row_cids = X.index.get_level_values(0)
        for cid in CIDS:
            mask = row_cids == cid
            coefs = model.cid_coefs_[cid]
            np.testing.assert_allclose(
                preds[mask], coefs[0] + X.values[mask] @ coefs[1:]
            )

    def test_n_clusters_above_number_of_cids_falls_back(self, panel):
        X, y = panel

        with pytest.warns(UserWarning, match="Falling back"):
            model = GlobalLocalRegression(groups="cluster", n_clusters=10).fit(X, y)

        assert model.n_groups_ == 1

    @pytest.mark.parametrize(
        "first, second",
        [
            ({"groups": GROUPS}, {"groups": None}),
            ({"groups": "cluster"}, {"groups": GROUPS}),
            ({"shrink_to_global": True}, {"shrink_to_global": False}),
        ],
    )
    def test_refit_matches_fresh_fit(self, panel, first, second):
        X, y = panel
        model = GlobalLocalRegression(**first).fit(X, y)

        model.set_params(**second).fit(X, y)
        fresh = clone(model).fit(X, y)

        assert vars(model).keys() == vars(fresh).keys()
        for cid in CIDS:
            np.testing.assert_allclose(model.cid_coefs_[cid], fresh.cid_coefs_[cid])

    @pytest.mark.parametrize(
        "params, error",
        [
            ({"groups": ["AUD", "CAD"]}, TypeError),
            ({"groups": "kmeans"}, TypeError),
            ({"groups": "cluster", "n_clusters": 2.0}, TypeError),
            ({"groups": "cluster", "min_overlap": 36.0}, TypeError),
            ({"groups": "cluster", "n_clusters": 0}, ValueError),
            ({"groups": "cluster", "linkage": "invalid"}, ValueError),
            ({"groups": "cluster", "min_overlap": 1}, ValueError),
            ({"groups": GROUPS, "scale_group_lambda": 1}, TypeError),
        ],
    )
    def test_invalid_groups_parameters_raise(self, panel, params, error):
        X, y = panel

        with pytest.raises(error):
            GlobalLocalRegression(**params).fit(X, y)
