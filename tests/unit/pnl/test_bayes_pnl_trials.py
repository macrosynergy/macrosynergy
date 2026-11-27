"""Independent references and selection calibration for BayesianPnLTrials."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special, stats

from macrosynergy.pnl import BayesianPnLTrials as ExportedTrials
from macrosynergy.pnl import estimate_correlation as exported_estimate_correlation
from macrosynergy.pnl.bayes_pnl import (
    BayesianPnLTrials,
    GridTruncationWarning,
    NormalPrior,
    effective_number_of_trials,
    estimate_correlation,
    expected_maximum_sharpe,
    period_metrics,
)

R3 = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]])
DATA3 = pd.DataFrame(
    [
        [0.9, 0.6, 0.2],
        [0.1, 0.4, -0.3],
        [1.2, 0.8, 0.5],
        [-0.2, 0.1, 0.0],
        [0.7, 0.9, 0.4],
    ],
    columns=["a", "b", "c"],
)


def equicorrelation(k, rho):
    return rho * np.ones((k, k)) + (1 - rho) * np.eye(k)


# ----------------------------------------------------------------- 1. conjugate


@pytest.mark.parametrize("w", [0.0, 0.4])
def test_single_version_with_fixed_scales_is_conjugate_normal(w):
    x = np.array([[0.8], [0.1], [0.5], [-0.2]])
    T, noise, period = len(x), 0.6, 3.0
    prior = NormalPrior(0.2, 0.7)
    fit = BayesianPnLTrials(
        x,
        period_years=period,
        noise_sd=noise,
        mean_prior=prior,
        tau_grid=[0.0],
        omega_grid=[w],
    )
    v = (noise**2 + w**2) / T
    post_var = 1 / (1 / prior.sd**2 + 1 / v)
    post_mean = post_var * (prior.mean / prior.sd**2 + x.mean() / v)
    assert fit.expectation() == pytest.approx(post_mean, abs=1e-12)
    assert fit.std() == pytest.approx(np.sqrt(post_var), abs=1e-12)
    assert fit.expectation("idea") == pytest.approx(post_mean, abs=1e-12)
    assert fit.prob_mean_below(0.3) == pytest.approx(
        stats.norm.cdf(0.3, post_mean, np.sqrt(post_var)), abs=1e-12
    )
    expected_evidence = stats.multivariate_normal.logpdf(
        x.ravel(),
        mean=np.full(T, prior.mean),
        cov=noise**2 * np.eye(T) + w**2 * np.eye(T) + prior.sd**2 * np.ones((T, T)),
    )
    assert fit.log_evidence == pytest.approx(expected_evidence, abs=1e-10)
    for horizon in (None, 1.0, 6.0):
        h = period if horizon is None else horizon
        sd = np.sqrt(post_var + (w**2 + noise**2) * period / h)
        assert fit.predictive_cdf(0.1, horizon_years=horizon)[0] == pytest.approx(
            stats.norm.cdf(0.1, post_mean, sd), abs=1e-12
        )
        assert fit.predictive_quantile(0.9, horizon_years=horizon)[0] == pytest.approx(
            stats.norm.ppf(0.9, post_mean, sd), abs=1e-9
        )


# ----------------------------------------------------- 2. brute-force reference


def dense_reference(x, R, noise, prior, tau_scale, omega_scale, n=161, upper=2.6):
    """Dense TK-dimensional Gaussian per (omega, tau) node; Simpson over the scales.

    Does not use the scatter/mean decomposition or Sherman-Morrison: the joint
    normal of (theta, vec(data)) is formed and conditioned directly.
    """
    T, K = x.shape
    y = x.ravel()
    omegas = np.linspace(0, upper, n)
    taus = np.linspace(0, upper, n)
    V = noise**2 * R
    log_w = np.empty((n, n))
    mean_sel = np.empty((n, n))
    var_sel = np.empty((n, n))
    for i, om in enumerate(omegas):
        for j, ta in enumerate(taus):
            G = prior.sd**2 * np.ones((K, K)) + ta**2 * np.eye(K)
            cov_y = (
                np.kron(np.eye(T), V + om**2 * np.ones((K, K)))
                + np.kron(np.ones((T, T)), G)
            )
            cross = np.kron(np.ones((1, T)), G)  # Cov(theta, y)
            chol = np.linalg.cholesky(cov_y)
            resid = y - prior.mean
            z = np.linalg.solve(chol, resid)
            log_w[i, j] = (
                -0.5 * z @ z
                - np.log(np.diag(chol)).sum()
                - 0.5 * len(y) * np.log(2 * np.pi)
                + stats.halfnorm.logpdf(om, scale=omega_scale)
                + stats.halfnorm.logpdf(ta, scale=tau_scale)
            )
            solved = np.linalg.solve(chol.T, np.linalg.solve(chol, cross.T))
            mean_sel[i, j] = prior.mean + (cross @ np.linalg.solve(chol.T, z))[0]
            var_sel[i, j] = G[0, 0] - (cross @ solved)[0, 0]
    weights = np.exp(log_w - log_w.max())
    simpson = np.outer(
        integrate.simpson(np.eye(n), x=omegas), integrate.simpson(np.eye(n), x=taus)
    )
    weights = weights * simpson
    weights /= weights.sum()
    return omegas, weights, mean_sel, var_sel


def test_posterior_matches_dense_gaussian_brute_force():
    prior = NormalPrior(0.0, 1.0)
    noise = 0.6
    omegas, weights, mean_sel, var_sel = dense_reference(
        DATA3.to_numpy(), R3, noise, prior, 0.5, 0.5
    )
    sd_sel = np.sqrt(var_sel)
    ref_mean = float((weights * mean_sel).sum())
    ref_sd = float(np.sqrt((weights * (var_sel + (mean_sel - ref_mean) ** 2)).sum()))
    ref_prob = float((weights * special.ndtr(mean_sel / sd_sel)).sum())
    pred_sd = np.sqrt(var_sel + (omegas[:, None] ** 2 + noise**2) * 3.0 / 1.0)
    ref_pred = float((weights * special.ndtr((mean_sel - 0.5) / pred_sd)).sum())
    fine = np.linspace(0, 2.6, 161)
    for kwargs, tol in (({}, 3e-3), ({"omega_grid": fine, "tau_grid": fine}, 2e-4)):
        fit = BayesianPnLTrials(
            DATA3,
            R3,
            construction_correlation=None,
            period_years=3.0,
            selected="a",
            noise_sd=noise,
            mean_prior=prior,
            **kwargs,
        )
        assert fit.expectation() == pytest.approx(ref_mean, abs=tol)
        assert fit.std() == pytest.approx(ref_sd, abs=tol)
        assert fit.prob_mean_above(0.0) == pytest.approx(ref_prob, abs=tol)
        assert fit.predictive_prob_above(0.5, horizon_years=1.0) == pytest.approx(
            ref_pred, abs=tol
        )


def test_fixed_scale_evidence_and_moments_match_dense_gaussian():
    prior = NormalPrior(0.1, 0.8)
    x = DATA3.to_numpy()
    T, K = x.shape
    fit = BayesianPnLTrials(
        DATA3, R3, period_years=3, noise_sd=0.5, mean_prior=prior,
        omega_grid=[0.3], tau_grid=[0.2], construction_correlation=None,
    )
    G = 0.8**2 * np.ones((K, K)) + 0.2**2 * np.eye(K)
    Sw = 0.25 * R3 + 0.09 * np.ones((K, K))
    cov_y = np.kron(np.eye(T), Sw) + np.kron(np.ones((T, T)), G)
    assert fit.log_evidence == pytest.approx(
        stats.multivariate_normal.logpdf(x.ravel(), np.full(T * K, 0.1), cov_y),
        abs=1e-10,
    )
    cross = np.kron(np.ones((1, T)), G)
    gain = np.linalg.solve(cov_y, cross.T).T
    post_mean = 0.1 + gain @ (x.ravel() - 0.1)
    post_cov = G - gain @ cross.T
    for v, name in enumerate(DATA3.columns):
        assert fit.expectation(name) == pytest.approx(post_mean[v], abs=1e-12)
        assert fit.std(name) == pytest.approx(np.sqrt(post_cov[v, v]), abs=1e-12)
    # The common level m: joint normal of (m, data) formed directly.
    cross_m = np.full((1, T * K), 0.8**2)
    m_mean = 0.1 + (cross_m @ np.linalg.solve(cov_y, x.ravel() - 0.1))[0]
    m_var = 0.64 - (cross_m @ np.linalg.solve(cov_y, cross_m.T))[0, 0]
    assert fit.expectation("idea") == pytest.approx(m_mean, abs=1e-12)
    assert fit.std("idea") == pytest.approx(np.sqrt(m_var), abs=1e-12)
    q = fit.weights.to_numpy()
    expected = q @ stats.norm.cdf(0.2, post_mean, np.sqrt(np.diag(post_cov)))
    assert fit.prob_mean_below(0.2, "draw") == pytest.approx(expected, abs=1e-12)


# ----------------------------------------------------------------------- 3. T=1


def test_single_period_works_and_equals_dense_limit():
    x = DATA3.iloc[[0]]
    prior = NormalPrior(0.0, 1.0)
    fit = BayesianPnLTrials(
        x, R3, period_years=3, noise_sd=0.6, omega_grid=[0.3], tau_grid=[0.4],
        construction_correlation=None,
    )
    G = np.ones((3, 3)) + 0.16 * np.eye(3)
    Lam = G + 0.36 * R3 + 0.09 * np.ones((3, 3))
    s = x.to_numpy()[0]
    mean = G @ np.linalg.solve(Lam, s)
    cov = G - G @ np.linalg.solve(Lam, G)
    np.testing.assert_allclose(
        [fit.expectation(v) for v in "abc"], mean, atol=1e-12
    )
    np.testing.assert_allclose(
        [fit.std(v) ** 2 for v in "abc"], np.diag(cov), atol=1e-12
    )
    assert fit.log_evidence == pytest.approx(
        stats.multivariate_normal.logpdf(s, np.zeros(3), Lam), abs=1e-12
    )
    default = BayesianPnLTrials(x, R3, period_years=3, selected="a")
    assert np.isfinite(default.summary().to_numpy(dtype=float)[:, :4]).all()
    assert prior == default.mean_prior


# ----------------------------------------------------------------- 4. duplicates


def near_duplicate_setup():
    rho = 0.999
    R4 = np.eye(4)
    R4[:3, :3] = R3
    R4[3, :3] = R4[:3, 3] = rho * R3[2]
    R4[3, 2] = R4[2, 3] = rho
    rng = np.random.default_rng(3)
    copy = DATA3.c + 0.6 * np.sqrt(2 * (1 - rho)) * rng.standard_normal(len(DATA3))
    independent = np.eye(4)
    independent[:3, :3] = R3
    return DATA3.assign(c_copy=copy), R4, DATA3.assign(c_copy=DATA3.c), independent


def test_near_duplicate_versions_add_almost_nothing():
    """With construction effects correlated like the noise (C = R)."""
    data4, R4, data_indep, R_indep = near_duplicate_setup()
    kwargs = dict(
        period_years=3, selected="a", noise_sd=0.6, construction_correlation="noise"
    )
    base = BayesianPnLTrials(DATA3, R3, **kwargs)
    dup = BayesianPnLTrials(data4, R4, **kwargs)
    indep = BayesianPnLTrials(data_indep, R_indep, **kwargs)
    assert dup.expectation() == pytest.approx(base.expectation(), abs=0.02)
    assert dup.std() == pytest.approx(base.std(), rel=0.02)
    # The same column declared independent is genuinely new information.
    shift_dup = abs(dup.expectation() - base.expectation())
    assert abs(indep.expectation() - base.expectation()) > 3 * shift_dup
    assert effective_number_of_trials(np.eye(7)) == pytest.approx(7)
    assert effective_number_of_trials(equicorrelation(10, 0.999)) == pytest.approx(
        1.0, abs=0.01
    )
    assert dup.effective_trials < base.effective_trials + 0.1


def test_exchangeable_construction_prior_reads_a_duplicate_as_small_tau():
    """Default C = I: a coinciding pair of effects is evidence of small tau.

    Documents the default's behaviour rather than endorsing it: the noise
    model discounts the duplicated sampling error, but the iid prior on
    construction effects treats the copy as a second independent effect.
    """
    data4, R4, _, _ = near_duplicate_setup()
    independent = dict(period_years=3, selected="a", noise_sd=0.6, construction_correlation=None)
    base = BayesianPnLTrials(DATA3, R3, **independent)
    dup = BayesianPnLTrials(data4, R4, **independent)
    tau_base = base.scale_summary().loc["tau", "mean"]
    tau_dup = dup.scale_summary().loc["tau", "mean"]
    assert tau_dup < 0.5 * tau_base
    assert dup.expectation() < base.expectation() - 0.03


def test_general_construction_correlation_matches_dense_gaussian():
    C = np.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.4], [0.0, 0.4, 1.0]])
    x = DATA3.to_numpy()
    T, K = x.shape
    fit = BayesianPnLTrials(
        DATA3, R3, period_years=3, noise_sd=0.5, omega_grid=[0.3], tau_grid=[0.4],
        construction_correlation=pd.DataFrame(C, index=list("abc"), columns=list("abc")),
    )
    G = np.ones((K, K)) + 0.16 * C
    cov_y = np.kron(np.eye(T), 0.25 * R3 + 0.09) + np.kron(np.ones((T, T)), G)
    cross = np.kron(np.ones((1, T)), G)
    gain = np.linalg.solve(cov_y, cross.T).T
    np.testing.assert_allclose(
        [fit.expectation(v) for v in "abc"], gain @ x.ravel(), atol=1e-12
    )
    np.testing.assert_allclose(
        [fit.std(v) ** 2 for v in "abc"], np.diag(G - gain @ cross.T), atol=1e-12
    )
    assert fit.log_evidence == pytest.approx(
        stats.multivariate_normal.logpdf(x.ravel(), np.zeros(T * K), cov_y), abs=1e-10
    )


def test_expected_maximum_sharpe():
    sharpes = np.array([0.1, 0.4, -0.2, 0.3, 0.0])
    g = np.euler_gamma
    expected = np.std(sharpes, ddof=1) * (
        (1 - g) * stats.norm.ppf(1 - 1 / 5) + g * stats.norm.ppf(1 - 1 / (5 * np.e))
    )
    assert expected_maximum_sharpe(sharpes) == pytest.approx(expected)
    assert expected_maximum_sharpe(sharpes, n_trials=1) == 0.0
    assert expected_maximum_sharpe([0.3]) == 0.0
    assert expected_maximum_sharpe(sharpes, 50) > expected_maximum_sharpe(sharpes)
    # Close to the expected maximum of N standard normals for large N.
    draws = np.random.default_rng(0).standard_normal((20000, 100)).max(axis=1)
    assert expected_maximum_sharpe(
        np.random.default_rng(1).standard_normal(100000), 100
    ) == pytest.approx(draws.mean(), rel=0.02)


# ------------------------------------------------------- 5./6. selection checks


def simulate_world(rng, K, T, R_chol, noise, omega, tau, m):
    theta = m + tau * rng.standard_normal(K)
    eta = omega * rng.standard_normal((T, 1))
    data = theta + eta + noise * rng.standard_normal((T, K)) @ R_chol.T
    return theta, data


def test_selected_version_posterior_is_calibrated_under_selection():
    """Worlds drawn from the model's own prior; the best version is reported.

    Posterior intervals and probabilities of the selected version must be
    calibrated over worlds, while a single-version fit of the selected
    version alone, ignoring the search, must overstate it.
    """
    rng = np.random.default_rng(20261127)
    K, T, noise, period, n_worlds = 20, 4, 0.6, 3.0, 1000
    tau_scale, omega_scale = 0.2, 0.3
    omega_grid = np.linspace(0, 4 * omega_scale, 21)
    tau_grid = np.linspace(0, 4 * tau_scale, 21)
    R = equicorrelation(K, 0.3)
    R_chol = np.linalg.cholesky(R)
    common = dict(
        period_years=period, noise_sd=noise, omega_grid=omega_grid,
        tau_grid=tau_grid, tau_scale=tau_scale, omega_scale=omega_scale,
    )
    covered = covered_alone = 0
    probs, probs_alone, positive = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GridTruncationWarning)
        for _ in range(n_worlds):
            m = rng.standard_normal()
            omega = abs(omega_scale * rng.standard_normal())
            tau = abs(tau_scale * rng.standard_normal())
            if omega > omega_grid[-1] or tau > tau_grid[-1]:
                continue  # prior truncated to the grid support
            theta, data = simulate_world(rng, K, T, R_chol, noise, omega, tau, m)
            best = int(np.argmax(data.mean(axis=0)))
            frame = pd.DataFrame(data, columns=["V{}".format(i) for i in range(K)])
            fit = BayesianPnLTrials(
                frame, R, selected="V{}".format(best), **common
            )
            alone = BayesianPnLTrials(frame.iloc[:, [best]], **common)
            low, high = fit.interval(0.9)
            covered += low <= theta[best] <= high
            low, high = alone.interval(0.9)
            covered_alone += low <= theta[best] <= high
            probs.append(fit.prob_mean_above(0.0))
            probs_alone.append(alone.prob_mean_above(0.0))
            positive.append(theta[best] > 0)
    n = len(probs)
    assert n > 990
    probs, probs_alone, positive = map(np.asarray, (probs, probs_alone, positive))
    se_cover = np.sqrt(0.09 / n)
    coverage, coverage_alone = covered / n, covered_alone / n
    assert abs(coverage - 0.9) < 3.5 * se_cover
    # Calibration in the mean: E[P(theta>0 | D)] = P(theta>0) under the prior.
    gap = probs - positive
    assert abs(gap.mean()) < 3.5 * gap.std(ddof=1) / np.sqrt(n)
    # Calibration within bins of the posterior probability.
    for lo, hi in ((0.0, 0.3), (0.3, 0.7), (0.7, 1.0)):
        inside = (probs >= lo) & (probs < hi)
        if inside.sum() > 50:
            diff = positive[inside] - probs[inside]
            assert abs(diff.mean()) < 3.5 * diff.std(ddof=1) / np.sqrt(inside.sum())
    # Power: ignoring the other versions overstates the selected version.
    gap_alone = probs_alone - positive
    assert gap_alone.mean() > 0.05
    assert gap_alone.mean() > 5 * gap_alone.std(ddof=1) / np.sqrt(n)
    assert coverage_alone < 0.9 - 5 * se_cover


def test_no_true_value_best_of_twenty_rarely_looks_convincing():
    rng = np.random.default_rng(7)
    K, T, noise, period, n_worlds = 20, 5, 1 / np.sqrt(3), 3.0, 300
    R = equicorrelation(K, 0.3)
    R_chol = np.linalg.cholesky(R)
    hits = hits_alone = 0
    for _ in range(n_worlds):
        _, data = simulate_world(rng, K, T, R_chol, noise, 0.05, 0.0, 0.0)
        best = int(np.argmax(data.mean(axis=0)))
        frame = pd.DataFrame(data, columns=["V{}".format(i) for i in range(K)])
        fit = BayesianPnLTrials(
            frame, R, period_years=period, selected="V{}".format(best),
            n_omega=21, n_tau=21,
        )
        alone = BayesianPnLTrials(
            frame.iloc[:, [best]], period_years=period, n_omega=21, n_tau=21
        )
        hits += fit.prob_mean_above(0.0) > 0.95
        hits_alone += alone.prob_mean_above(0.0) > 0.95
    assert hits / n_worlds <= 0.10
    assert hits_alone / n_worlds > hits / n_worlds + 0.15


# ------------------------------------------------------------ 7. period metrics


@pytest.fixture
def daily_pnl():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2002-01-01", "2010-12-31")
    common = rng.standard_normal(len(dates))
    return pd.DataFrame(
        {
            "a": 0.05 + common + 0.8 * rng.standard_normal(len(dates)),
            "b": 0.02 + common + 0.8 * rng.standard_normal(len(dates)),
            "c": 0.03 + 0.5 * common + rng.standard_normal(len(dates)),
        },
        index=dates,
    )


def test_period_metrics_default_unchanged_and_one_year_blocks(daily_pnl):
    default = period_metrics(daily_pnl)
    explicit = period_metrics(daily_pnl, period_years=3)
    pd.testing.assert_frame_equal(default, explicit)
    assert len(default) == 9 and (default.period_years == 3).all()
    block = daily_pnl.loc["2008-01-01":"2010-12-31", "a"]
    last = default[default.variant == "a"].iloc[-1]
    assert last.sharpe == pytest.approx(np.sqrt(252) * block.mean() / block.std(ddof=1))
    yearly = period_metrics(daily_pnl, period_years=1)
    assert len(yearly) == 27 and (yearly.period_years == 1).all()
    for year in (2002, 2006, 2010):
        values = daily_pnl.loc[str(year), "b"]
        row = yearly[
            (yearly.variant == "b") & (yearly.period_end == pd.Timestamp(year, 12, 31))
        ].iloc[0]
        assert row.period_start == pd.Timestamp(year, 1, 1)
        assert row.n_observations == len(values)
        assert row.sharpe == pytest.approx(
            np.sqrt(252) * values.mean() / values.std(ddof=1)
        )
    two = period_metrics(daily_pnl, period_years=2)
    assert sorted(two.period_end.dt.year.unique()) == [2004, 2006, 2008, 2010]
    with pytest.raises(ValueError):
        period_metrics(daily_pnl, period_years=0)


def test_from_pnl_round_trip(daily_pnl):
    fit = BayesianPnLTrials.from_pnl(
        daily_pnl,
        selected="a",
        n_omega=21,
        n_tau=21,
        correlation_method="sample",
    )
    yearly = period_metrics(daily_pnl, period_years=1)
    expected = yearly.pivot(index="period_end", columns="variant", values="sharpe")
    np.testing.assert_allclose(fit.data, expected[["a", "b", "c"]].to_numpy())
    np.testing.assert_allclose(
        fit.noise_correlation, daily_pnl.corr().to_numpy(), atol=1e-12
    )
    assert fit.period_years == 1 and fit.selected == "a"
    assert set(fit.periods.period_years) == {1}
    direct = BayesianPnLTrials(
        expected[["a", "b", "c"]], daily_pnl.corr(), period_years=1, selected="a",
        n_omega=21, n_tau=21,
    )
    assert fit.expectation() == pytest.approx(direct.expectation())
    with pytest.raises(ValueError, match="duplicate"):
        BayesianPnLTrials.from_pnl(daily_pnl.assign(d=2 * daily_pnl.a))
    with pytest.raises(ValueError, match="collinear"):
        BayesianPnLTrials.from_pnl(daily_pnl.assign(d=daily_pnl.a + daily_pnl.b))


# --------------------------------------------------------- 8. validation & misc


@pytest.mark.parametrize(
    "data, correlation, kwargs",
    [
        (np.ones(5), None, {}),
        (np.array([[0.1, np.nan]]), np.eye(2), {}),
        (DATA3, None, {}),
        (DATA3, np.eye(2), {}),
        (DATA3, np.ones((3, 3)), {}),
        (DATA3, np.array([[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]]), {}),
        (DATA3, R3, {"selected": "zzz"}),
        (DATA3.rename(columns={"a": "idea"}), R3, {}),
        (DATA3, R3, {"noise_sd": -1}),
        (DATA3, R3, {"weights": [1, -1, 1]}),
        (DATA3, R3, {"tau_grid": [0.2, 0.1]}),
        (DATA3, R3, {"n_tau": 1}),
    ],
)
def test_invalid_inputs_are_rejected(data, correlation, kwargs):
    with pytest.raises((TypeError, ValueError)):
        BayesianPnLTrials(data, correlation, period_years=3, **kwargs)


def test_target_errors_monotonicity_and_aligned_correlation():
    shuffled = pd.DataFrame(R3, index=list("abc"), columns=list("abc")).loc[
        list("cab"), list("bca")
    ]
    fit = BayesianPnLTrials(DATA3, shuffled, period_years=3, n_omega=21, n_tau=21)
    reference = BayesianPnLTrials(DATA3, R3, period_years=3, n_omega=21, n_tau=21)
    assert fit.expectation("b") == pytest.approx(reference.expectation("b"))
    with pytest.raises(ValueError):
        fit.expectation("selected")
    with pytest.raises(ValueError):
        fit.expectation("unknown")
    with pytest.raises(ValueError):
        fit.predictive_cdf(0.0, target="idea")
    with pytest.raises(TypeError):
        BayesianPnLTrials(DATA3, R3, period_years=3, mean_prior=0.0)
    thresholds = np.linspace(-1, 1.5, 11)
    for target in (None, "idea", "a"):
        above = [fit.prob_mean_above(t, target) for t in thresholds]
        assert np.all(np.diff(above) < 0)
    nexts = [fit.predictive_prob_above(t, "a") for t in thresholds]
    assert np.all(np.diff(nexts) < 0)
    q = fit.quantile([0.1, 0.5, 0.9], "draw")
    np.testing.assert_allclose(fit.cdf(q, "draw"), [0.1, 0.5, 0.9], atol=1e-9)
    assert ExportedTrials is BayesianPnLTrials


def test_summary_scale_summary_and_defaults():
    fit = BayesianPnLTrials(
        DATA3, R3, period_years=3, selected="b", weights={"a": 1, "b": 1, "c": 2},
        n_omega=21, n_tau=21,
    )
    table = fit.summary(thresholds=(0.0, 0.5))
    assert list(table.index) == ["selected", "idea", "draw", "a", "b", "c"]
    assert table.loc["selected", "expectation"] == pytest.approx(
        table.loc["b", "expectation"]
    )
    assert np.isnan(table.loc["idea", "prob_next_above_0"])
    assert fit.expectation() == fit.expectation("b")
    assert fit.weights["c"] == pytest.approx(0.5)
    grand = DATA3.to_numpy().mean()
    assert fit.noise_sd == pytest.approx(np.sqrt((1 + grand**2 / 504) / 3))
    monthly = BayesianPnLTrials(
        DATA3, R3, period_years=3, observations_per_year=12, n_omega=5, n_tau=5
    )
    assert monthly.noise_sd == pytest.approx(np.sqrt((1 + grand**2 / 24) / 3))
    assert fit.mass.shape == (len(fit.omega), len(fit.tau))
    assert fit.mass.sum() == pytest.approx(1.0)
    scales = fit.scale_summary()
    assert list(scales.index) == ["omega", "tau"]
    assert (scales.lo <= scales["mean"]).all() and (scales["mean"] <= scales.hi).all()
    no_selection = BayesianPnLTrials(DATA3, R3, period_years=3, n_omega=21, n_tau=21)
    assert no_selection.expectation() == no_selection.expectation("draw")
    assert "selected" not in no_selection.summary().index


def test_edge_mass_warning():
    wild = DATA3 * 10
    with pytest.warns(GridTruncationWarning):
        BayesianPnLTrials(wild, R3, period_years=3, omega_grid=np.linspace(0, 0.5, 11))


def test_large_panel_fit_is_fast():
    import time

    rng = np.random.default_rng(5)
    K, T = 100, 21
    R = equicorrelation(K, 0.5)
    _, data = simulate_world(rng, K, T, np.linalg.cholesky(R), 1.0, 0.2, 0.2, 0.3)
    start = time.perf_counter()
    fit = BayesianPnLTrials(data, R, period_years=1, selected="V0")
    assert time.perf_counter() - start < 2.0
    assert np.isfinite(fit.std())


def test_singular_noise_correlation_is_rejected():
    near = np.full((3, 3), 1 - 1e-13)
    np.fill_diagonal(near, 1.0)
    with pytest.raises(ValueError, match="singular|positive definite"):
        BayesianPnLTrials(DATA3, near, period_years=3)


def test_default_grid_reaches_data_far_from_the_mean_prior():
    rng = np.random.default_rng(11)
    far = pd.DataFrame(50 + 5 * rng.standard_normal((5, 3)), columns=list("abc"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", GridTruncationWarning)
        fit = BayesianPnLTrials(far, R3, period_years=3, noise_sd=5.0)
    wide = BayesianPnLTrials(
        far, R3, period_years=3, noise_sd=5.0,
        tau_grid=np.linspace(0, 400, 1601), omega_grid=np.linspace(0, 60, 241),
    )
    assert fit.expectation("a") == pytest.approx(wide.expectation("a"), rel=0.02)


# ------------------------------------------------- 9. correlation estimation

METHODS = ("sample", "ledoit_wolf", "factor")


def simulate(truth, T, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, len(truth))) @ np.linalg.cholesky(truth).T


def factor_truth(K, n_factors=3, seed=99):
    rng = np.random.default_rng(seed)
    loadings = rng.uniform(0.4, 0.8, (K, n_factors)) * rng.choice([-1, 1], (K, n_factors))
    loadings *= 0.9 / np.sqrt((loadings**2).sum(axis=1) + 0.5)[:, None]
    truth = loadings @ loadings.T
    np.fill_diagonal(truth, 1.0)
    return truth


def test_estimate_correlation_export_and_shape():
    assert exported_estimate_correlation is estimate_correlation
    x = simulate(equicorrelation(4, 0.3), 200, 0)
    matrix, info = estimate_correlation(x)
    assert info["method"] == "ledoit_wolf" and matrix.shape == (4, 4)
    frame = pd.DataFrame(x, columns=list("wxyz"))
    labelled, _ = estimate_correlation(frame)
    assert list(labelled.index) == list("wxyz") and list(labelled.columns) == list("wxyz")
    np.testing.assert_allclose(labelled.to_numpy(), matrix)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("K,T", [(10, 200), (40, 30)])
def test_estimate_correlation_is_valid_matrix(method, K, T):
    if method == "sample" and K > T:
        pytest.skip("sample correlation is singular when K > T")
    x = simulate(equicorrelation(K, 0.4), T, 1)
    x[0, 0] = np.nan  # rows with NaN are dropped
    matrix, info = estimate_correlation(x, method)
    assert info["n_observations"] == T - 1 and info["n_variables"] == K
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(matrix)[0] > 0
    np.linalg.cholesky(matrix)


def test_sample_method_matches_corrcoef():
    x = simulate(R3, 300, 2)
    matrix, _ = estimate_correlation(x, "sample")
    np.testing.assert_allclose(matrix, np.corrcoef(x, rowvar=False), atol=1e-12)


def test_estimators_beat_sample_in_known_worlds():
    cases = {
        "equi": (equicorrelation(100, 0.5), 252, "ledoit_wolf"),
        "factor": (factor_truth(60), 250, "factor"),
    }
    for truth, T, method in cases.values():
        inverse = np.linalg.inv(truth)
        errors = {m: [] for m in ("sample", method)}
        for seed in range(20):
            x = simulate(truth, T, seed)
            for m in errors:
                R, _ = estimate_correlation(x, m)
                errors[m].append(
                    (
                        np.linalg.norm(R - truth),
                        np.linalg.norm(np.linalg.inv(R) - inverse),
                        np.linalg.cond(R),
                    )
                )
        mean_sample = np.mean(errors["sample"], axis=0)
        mean_method = np.mean(errors[method], axis=0)
        # inverse error and condition number are clearly smaller (here: >2x, >1.3x)
        assert mean_method[1] < 0.5 * mean_sample[1]
        assert mean_method[2] < 0.75 * mean_sample[2]
        assert mean_method[0] < mean_sample[0]


def test_equicorrelated_world_gets_material_shrinkage():
    x = simulate(equicorrelation(100, 0.5), 252, 0)
    _, info = estimate_correlation(x, "ledoit_wolf")
    assert 0.3 < info["shrinkage"] <= 1.0
    assert info["target_correlation"] == pytest.approx(0.5, abs=0.03)
    # K / T = 0.4 with structure that is not equicorrelated: partial shrinkage
    x = simulate(factor_truth(100), 250, 0)
    _, info = estimate_correlation(x, "ledoit_wolf")
    assert 0.02 < info["shrinkage"] < 0.5


def test_large_sample_limit():
    truth = equicorrelation(5, 0.5)
    x = simulate(truth, 50_000, 3)
    for method in METHODS:
        matrix, info = estimate_correlation(x, method)
        assert np.abs(matrix - truth).max() < 0.015
    # unstructured truth: the Ledoit-Wolf intensity vanishes
    x = simulate(R3, 50_000, 3)
    matrix, info = estimate_correlation(x, "ledoit_wolf")
    assert info["shrinkage_unfloored"] < 0.01
    assert np.abs(matrix - R3).max() < 0.015
    matrix, _ = estimate_correlation(x, "sample")
    assert np.abs(matrix - R3).max() < 0.015
    # a true factor structure is recovered by the factor model at large T
    big = factor_truth(8, 2)
    matrix, info = estimate_correlation(simulate(big, 50_000, 3), "factor")
    assert info["n_factors"] == 2 and np.abs(matrix - big).max() < 0.05


def test_marchenko_pastur_choice_and_overrides():
    truth = factor_truth(60)
    chosen = [
        estimate_correlation(simulate(truth, 250, s), "factor")[1]["n_factors"]
        for s in range(15)
    ]
    assert np.mean(np.array(chosen) == 3) >= 0.8
    x = simulate(truth, 250, 0)
    _, info = estimate_correlation(x, "factor", n_factors=5)
    assert info["n_factors"] == 5 and info["n_factors_source"] == "user"
    _, info = estimate_correlation(x, "factor")
    assert info["n_factors_source"] == "mp_edge"
    # bounds: at least one, at most K - 1
    _, info = estimate_correlation(simulate(np.eye(8), 100, 0), "factor")
    assert info["n_factors"] >= 1
    _, info = estimate_correlation(simulate(equicorrelation(3, 0.99), 5000, 0), "factor")
    assert 1 <= info["n_factors"] <= 2


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ((np.ones((5, 2)),), {}),  # constant series
        ((np.zeros(5),), {}),  # one-dimensional
        ((np.random.default_rng(0).normal(size=(1, 3)),), {}),  # one row
        ((np.random.default_rng(0).normal(size=(9, 3)), "bogus"), {}),
        ((np.random.default_rng(0).normal(size=(9, 3)), "sample"), {"n_factors": 1}),
        ((np.random.default_rng(0).normal(size=(9, 3)), "factor"), {"n_factors": 0}),
        ((np.random.default_rng(0).normal(size=(9, 3)), "factor"), {"n_factors": 3}),
        ((np.random.default_rng(0).normal(size=(9, 3)), "factor"), {"n_factors": 1.5}),
        ((np.array([[1.0, np.inf], [2.0, 3.0], [0.5, 1.0]]),), {}),
        ((np.full((4, 2), np.nan),), {}),
    ],
)
def test_estimate_correlation_rejects_invalid_arguments(args, kwargs):
    with pytest.raises(ValueError):
        estimate_correlation(*args, **kwargs)


def test_from_pnl_passes_method_and_defaults_to_ledoit_wolf(daily_pnl):
    kw = dict(selected="a", n_omega=21, n_tau=21)
    default = BayesianPnLTrials.from_pnl(daily_pnl, **kw)
    assert default.diagnostics["correlation_method"] == "ledoit_wolf"
    assert "shrinkage" in default.diagnostics["correlation_info"]
    sample = BayesianPnLTrials.from_pnl(daily_pnl, correlation_method="sample", **kw)
    assert sample.diagnostics["correlation_method"] == "sample"
    factor = BayesianPnLTrials.from_pnl(
        daily_pnl, correlation_method="factor", n_factors=1, **kw
    )
    assert factor.diagnostics["correlation_info"]["n_factors"] == 1
    expected, _ = estimate_correlation(
        daily_pnl.loc[
            default.periods.period_start.min() : default.periods.period_end.max()
        ],
        "ledoit_wolf",
    )
    np.testing.assert_allclose(default.noise_correlation, expected.to_numpy(), atol=1e-12)
    # long data, few versions: shrinkage is negligible and the fits agree
    np.testing.assert_allclose(
        default.noise_correlation, sample.noise_correlation, atol=0.03
    )
    assert default.expectation() == pytest.approx(sample.expectation(), abs=0.03)
    assert default.prob_mean_above(0.0) == pytest.approx(
        sample.prob_mean_above(0.0), abs=0.03
    )
    with pytest.raises(ValueError):
        BayesianPnLTrials.from_pnl(daily_pnl, correlation_method="bogus")


def test_from_pnl_shrinkage_accepts_more_versions_than_observations():
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2020-01-01", "2020-12-31")
    pnl = pd.DataFrame(
        rng.standard_normal((len(index), 300)) * 0.01,
        index=index,
        columns=["v{}".format(i) for i in range(300)],
    )
    fit = BayesianPnLTrials.from_pnl(pnl, n_omega=11, n_tau=11)
    assert fit.n_variants == 300
    with pytest.raises(ValueError, match="duplicate"):
        BayesianPnLTrials.from_pnl(pnl.assign(dup=pnl.v0))


def test_shrunk_correlation_improves_selection_aware_posterior():
    """K = 60 versions, no true value, R estimated from 250 daily observations.

    The best version is selected on 5 yearly Sharpe ratios with true
    equicorrelation 0.5. P(theta_selected > 0) is below 0.5 + noise only when
    R is estimated well; the sample R over-trusts spurious contrasts and
    inflates it.
    """
    K, observations, periods, seeds = 60, 250, 5, 30
    truth = equicorrelation(K, 0.5)
    factor = np.linalg.cholesky(truth)
    probabilities = {m: [] for m in METHODS}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        sharpes = rng.standard_normal((periods, K)) @ factor.T
        daily = rng.standard_normal((observations, K)) @ factor.T
        best = "V{}".format(int(sharpes.mean(axis=0).argmax()))
        for method in METHODS:
            R, _ = estimate_correlation(daily, method)
            fit = BayesianPnLTrials(
                sharpes,
                R,
                period_years=1,
                noise_sd=1.0,
                selected=best,
                construction_correlation="noise",
                n_tau=31,
                n_omega=31,
            )
            probabilities[method].append(fit.prob_mean_above(0.0))
    mean = {m: float(np.mean(v)) for m, v in probabilities.items()}
    print("mean P(theta_selected > 0):", mean)
    assert mean["ledoit_wolf"] < mean["sample"] - 0.02
    assert mean["factor"] < mean["sample"] - 0.02
    for method in ("ledoit_wolf", "factor"):
        better = np.mean(np.array(probabilities[method]) < np.array(probabilities["sample"]))
        assert better > 0.7
