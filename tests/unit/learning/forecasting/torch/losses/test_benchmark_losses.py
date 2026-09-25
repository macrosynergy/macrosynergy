import unittest

import numpy as np
import torch

from macrosynergy.learning.forecasting.torch.losses import (
    ActiveReturnLoss,
    ActiveWeightModule,
    BenchmarkFeasibilityPenalty,
    BenchmarkWeightedIC,
    NegCrossSectionalIC,
)


class TestActiveWeightModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(21)
        cls.raw = torch.tensor(rng.normal(size=(7, 15)), dtype=torch.float64)

    def test_types_init(self):
        with self.assertRaises(TypeError):
            ActiveWeightModule(active_share="half")
        with self.assertRaises(ValueError):
            ActiveWeightModule(active_share=0)
        with self.assertRaises(ValueError):
            ActiveWeightModule(active_share=-0.1)
        with self.assertRaises(ValueError):
            ActiveWeightModule(eps=0)

    def test_valid_forward(self):
        for target in (0.1, 0.5, 1.0):
            active = ActiveWeightModule(active_share=target)(self.raw)

            # Sums to zero, so the total book stays fully invested
            torch.testing.assert_close(
                active.sum(dim=1), torch.zeros(self.raw.shape[0], dtype=torch.float64)
            )
            # Realises the requested active share
            torch.testing.assert_close(
                0.5 * active.abs().sum(dim=1),
                torch.full((self.raw.shape[0],), float(target), dtype=torch.float64),
            )

    def test_valid_scale_invariance(self):
        """
        The module fixes the size of the book, so rescaling the raw head outputs changes
        nothing: the network controls where to deviate, not by how much in total.
        """
        module = ActiveWeightModule(active_share=0.3)
        torch.testing.assert_close(module(self.raw), module(self.raw * 100.0))

    def test_valid_degenerate_input(self):
        """A constant row carries no view, and must not divide by zero."""
        constant = torch.ones(3, 15, dtype=torch.float64)
        active = ActiveWeightModule(active_share=0.5)(constant)
        self.assertTrue(torch.isfinite(active).all())
        torch.testing.assert_close(active, torch.zeros_like(active))

    def test_valid_gradients(self):
        raw = self.raw.clone().requires_grad_(True)
        ActiveWeightModule(active_share=0.5)(raw).sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())


class TestActiveReturnLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(22)
        cls.n_periods, cls.n_assets = 12, 15
        raw = rng.normal(size=(cls.n_periods, cls.n_assets))
        cls.raw = torch.tensor(raw, dtype=torch.float64)
        cls.active = ActiveWeightModule(active_share=0.5)(cls.raw)
        cls.returns = torch.tensor(
            rng.normal(size=(cls.n_periods, cls.n_assets)), dtype=torch.float64
        )

    def test_types_init(self):
        with self.assertRaises(TypeError):
            ActiveReturnLoss(objective=3)
        with self.assertRaises(ValueError):
            ActiveReturnLoss(objective="sharpe")
        with self.assertRaises(TypeError):
            ActiveReturnLoss(unbiased="yes")
        with self.assertRaises(ValueError):
            ActiveReturnLoss(reg_tracking_error=-1)

    def test_valid_forward_mean(self):
        loss = ActiveReturnLoss(objective="mean")
        expected = -float((self.active * self.returns).sum(dim=1).mean())
        self.assertAlmostEqual(float(loss(self.active, self.returns)), expected, places=12)

    def test_valid_forward_ir(self):
        loss = ActiveReturnLoss(objective="ir")
        active_returns = (self.active * self.returns).sum(dim=1).numpy()
        expected = -active_returns.mean() / active_returns.std(ddof=1)
        self.assertAlmostEqual(float(loss(self.active, self.returns)), expected, places=8)

    def test_valid_invariant_to_common_return_component(self):
        """
        The documented property that motivates the active parameterisation: because the
        active weights sum to zero, adding any per-period constant to every asset's
        return leaves the active return untouched. The benchmark's own return therefore
        cancels, whether `y_true` holds total or relative returns.
        """
        for objective in ("mean", "ir"):
            loss = ActiveReturnLoss(objective=objective)
            base = float(loss(self.active, self.returns))

            market = torch.tensor(
                np.random.default_rng(5).normal(size=(self.n_periods, 1)) * 10,
                dtype=torch.float64,
            )
            shifted = self.returns + market
            self.assertAlmostEqual(
                float(loss(self.active, shifted)), base, places=8, msg=objective
            )

    def test_valid_masked(self):
        returns = self.returns.clone()
        returns[0, :3] = float("nan")
        loss = ActiveReturnLoss(objective="mean")
        value = float(loss(self.active, returns))
        self.assertTrue(np.isfinite(value))

        masked = self.returns.clone()
        masked[0, :3] = 0.0
        self.assertAlmostEqual(value, float(loss(self.active, masked)), places=12)

    def test_valid_tracking_error_penalty(self):
        reg = 0.5
        vol = torch.tensor(np.linspace(0.5, 2.0, self.n_assets), dtype=torch.float64)
        loss = ActiveReturnLoss(
            objective="mean", reg_tracking_error=reg, asset_vol=vol
        )
        base = -float((self.active * self.returns).sum(dim=1).mean())
        penalty = float(((self.active * vol) ** 2).sum(dim=1).mean())
        self.assertAlmostEqual(
            float(loss(self.active, self.returns)), base + reg * penalty, places=10
        )

    def test_valid_gradients(self):
        for objective in ("mean", "ir"):
            raw = self.raw.clone().requires_grad_(True)
            active = ActiveWeightModule(active_share=0.5)(raw)
            ActiveReturnLoss(objective=objective)(active, self.returns).backward()
            self.assertTrue(torch.isfinite(raw.grad).all(), msg=objective)


class TestBenchmarkWeightedIC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(23)
        cls.n_periods, cls.n_assets = 8, 20
        cls.pred_np = rng.normal(size=(cls.n_periods, cls.n_assets))
        cls.true_np = rng.normal(size=(cls.n_periods, cls.n_assets))
        cls.pred = torch.tensor(cls.pred_np, dtype=torch.float64)
        cls.true = torch.tensor(cls.true_np, dtype=torch.float64)

        weights = rng.random(cls.n_assets) ** 3          # a realistically skewed index
        cls.bench_np = weights / weights.sum()
        cls.bench = torch.tensor(cls.bench_np, dtype=torch.float64)

    def test_types_init(self):
        with self.assertRaises(TypeError):
            BenchmarkWeightedIC(power="one")
        with self.assertRaises(ValueError):
            BenchmarkWeightedIC(power=1.5)
        with self.assertRaises(ValueError):
            BenchmarkWeightedIC(power=-0.1)
        with self.assertRaises(ValueError):
            BenchmarkWeightedIC(min_names=1)

    def test_valid_power_zero_recovers_unweighted_ic(self):
        """`power=0` is the control: it must reproduce `NegCrossSectionalIC` exactly."""
        weighted = BenchmarkWeightedIC(benchmark=self.bench, power=0, min_names=10)
        plain = NegCrossSectionalIC(min_names=10)
        self.assertAlmostEqual(
            float(weighted(self.pred, self.true)),
            float(plain(self.pred, self.true)),
            places=10,
        )

    def test_valid_equal_benchmark_matches_unweighted_ic(self):
        weighted = BenchmarkWeightedIC(benchmark="equal", power=1, min_names=10)
        plain = NegCrossSectionalIC(min_names=10)
        self.assertAlmostEqual(
            float(weighted(self.pred, self.true)),
            float(plain(self.pred, self.true)),
            places=10,
        )

    def test_valid_forward(self):
        """Weighted correlation computed independently under the benchmark measure."""
        loss = BenchmarkWeightedIC(benchmark=self.bench, power=1, min_names=10)

        ics = []
        for p, t in zip(self.pred_np, self.true_np):
            q = self.bench_np / self.bench_np.sum()
            pc = p - q @ p
            tc = t - q @ t
            ics.append((q * pc * tc).sum() / np.sqrt((q * pc**2).sum() * (q * tc**2).sum()))
        self.assertAlmostEqual(
            float(loss(self.pred, self.true)), -float(np.mean(ics)), places=10
        )

    def test_valid_weighting_changes_the_objective(self):
        """
        The weighting must actually bite: a perfect ranking among the largest names and a
        reversed one among the smallest should score better under weighting than a book
        that gets the small names right and the large ones wrong.
        """
        order = np.argsort(-self.bench_np)
        large, small = order[:5], order[5:]

        good_large = np.zeros_like(self.true_np)
        good_large[:, large] = self.true_np[:, large]
        good_large[:, small] = -self.true_np[:, small]

        good_small = np.zeros_like(self.true_np)
        good_small[:, large] = -self.true_np[:, large]
        good_small[:, small] = self.true_np[:, small]

        weighted = BenchmarkWeightedIC(benchmark=self.bench, power=1, min_names=10)
        a = float(weighted(torch.tensor(good_large), self.true))
        b = float(weighted(torch.tensor(good_small), self.true))
        self.assertLess(a, b)

    def test_valid_period_varying_benchmark(self):
        """A (batch_size, n_assets) benchmark applies its own weights to each period."""
        rng = np.random.default_rng(31)
        bench = rng.random((self.n_periods, self.n_assets))
        bench = bench / bench.sum(axis=1, keepdims=True)
        loss = BenchmarkWeightedIC(
            benchmark=torch.tensor(bench, dtype=torch.float64), power=1, min_names=10
        )

        ics = []
        for p, t, q in zip(self.pred_np, self.true_np, bench):
            pc, tc = p - q @ p, t - q @ t
            ics.append((q * pc * tc).sum() / np.sqrt((q * pc**2).sum() * (q * tc**2).sum()))
        self.assertAlmostEqual(
            float(loss(self.pred, self.true)), -float(np.mean(ics)), places=10
        )

    def test_valid_masked_renormalises_the_benchmark(self):
        """
        A name with no observed return is dropped from the benchmark and the remaining
        weights renormalised, rather than held at a stale weight.
        """
        true = self.true.clone()
        true[0, :4] = float("nan")
        loss = BenchmarkWeightedIC(benchmark=self.bench, power=1, min_names=10)
        value = float(loss(self.pred, true))
        self.assertTrue(np.isfinite(value))

        # Period 0 scored on the surviving names alone, under a renormalised benchmark
        observed = np.arange(self.n_assets) >= 4
        q = self.bench_np[observed] / self.bench_np[observed].sum()
        p, t = self.pred_np[0][observed], self.true_np[0][observed]
        pc, tc = p - q @ p, t - q @ t
        ic0 = (q * pc * tc).sum() / np.sqrt((q * pc**2).sum() * (q * tc**2).sum())

        rest = []
        for p, t in zip(self.pred_np[1:], self.true_np[1:]):
            q = self.bench_np
            pc, tc = p - q @ p, t - q @ t
            rest.append((q * pc * tc).sum() / np.sqrt((q * pc**2).sum() * (q * tc**2).sum()))
        self.assertAlmostEqual(value, -float(np.mean([ic0] + rest)), places=10)

    def test_valid_gradients(self):
        pred = self.pred.clone().requires_grad_(True)
        BenchmarkWeightedIC(benchmark=self.bench, power=0.5, min_names=10)(
            pred, self.true
        ).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad.abs().sum()), 0)


class TestBenchmarkFeasibilityPenalty(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_assets = 10
        weights = np.arange(1, cls.n_assets + 1, dtype=float)
        cls.bench_np = weights / weights.sum()
        cls.bench = torch.tensor(cls.bench_np, dtype=torch.float64)
        cls.true = torch.zeros(3, cls.n_assets, dtype=torch.float64)

    def test_types_init(self):
        with self.assertRaises(TypeError):
            BenchmarkFeasibilityPenalty(reg_feasibility="one")
        with self.assertRaises(ValueError):
            BenchmarkFeasibilityPenalty(reg_feasibility=-1)

    def test_valid_feasible_book_is_free(self):
        """Any book that stays long-only costs nothing."""
        active = torch.zeros(3, self.n_assets, dtype=torch.float64)
        penalty = BenchmarkFeasibilityPenalty(benchmark=self.bench)
        self.assertEqual(float(penalty(active, self.true)), 0.0)

        # Underweight each name by exactly half its benchmark weight: still feasible
        active = -0.5 * self.bench.unsqueeze(0).expand(3, self.n_assets).contiguous()
        self.assertEqual(float(penalty(active, self.true)), 0.0)

    def test_valid_infeasible_book_is_penalised(self):
        active = torch.zeros(3, self.n_assets, dtype=torch.float64)
        active[:, 0] = -0.5          # far below the smallest benchmark weight
        penalty = BenchmarkFeasibilityPenalty(benchmark=self.bench)

        shortfall = 0.5 - self.bench_np[0]
        self.assertAlmostEqual(
            float(penalty(active, self.true)), shortfall**2, places=12
        )

    def test_valid_room_scales_with_benchmark_weight(self):
        """
        The point of the term: the same underweight is feasible in a large index name and
        infeasible in a small one.
        """
        penalty = BenchmarkFeasibilityPenalty(benchmark=self.bench)
        size = float(self.bench_np[-1]) * 0.9   # under the largest weight

        on_large = torch.zeros(3, self.n_assets, dtype=torch.float64)
        on_large[:, -1] = -size
        self.assertEqual(float(penalty(on_large, self.true)), 0.0)

        on_small = torch.zeros(3, self.n_assets, dtype=torch.float64)
        on_small[:, 0] = -size
        self.assertGreater(float(penalty(on_small, self.true)), 0.0)

    def test_valid_reg_scales_the_penalty(self):
        active = torch.zeros(3, self.n_assets, dtype=torch.float64)
        active[:, 0] = -0.5
        single = BenchmarkFeasibilityPenalty(benchmark=self.bench, reg_feasibility=1)
        double = BenchmarkFeasibilityPenalty(benchmark=self.bench, reg_feasibility=2)
        self.assertAlmostEqual(
            2 * float(single(active, self.true)),
            float(double(active, self.true)),
            places=12,
        )

    def test_valid_gradients(self):
        active = torch.zeros(3, self.n_assets, dtype=torch.float64)
        active[:, 0] = -0.5
        active = active.requires_grad_(True)
        BenchmarkFeasibilityPenalty(benchmark=self.bench)(active, self.true).backward()
        self.assertTrue(torch.isfinite(active.grad).all())
        # Only the violating position is pushed
        self.assertGreater(abs(float(active.grad[0, 0])), 0)
        self.assertEqual(float(active.grad[0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
