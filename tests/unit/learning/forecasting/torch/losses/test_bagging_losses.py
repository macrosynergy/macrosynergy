import unittest

import numpy as np
import torch
import torch.nn as nn

from macrosynergy.learning import (
    AssetBaggingLoss,
    LongShortModule,
    NegMeanPortfolioReturn,
    NegSharpeRatio,
)


class TestAssetBaggingLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.n_periods, self.n_assets = 24, 40
        self.raw = torch.randn(self.n_periods, self.n_assets)
        self.returns = torch.randn(self.n_periods, self.n_assets)
        self.long_short = LongShortModule(dollar_neutral=True)(self.raw)
        self.softmax = nn.Softmax(dim=1)(self.raw)

    # ------------------------------------------------------------------ validation

    def test_rejects_non_module_loss(self):
        with self.assertRaises(TypeError):
            AssetBaggingLoss(loss_func="NegSharpeRatio")

    def test_rejects_out_of_range_fraction(self):
        for fraction in (0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                AssetBaggingLoss(NegSharpeRatio(), asset_fraction=fraction)

    def test_rejects_boolean_fraction(self):
        # bool is a Number and an Integral, so it must be excluded explicitly
        with self.assertRaises(TypeError):
            AssetBaggingLoss(NegSharpeRatio(), asset_fraction=True)
        with self.assertRaises(TypeError):
            AssetBaggingLoss(NegSharpeRatio(), draws=True)

    def test_rejects_bad_draws_and_scheme(self):
        with self.assertRaises(ValueError):
            AssetBaggingLoss(NegSharpeRatio(), draws=0)
        with self.assertRaises(ValueError):
            AssetBaggingLoss(NegSharpeRatio(), renormalize="quadratic")

    # ------------------------------------------------------------------- behaviour

    def test_full_fraction_is_exact_passthrough(self):
        """asset_fraction=1 must not perturb the loss at all, so it is a valid control."""
        base = NegSharpeRatio()
        wrapped = AssetBaggingLoss(base, asset_fraction=1.0, draws=5)
        self.assertAlmostEqual(
            float(wrapped(self.long_short, self.returns)),
            float(base(self.long_short, self.returns)),
            places=10,
        )

    def test_subset_is_shared_across_periods(self):
        """
        Every period of a batch must see the same asset subset.

        If the subset were redrawn per period, a column zeroed in one period would be
        present in another. Comparing against a manually-held-out single subset is the
        direct check: with one draw and a seeded generator the wrapper must equal the
        base loss computed on some fixed column subset of the full width.
        """
        generator = torch.Generator().manual_seed(7)
        wrapped = AssetBaggingLoss(
            NegSharpeRatio(), asset_fraction=0.5, draws=1, generator=generator
        )
        value = float(wrapped(self.long_short, self.returns))

        n_draw = int(round(0.5 * self.n_assets))
        replay = torch.Generator().manual_seed(7)
        keep = torch.randperm(self.n_assets, generator=replay)[:n_draw]

        weights = self.long_short[:, keep]
        weights = weights - weights.mean(dim=1, keepdim=True)
        weights = weights / weights.abs().sum(dim=1, keepdim=True)
        expected = float(NegSharpeRatio()(weights, self.returns[:, keep]))

        self.assertAlmostEqual(value, expected, places=6)

    def test_min_assets_floor(self):
        """A tiny fraction must not reduce the cross-section below the floor."""
        wrapped = AssetBaggingLoss(
            NegSharpeRatio(), asset_fraction=0.01, draws=1, min_assets=5
        )
        self.assertTrue(np.isfinite(float(wrapped(self.long_short, self.returns))))

    def test_averaging_reduces_variance_across_calls(self):
        """More draws per call must give a less variable loss, which is the whole point."""

        def spread(draws):
            values = []
            for seed in range(30):
                generator = torch.Generator().manual_seed(seed)
                wrapped = AssetBaggingLoss(
                    NegSharpeRatio(), asset_fraction=0.5, draws=draws, generator=generator
                )
                values.append(float(wrapped(self.long_short, self.returns)))
            return float(np.std(values))

        self.assertLess(spread(draws=8), spread(draws=1))

    # -------------------------------------------------------------- renormalization

    def test_detects_and_restores_dollar_neutral_gross(self):
        """A long-short book's subset must come back neutral and at unit gross."""
        wrapped = AssetBaggingLoss(NegSharpeRatio(), asset_fraction=0.5)
        self.assertEqual(wrapped._detect_scheme(self.long_short), "l1_neutral")

        subset = wrapped._rescale(self.long_short[:, :20], "l1_neutral")
        torch.testing.assert_close(
            subset.sum(dim=1), torch.zeros(self.n_periods), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            subset.abs().sum(dim=1), torch.ones(self.n_periods), atol=1e-6, rtol=0
        )

    def test_detects_and_restores_softmax(self):
        """A long-only book's subset must come back non-negative and summing to one."""
        wrapped = AssetBaggingLoss(NegSharpeRatio(), asset_fraction=0.5)
        self.assertEqual(wrapped._detect_scheme(self.softmax), "sum")

        subset = wrapped._rescale(self.softmax[:, :20], "sum")
        self.assertTrue(bool((subset >= 0).all()))
        torch.testing.assert_close(
            subset.sum(dim=1), torch.ones(self.n_periods), atol=1e-6, rtol=0
        )

    def test_subset_then_modify_equals_modify_then_restore(self):
        """
        Restoring the constraint on a subset must equal having only ever had those names.

        This is the property that makes it legitimate to apply the signal modifier once
        over the full cross-section and subset afterwards, rather than subsetting the raw
        head outputs and applying the modifier to each draw. The two agree because every
        modifier in use is an elementwise map followed by division by a sum over the
        included names, so the subsetting and the normalization commute.

        **It would not hold for a non-separable modifier** -- a hard position cap, a
        top-k selector, a rank transform or any winsorization -- because which names bind
        depends on the set. If one is added, this test fails, and the draw must move
        inside the model's forward instead.
        """
        wrapped = AssetBaggingLoss(NegSharpeRatio(), asset_fraction=0.75)
        keep = torch.randperm(self.n_assets)[:30]

        for modifier, scheme in (
            (LongShortModule(dollar_neutral=True), "l1_neutral"),
            (LongShortModule(dollar_neutral=False), "l1"),
            (nn.Softmax(dim=1), "sum"),
        ):
            with self.subTest(modifier=type(modifier).__name__, scheme=scheme):
                # Applied over the full width, subsetted, then restored
                restored = wrapped._rescale(modifier(self.raw)[:, keep], scheme)
                # Applied to the subset of raw head outputs directly
                direct = modifier(self.raw[:, keep])
                torch.testing.assert_close(restored, direct, atol=1e-6, rtol=0)

    def test_unconstrained_head_is_left_alone(self):
        """Raw outputs satisfy neither convention and must not be silently rescaled."""
        wrapped = AssetBaggingLoss(NegSharpeRatio(), asset_fraction=0.5)
        self.assertEqual(wrapped._detect_scheme(self.raw), "none")
        torch.testing.assert_close(wrapped._rescale(self.raw, "none"), self.raw)

    def test_explicit_scheme_overrides_detection(self):
        """renormalize='none' must skip restoration even for a constrained book."""
        generator = torch.Generator().manual_seed(3)
        wrapped = AssetBaggingLoss(
            NegSharpeRatio(),
            asset_fraction=0.5,
            renormalize="none",
            generator=generator,
        )
        value = float(wrapped(self.long_short, self.returns))

        replay = torch.Generator().manual_seed(3)
        keep = torch.randperm(self.n_assets, generator=replay)[: int(0.5 * self.n_assets)]
        expected = float(
            NegSharpeRatio()(self.long_short[:, keep], self.returns[:, keep])
        )
        self.assertAlmostEqual(value, expected, places=6)

    def test_renormalization_matters_for_a_scale_dependent_loss(self):
        """
        Restoring gross exposure changes a loss that is linear in the weights.

        `NegSharpeRatio` is a ratio and nearly scale-free, so it would hide this. A mean
        portfolio return is linear in the weights, and a subset left un-normalized carries
        less than unit gross, so the two must differ.
        """
        generator = torch.Generator().manual_seed(11)
        without = AssetBaggingLoss(
            NegMeanPortfolioReturn(),
            asset_fraction=0.5,
            renormalize="none",
            generator=generator,
        )
        value_without = float(without(self.long_short, self.returns))

        generator = torch.Generator().manual_seed(11)
        with_scaling = AssetBaggingLoss(
            NegMeanPortfolioReturn(),
            asset_fraction=0.5,
            renormalize="auto",
            generator=generator,
        )
        value_with = float(with_scaling(self.long_short, self.returns))

        self.assertNotAlmostEqual(value_without, value_with, places=4)

    # ----------------------------------------------------------------- gradient flow

    def test_gradient_reaches_the_head(self):
        """Subsetting must not detach the graph, or nothing trains."""
        raw = torch.randn(self.n_periods, self.n_assets, requires_grad=True)
        weights = LongShortModule(dollar_neutral=True)(raw)
        loss = AssetBaggingLoss(NegSharpeRatio(), asset_fraction=0.75, draws=4)(
            weights, self.returns
        )
        loss.backward()

        self.assertIsNotNone(raw.grad)
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        self.assertGreater(float(raw.grad.norm()), 0.0)

    def test_handles_non_finite_returns(self):
        """Missing returns are the base loss's problem, but must not crash the wrapper."""
        returns = self.returns.clone()
        returns[0, :5] = float("nan")
        loss = AssetBaggingLoss(NegSharpeRatio(), asset_fraction=0.75, draws=3)(
            self.long_short, returns
        )
        self.assertTrue(np.isfinite(float(loss)))


if __name__ == "__main__":
    unittest.main()
