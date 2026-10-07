import unittest

import numpy as np
import torch

from macrosynergy.learning.forecasting.torch.losses import (
    NegCrossSectionalIC,
    NegRankIC,
    NegSharpeRatioExAnteVol,
    NegThreeComponentLoss,
)


def reference(pred, true, vol=None, side=None, form="ic", a=0.0, d=0.0, min_names=10, clip=5.0):
    """The loss, computed independently in numpy, month by month."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    rows = []
    for t in range(pred.shape[0]):
        ok = np.isfinite(true[t])
        if vol is not None and (form == "sharpe" or a != 0):
            ok &= np.isfinite(vol[t])
        if ok.sum() < min_names:
            continue
        if d != 0 and not (np.isfinite(side[t, 0]) and side[t, 0] > 0):
            continue
        p, r = pred[t][ok], true[t][ok]
        size = 1.0
        if vol is not None and a != 0:
            v = vol[t][ok]
            size *= np.exp(np.mean(np.log(v[v > 0]))) ** (-a)
        if d != 0:
            size *= side[t, 0] ** d
        rows.append((p, r, vol[t][ok] if vol is not None else None, size))
    if not rows:
        return 0.0
    sizes = np.array([row[3] for row in rows])
    sizes = sizes / sizes.mean()
    if clip is not None:
        sizes = np.clip(sizes, 1.0 / clip, clip)
    if form == "ic":
        ics = []
        for p, r, _, _ in rows:
            pc, rc = p - p.mean(), r - r.mean()
            ics.append(pc @ rc / np.sqrt((pc @ pc) * (rc @ rc)))
        return -float(np.sum(sizes * np.array(ics)) / np.sum(sizes))
    rewards, variances = [], []
    for p, r, v, _ in rows:
        s = (p - p.mean()) / p.std()
        rewards.append(s @ r)
        variances.append(np.sum(s ** 2 * v ** 2))
    n = len(rows)
    return -float(np.sum(sizes * np.array(rewards)) / n
                  / np.sqrt(np.sum(sizes ** 2 * np.array(variances)) / n + 1e-8))


class TestNegThreeComponentLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        cls.n_periods, cls.n_assets = 8, 24
        cls.pred = rng.normal(size=(cls.n_periods, cls.n_assets))
        cls.true = rng.normal(size=(cls.n_periods, cls.n_assets)) * 5
        cls.true[0, :5] = np.nan          # 19 observed
        cls.true[1, :20] = np.nan         # 4 observed: below min_names
        month_scale = np.exp(rng.normal(size=(cls.n_periods, 1)))
        cls.vol = np.exp(rng.normal(size=(cls.n_periods, cls.n_assets)) * 0.3) * month_scale * 5
        cls.side = np.exp(rng.normal(size=(cls.n_periods, 1)))
        cls.t = lambda self, x: torch.tensor(x, dtype=torch.float64)

    def loss_value(self, loss, pred=None, true=None, vol=True, side=True):
        pred = self.pred if pred is None else pred
        true = self.true if true is None else true
        kwargs = {}
        if vol and loss.requires_vol:
            kwargs["vol"] = self.t(self.vol)
        if side and loss.side_inputs:
            kwargs["side"] = self.t(self.side)
        return loss(self.t(pred), self.t(true), **kwargs)

    # ------------------------------------------------------------------ construction

    def test_types_init(self):
        with self.assertRaises(ValueError):
            NegThreeComponentLoss(form="mean")
        with self.assertRaises(TypeError):
            NegThreeComponentLoss(vol_exponent="one")
        with self.assertRaises(TypeError):
            NegThreeComponentLoss(dispersion_exponent=True)
        with self.assertRaises(TypeError):
            NegThreeComponentLoss(min_names="ten")
        with self.assertRaises(ValueError):
            NegThreeComponentLoss(min_names=1)
        with self.assertRaises(TypeError):
            NegThreeComponentLoss(rank_targets="yes")
        with self.assertRaises(ValueError):
            NegThreeComponentLoss(size_clip=0.5)
        with self.assertRaises(ValueError):
            NegThreeComponentLoss(eps=0)

    def test_what_fit_must_supply_follows_the_configuration(self):
        plain = NegThreeComponentLoss()
        self.assertFalse(plain.requires_vol)
        self.assertEqual(plain.side_inputs, ())
        self.assertTrue(NegThreeComponentLoss(vol_exponent=-1).requires_vol)
        self.assertTrue(NegThreeComponentLoss(form="sharpe").requires_vol)
        sized = NegThreeComponentLoss(dispersion_exponent=-1)
        self.assertEqual(sized.side_inputs, ("disp_forecast",))
        self.assertFalse(sized.requires_vol)

    def test_sklearn_params_round_trip(self):
        from sklearn.base import clone
        loss = NegThreeComponentLoss(form="sharpe", vol_exponent=1, dispersion_exponent=-1, size_clip=None)
        twin = clone(loss)
        self.assertEqual(twin.get_params(), loss.get_params())
        self.assertTrue(twin.requires_vol)
        self.assertEqual(twin.side_inputs, ("disp_forecast",))

    # ---------------------------------------------------------------- required inputs

    def test_missing_inputs_raise(self):
        pred, true = self.t(self.pred), self.t(self.true)
        with self.assertRaises(ValueError):
            NegThreeComponentLoss(form="sharpe")(pred, true)
        with self.assertRaises(ValueError):
            NegThreeComponentLoss(vol_exponent=1)(pred, true)
        with self.assertRaises(ValueError):
            NegThreeComponentLoss(dispersion_exponent=1)(pred, true)
        with self.assertRaises(ValueError):
            NegThreeComponentLoss(dispersion_exponent=1)(pred, true, side=self.t(self.side[:3]))

    # ------------------------------------------------------------------- the ic form

    def test_no_levers_is_the_ic_loss(self):
        mine = NegThreeComponentLoss()
        for rank, library in ((False, NegCrossSectionalIC()), (True, NegRankIC())):
            ours = NegThreeComponentLoss(rank_targets=rank)
            p1 = self.t(self.pred).requires_grad_(True)
            p2 = self.t(self.pred).requires_grad_(True)
            a, b = ours(p1, self.t(self.true)), library(p2, self.t(self.true))
            self.assertAlmostEqual(float(a), float(b), places=12)
            a.backward(), b.backward()
            torch.testing.assert_close(p1.grad, p2.grad)
        del mine

    def test_ic_form_with_levers_matches_reference(self):
        for a, d in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (1.0, -1.0), (0.5, 0.5)):
            loss = NegThreeComponentLoss(vol_exponent=a, dispersion_exponent=d)
            ref = reference(self.pred, self.true, self.vol, self.side, "ic", a, d)
            self.assertAlmostEqual(float(self.loss_value(loss)), ref, places=10, msg=f"a={a}, d={d}")

    def test_sizes_are_normalised_clipped_and_zero_for_unusable_months(self):
        loss = NegThreeComponentLoss(vol_exponent=1.0, size_clip=2.0)
        true, vol = self.t(self.true), self.t(self.vol)
        mask = torch.isfinite(true) & torch.isfinite(vol)
        usable = mask.sum(dim=1) >= 10
        g = loss.sizes(vol, None, mask, usable, torch.float64)
        self.assertEqual(float(g[1]), 0.0)                       # month 1 has 4 names
        self.assertTrue(bool((g[usable] >= 0.5 - 1e-12).all() and (g[usable] <= 2.0 + 1e-12).all()))
        unclipped = NegThreeComponentLoss(vol_exponent=1.0, size_clip=None)
        h = unclipped.sizes(vol, None, mask, usable, torch.float64)
        self.assertAlmostEqual(float(h[usable].mean()), 1.0, places=12)

    def test_inverse_vol_sizing_gives_a_calm_month_more_weight(self):
        loss = NegThreeComponentLoss(vol_exponent=1.0, size_clip=None)
        vol = self.t(self.vol)
        mask = torch.isfinite(self.t(self.true))
        usable = mask.sum(dim=1) >= 10
        g = loss.sizes(vol, None, mask, usable, torch.float64)
        aggregate = loss.aggregate_vol(vol, mask)
        idx = [i for i in range(self.n_periods) if bool(usable[i])]
        self.assertEqual(sorted(idx, key=lambda i: float(g[i])), sorted(idx, key=lambda i: -float(aggregate[i])))

    def test_aggregate_vol_is_the_geometric_mean(self):
        loss = NegThreeComponentLoss(vol_exponent=1.0)
        vol = self.t(self.vol)
        mask = torch.isfinite(self.t(self.true))
        got = loss.aggregate_vol(vol, mask)
        for t in range(self.n_periods):
            v = self.vol[t][np.isfinite(self.true[t])]
            self.assertAlmostEqual(float(got[t]), float(np.exp(np.mean(np.log(v)))), places=10)

    def test_months_with_unusable_side_input_are_dropped(self):
        side = self.side.copy()
        side[2, 0] = np.nan
        side[3, 0] = -1.0
        loss = NegThreeComponentLoss(dispersion_exponent=1.0)
        got = loss(self.t(self.pred), self.t(self.true), side=self.t(side))
        keep = [t for t in range(self.n_periods) if t not in (2, 3)]
        ref = reference(self.pred[keep], self.true[keep], None, side[keep], "ic", 0.0, 1.0)
        self.assertAlmostEqual(float(got), ref, places=10)

    # ---------------------------------------------------------------- the sharpe form

    def test_sharpe_form_matches_reference(self):
        for a, d in ((0.0, 0.0), (1.0, 0.0), (0.0, -1.0), (1.0, 1.0)):
            loss = NegThreeComponentLoss(form="sharpe", vol_exponent=a, dispersion_exponent=d)
            ref = reference(self.pred, self.true, self.vol, self.side, "sharpe", a, d)
            self.assertAlmostEqual(float(self.loss_value(loss)), ref, places=8, msg=f"a={a}, d={d}")

    def test_a_perfect_ranking_beats_a_reversed_one_in_the_sharpe_form(self):
        loss = NegThreeComponentLoss(form="sharpe")
        true = self.true.copy()
        true[0, :5], true[1, :20] = 0.0, 0.0   # complete the months so every one is usable
        good = float(loss(self.t(true), self.t(true), vol=self.t(self.vol)))
        bad = float(loss(self.t(-true), self.t(true), vol=self.t(self.vol)))
        self.assertLess(good, bad)

    # ---------------------------------------------------------- invariance and gradients

    def test_invariant_to_level_and_scale_of_each_month(self):
        for form in ("ic", "sharpe"):
            loss = NegThreeComponentLoss(form=form, vol_exponent=1.0, dispersion_exponent=-1.0)
            base = float(self.loss_value(loss))
            shifted = self.pred + np.arange(self.n_periods)[:, None] * 3.0
            scaled = self.pred * np.exp(np.arange(self.n_periods))[:, None]
            self.assertAlmostEqual(float(self.loss_value(loss, pred=shifted)), base, places=8)
            self.assertAlmostEqual(float(self.loss_value(loss, pred=scaled)), base, places=8)

    def test_gradient_is_level_free_and_zero_for_unobserved_names(self):
        for form in ("ic", "sharpe"):
            loss = NegThreeComponentLoss(form=form, vol_exponent=1.0, dispersion_exponent=1.0)
            pred = self.t(self.pred).requires_grad_(True)
            loss(pred, self.t(self.true), vol=self.t(self.vol), side=self.t(self.side)).backward()
            grad = pred.grad.numpy()
            self.assertTrue(np.isfinite(grad).all())
            observed = np.isfinite(self.true)
            np.testing.assert_allclose((grad * observed).sum(axis=1), 0.0, atol=1e-10)
            self.assertTrue((grad[~observed] == 0).all())

    def test_a_batch_with_no_usable_month_is_a_no_op_that_keeps_the_graph(self):
        for loss in (NegThreeComponentLoss(), NegThreeComponentLoss(form="sharpe")):
            pred = self.t(self.pred).requires_grad_(True)
            true = torch.full_like(pred, float("nan"))
            out = loss(pred, true, vol=self.t(self.vol)) if loss.requires_vol else loss(pred, true)
            self.assertEqual(float(out.detach()), 0.0)
            out.backward()
            self.assertTrue((pred.grad == 0).all())

    def test_vol_with_nan_cells_masks_those_names(self):
        vol = self.vol.copy()
        vol[4, :10] = np.nan
        loss = NegThreeComponentLoss(form="sharpe")
        got = float(loss(self.t(self.pred), self.t(self.true), vol=self.t(vol)))
        self.assertAlmostEqual(got, reference(self.pred, self.true, vol, None, "sharpe"), places=8)


class TestMissingSecuritiesAreExcluded(unittest.TestCase):
    """A security with no return in a month is EXCLUDED from that month's loss (not filled with a zero
    return that its weight could earn or a zero risk it could add): its weight has exactly zero gradient
    and setting it to anything changes nothing. Holds for DB2's ratio, the IC loss and the new loss."""

    def test_missing_cells_have_no_influence_and_no_gradient(self):
        torch.manual_seed(0)
        T, N = 12, 40
        w = torch.randn(T, N, dtype=torch.float64)
        r = torch.randn(T, N, dtype=torch.float64) * 5
        vol = torch.rand(T, N, dtype=torch.float64) * 8 + 2
        side = torch.rand(T, 1, dtype=torch.float64) + 1
        miss = torch.rand(T, N) < 0.5
        r[miss] = float("nan")
        cases = {
            "db2": (NegSharpeRatioExAnteVol(), {"vol": vol}),
            "ic": (NegCrossSectionalIC(min_names=5), {}),
            "three_sharpe": (NegThreeComponentLoss(form="sharpe", dispersion_exponent=-1, min_names=5), {"vol": vol, "side": side}),
            "three_ic": (NegThreeComponentLoss(form="ic", vol_exponent=1, min_names=5), {"vol": vol}),
        }
        for name, (loss, kwargs) in cases.items():
            with self.subTest(loss=name):
                a = w.clone().requires_grad_(True)
                base = loss(a, r, **kwargs)
                base.backward()
                self.assertTrue(bool((a.grad[miss] == 0).all()))
                for replacement in (0.0, 1e3, -50.0):
                    b = torch.where(miss, torch.full_like(w, replacement), w)
                    self.assertAlmostEqual(float(loss(b, r, **kwargs)), float(base.detach()), places=10)


if __name__ == "__main__":
    unittest.main()
