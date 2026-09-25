import unittest

import numpy as np
import torch
from parameterized import parameterized

from macrosynergy.learning.forecasting.torch.samplers import (
    PanelBatchSampler,
    TimeSeriesSampler,
)


class TestPanelBatchSampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A balanced panel: 12 periods x 5 assets, laid out asset-major as
        # `sort_index()` on a (cid, real_date) MultiIndex would produce
        cls.n_periods = 12
        cls.n_assets = 5
        periods, assets = [], []
        for asset in range(cls.n_assets):
            for period in range(cls.n_periods):
                assets.append(asset)
                periods.append(period)
        cls.period_index = np.array(periods)
        cls.asset_index = np.array(assets)
        cls.n_rows = len(periods)

    def batches(self, **kwargs):
        kwargs.setdefault("period_index", self.period_index)
        kwargs.setdefault("shuffle", False)
        return list(iter(PanelBatchSampler(**kwargs)))

    def test_types_init(self):
        """period_index"""
        with self.assertRaises(ValueError):
            PanelBatchSampler(period_index=np.zeros((4, 2)))
        with self.assertRaises(ValueError):
            PanelBatchSampler(period_index=np.array([]))

        """mode"""
        with self.assertRaises(TypeError):
            PanelBatchSampler(period_index=self.period_index, mode=3)
        with self.assertRaises(ValueError):
            PanelBatchSampler(period_index=self.period_index, mode="security")

        """asset_index"""
        # required by the two modes whose batches are defined by their assets
        for mode in ("asset", "block"):
            with self.assertRaises(ValueError):
                PanelBatchSampler(
                    period_index=self.period_index, mode=mode, batch_assets=2
                )
        with self.assertRaises(ValueError):
            PanelBatchSampler(
                period_index=self.period_index, asset_index=self.asset_index[:-1]
            )

        """batch_periods"""
        with self.assertRaises(TypeError):
            PanelBatchSampler(period_index=self.period_index, batch_periods="four")
        with self.assertRaises(ValueError):
            PanelBatchSampler(period_index=self.period_index, batch_periods=0)

        """batch_assets"""
        with self.assertRaises(ValueError):
            PanelBatchSampler(
                period_index=self.period_index,
                asset_index=self.asset_index,
                mode="asset",
            )
        with self.assertRaises(TypeError):
            PanelBatchSampler(
                period_index=self.period_index,
                asset_index=self.asset_index,
                mode="asset",
                batch_assets="two",
            )
        with self.assertRaises(ValueError):
            PanelBatchSampler(
                period_index=self.period_index,
                asset_index=self.asset_index,
                mode="asset",
                batch_assets=0,
            )

        """shuffle, aggregate_last, drop_last"""
        with self.assertRaises(TypeError):
            PanelBatchSampler(period_index=self.period_index, shuffle="yes")
        with self.assertRaises(TypeError):
            PanelBatchSampler(period_index=self.period_index, aggregate_last="yes")
        with self.assertRaises(TypeError):
            PanelBatchSampler(period_index=self.period_index, drop_last="yes")
        with self.assertRaises(ValueError):
            PanelBatchSampler(
                period_index=self.period_index, aggregate_last=True, drop_last=True
            )

        """seed"""
        with self.assertRaises(TypeError):
            PanelBatchSampler(period_index=self.period_index, seed=1.5)
        with self.assertRaises(ValueError):
            PanelBatchSampler(period_index=self.period_index, seed=-1)

    @parameterized.expand(
        [
            ("period", dict(mode="period", batch_periods=5)),
            ("asset", dict(mode="asset", batch_assets=2)),
            ("block", dict(mode="block", batch_periods=5, batch_assets=2)),
            ("period_exact", dict(mode="period", batch_periods=4)),
            ("block_exact", dict(mode="block", batch_periods=4, batch_assets=1)),
        ]
    )
    def test_partitions_every_row_exactly_once(self, _name, kwargs):
        """
        Every row must appear in exactly one batch per epoch, in every mode.
        """
        kwargs = dict(kwargs)
        if kwargs["mode"] in ("asset", "block"):
            kwargs["asset_index"] = self.asset_index

        for shuffle in (False, True):
            batches = self.batches(shuffle=shuffle, seed=0, **kwargs)
            flat = np.concatenate([np.asarray(b) for b in batches])
            self.assertEqual(
                len(flat), self.n_rows, msg=f"{_name}, shuffle={shuffle}: row count"
            )
            np.testing.assert_array_equal(
                np.sort(flat),
                np.arange(self.n_rows),
                err_msg=f"{_name}, shuffle={shuffle}: rows not a partition",
            )

    @parameterized.expand(
        [
            (batch_size, aggregate_last, drop_last)
            for batch_size in (3, 5, 7, 12, 20)
            for aggregate_last, drop_last in ((True, False), (False, True), (False, False))
        ]
    )
    def test_period_mode_reproduces_timeseries_sampler(
        self, batch_size, aggregate_last, drop_last
    ):
        """
        With one row per period, "period" mode must reproduce `TimeSeriesSampler` exactly,
        including its handling of a short final block.
        """
        n = 12
        dataset = torch.utils.data.TensorDataset(torch.zeros(n, 2), torch.zeros(n, 1))

        reference = list(
            iter(
                TimeSeriesSampler(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    aggregate_last=aggregate_last,
                    drop_last=drop_last,
                )
            )
        )
        produced = self.batches(
            period_index=np.arange(n),
            mode="period",
            batch_periods=batch_size,
            aggregate_last=aggregate_last,
            drop_last=drop_last,
        )
        self.assertEqual(produced, reference)

    def test_valid_period_mode(self):
        # A period block carries every asset of its periods
        batches = self.batches(mode="period", batch_periods=4)
        self.assertEqual(len(batches), 3)
        for batch in batches:
            self.assertEqual(len(batch), 4 * self.n_assets)
            self.assertEqual(len(set(self.period_index[batch])), 4)
            self.assertEqual(len(set(self.asset_index[batch])), self.n_assets)
            # Periods within a block are contiguous
            observed = sorted(set(self.period_index[batch]))
            self.assertEqual(observed, list(range(observed[0], observed[0] + 4)))

    def test_valid_asset_mode(self):
        # An asset group carries every period of its assets
        batches = self.batches(
            mode="asset", asset_index=self.asset_index, batch_assets=2
        )
        self.assertEqual(len(batches), 3)  # 5 assets in groups of 2
        for batch in batches:
            self.assertEqual(len(set(self.period_index[batch])), self.n_periods)
            self.assertLessEqual(len(set(self.asset_index[batch])), 2)

    def test_valid_block_mode(self):
        # A block is the intersection of a period block and an asset group
        batches = self.batches(
            mode="block",
            asset_index=self.asset_index,
            batch_periods=4,
            batch_assets=2,
        )
        # 3 period blocks x 3 asset groups
        self.assertEqual(len(batches), 9)
        self.assertEqual(len(batches), len(
            PanelBatchSampler(
                period_index=self.period_index,
                asset_index=self.asset_index,
                mode="block",
                batch_periods=4,
                batch_assets=2,
            )
        ))
        for batch in batches:
            self.assertLessEqual(len(set(self.period_index[batch])), 4)
            self.assertLessEqual(len(set(self.asset_index[batch])), 2)

    def test_types_init_mc(self):
        base = dict(period_index=self.period_index, asset_index=self.asset_index, mode="mc")
        # exactly one of batch_assets or asset_fraction
        with self.assertRaises(ValueError):
            PanelBatchSampler(**base)
        with self.assertRaises(ValueError):
            PanelBatchSampler(**base, batch_assets=3, asset_fraction=0.5)
        with self.assertRaises(TypeError):
            PanelBatchSampler(**base, asset_fraction="most")
        with self.assertRaises(ValueError):
            PanelBatchSampler(**base, asset_fraction=0)
        with self.assertRaises(ValueError):
            PanelBatchSampler(**base, asset_fraction=1.5)
        with self.assertRaises(TypeError):
            PanelBatchSampler(**base, asset_fraction=0.5, draws_per_block="two")
        with self.assertRaises(ValueError):
            PanelBatchSampler(**base, asset_fraction=0.5, draws_per_block=0)

    def test_valid_mc_draw_size(self):
        """Each batch holds one period block and a draw of the requested size."""
        sampler = PanelBatchSampler(
            period_index=self.period_index,
            asset_index=self.asset_index,
            mode="mc",
            batch_periods=4,
            asset_fraction=0.6,
            draws_per_block=3,
            seed=0,
        )
        n_draw = int(round(0.6 * self.n_assets))
        batches = list(iter(sampler))

        # 3 period blocks x 3 draws
        self.assertEqual(len(batches), 9)
        self.assertEqual(len(sampler), 9)
        for batch in batches:
            self.assertEqual(len(set(self.asset_index[batch])), n_draw)
            self.assertLessEqual(len(set(self.period_index[batch])), 4)
            # A draw never splits a period block
            observed = sorted(set(self.period_index[batch]))
            self.assertEqual(observed, list(range(observed[0], observed[-1] + 1)))

    def test_valid_mc_draws_overlap(self):
        """
        Unlike "block", the draws are independent: assets recur across them rather than
        being partitioned. That resampling is the point of the mode.
        """
        sampler = PanelBatchSampler(
            period_index=self.period_index,
            asset_index=self.asset_index,
            mode="mc",
            batch_periods=self.n_periods,   # a single block, so draws are comparable
            asset_fraction=0.75,
            draws_per_block=6,
            shuffle=False,
            seed=1,
        )
        draws = [set(self.asset_index[b]) for b in sampler]
        self.assertEqual(len(draws), 6)

        # Some asset appears in more than one draw, and the draws are not identical
        overlaps = [len(a & b) for i, a in enumerate(draws) for b in draws[i + 1:]]
        self.assertTrue(any(o > 0 for o in overlaps))
        self.assertTrue(any(a != b for a in draws for b in draws))

    def test_valid_mc_is_not_a_partition(self):
        """
        Rows recur and some are missed, with mean multiplicity
        draws_per_block * asset_fraction.
        """
        fraction, draws = 0.75, 4
        sampler = PanelBatchSampler(
            period_index=self.period_index,
            asset_index=self.asset_index,
            mode="mc",
            batch_periods=4,
            asset_fraction=fraction,
            draws_per_block=draws,
            seed=2,
        )
        counts = np.bincount(
            np.concatenate([np.asarray(b) for b in sampler]), minlength=self.n_rows
        )
        self.assertEqual(len(counts), self.n_rows)
        self.assertGreater(counts.max(), 1)
        # The realised fraction is the rounded draw size over the panel, which differs
        # from the requested fraction on a small cross-section
        realised_fraction = sampler._n_draw / self.n_assets
        self.assertAlmostEqual(counts.mean(), draws * realised_fraction, places=10)

    def test_valid_mc_default_draws_cover_the_panel(self):
        """The default draw count sees every row at least once in expectation."""
        for fraction in (0.75, 0.5, 0.25):
            sampler = PanelBatchSampler(
                period_index=self.period_index,
                asset_index=self.asset_index,
                mode="mc",
                batch_periods=4,
                asset_fraction=fraction,
                seed=3,
            )
            self.assertGreaterEqual(sampler._draws * fraction, 1.0)

    def test_valid_mc_batch_assets_equivalent_to_fraction(self):
        kwargs = dict(
            period_index=self.period_index,
            asset_index=self.asset_index,
            mode="mc",
            batch_periods=4,
            draws_per_block=2,
            shuffle=False,
            seed=4,
        )
        by_count = list(iter(PanelBatchSampler(batch_assets=3, **kwargs)))
        by_fraction = list(iter(PanelBatchSampler(asset_fraction=3 / self.n_assets, **kwargs)))
        self.assertEqual([len(b) for b in by_count], [len(b) for b in by_fraction])

    def test_valid_mc_reproducible(self):
        kwargs = dict(
            period_index=self.period_index,
            asset_index=self.asset_index,
            mode="mc",
            batch_periods=4,
            asset_fraction=0.6,
            draws_per_block=3,
        )
        first = list(iter(PanelBatchSampler(seed=5, **kwargs)))
        again = list(iter(PanelBatchSampler(seed=5, **kwargs)))
        other = list(iter(PanelBatchSampler(seed=6, **kwargs)))
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_valid_shuffle_is_reproducible(self):
        # The same seed must give the same epoch, and a different seed a different one
        kwargs = dict(
            mode="block",
            asset_index=self.asset_index,
            batch_periods=4,
            batch_assets=2,
            shuffle=True,
        )
        first = self.batches(seed=0, **kwargs)
        again = self.batches(seed=0, **kwargs)
        other = self.batches(seed=1, **kwargs)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_valid_shuffle_preserves_period_contiguity(self):
        """
        Shuffling reorders batches and regroups assets; it must never split a period
        block, since keeping a batch inside one regime is the point of blocking.
        """
        batches = self.batches(
            mode="block",
            asset_index=self.asset_index,
            batch_periods=4,
            batch_assets=2,
            shuffle=True,
            seed=7,
        )
        for batch in batches:
            observed = sorted(set(self.period_index[batch]))
            self.assertEqual(observed, list(range(observed[0], observed[-1] + 1)))
            self.assertLessEqual(len(observed), 4)

    def test_valid_unbalanced_panel(self):
        """
        On an unbalanced panel an asset group can miss a period block entirely; those
        empty batches are dropped rather than yielded.
        """
        # Asset 0 is observed only in the first three periods
        periods = list(range(3)) + list(range(self.n_periods)) * 2
        assets = [0] * 3 + [1] * self.n_periods + [2] * self.n_periods
        batches = self.batches(
            period_index=np.array(periods),
            asset_index=np.array(assets),
            mode="block",
            batch_periods=4,
            batch_assets=1,
        )
        flat = np.concatenate([np.asarray(b) for b in batches])
        np.testing.assert_array_equal(np.sort(flat), np.arange(len(periods)))
        self.assertTrue(all(len(b) > 0 for b in batches))
        # Asset 0 contributes one block, assets 1 and 2 three each
        self.assertEqual(len(batches), 7)

    def test_valid_dataloader_integration(self):
        """The sampler must drive a DataLoader as a batch_sampler."""
        X = torch.arange(self.n_rows * 2, dtype=torch.float32).reshape(self.n_rows, 2)
        y = torch.zeros(self.n_rows, 1)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_sampler=PanelBatchSampler(
                period_index=self.period_index,
                asset_index=self.asset_index,
                mode="block",
                batch_periods=4,
                batch_assets=2,
                shuffle=True,
                seed=0,
            ),
        )
        seen = sum(batch[0].shape[0] for batch in loader)
        self.assertEqual(seen, self.n_rows)


if __name__ == "__main__":
    unittest.main()
