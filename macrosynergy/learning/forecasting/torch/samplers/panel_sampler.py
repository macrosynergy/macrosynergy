import numbers

import numpy as np
import torch

from torch.utils.data import Sampler


class PanelBatchSampler(Sampler):
    """
    Batch sampler for panel datasets, batching over periods, over securities, or over
    blocks of both.

    Parameters
    ----------
    period_index : array-like
        Period label of every row of the dataset. Labels are ordered by their sorted
        unique values, so any orderable dtype works.
    asset_index : array-like, optional
        Asset label of every row of the dataset. Required for the "asset" and "block"
        modes and ignored by "period". Default is None.
    mode : str, optional
        One of "period", "asset", "block" or "mc". Default is "period".
    batch_periods : int, optional
        Number of consecutive periods per batch, used by "period", "block" and "mc".
        Default is 32.
    batch_assets : int, optional
        Number of assets per batch, used by "asset", "block" and "mc". Default is None.
        For "mc" it may be given as `asset_fraction` instead.
    asset_fraction : float, optional
        Fraction of the assets drawn per batch, used by "mc" as an alternative to
        `batch_assets`. Default is None.
    draws_per_block : int, optional
        Number of independent asset draws taken per period block, used by "mc". Default
        is None, which takes `ceil(1 / asset_fraction)` so that each row is seen at least
        once per epoch in expectation.
    shuffle : bool, optional
        Whether to shuffle the order in which batches are visited, and — in the modes
        that subsample assets — which assets are grouped together. Default is True.
    aggregate_last : bool, optional
        Whether to merge a short final period block into the previous one. Cannot be
        combined with `drop_last`. Default is True.
    drop_last : bool, optional
        Whether to drop a short final period block. Cannot be combined with
        `aggregate_last`. Default is False.
    seed : int, optional
        Seed for the asset regrouping and batch ordering. Default is None, which draws
        from the ambient torch random state.

    Notes
    -----
    The choice of mode is not independent of the loss function, and getting the pairing
    wrong changes the objective silently rather than raising.

    A loss that is a **sum over periods** — a per-period cross-sectional correlation, or
    a per-period risk penalty — is invariant to how rows are grouped into batches.
    Subsampling assets makes each period's statistic noisier but leaves the quantity
    being optimised the same. Any of the three modes is admissible.

    A loss that is a **ratio over the batch** — `NegSharpeRatio` and its relatives, which
    collapse assets into one portfolio return per row and then take a mean and a standard
    deviation over the rows — is not. Subsampling assets changes which portfolio is being
    scored, and batching by asset removes the time dimension the ratio is taken over. Only
    "period" is admissible for those.

    The three modes, and what each is for:

    - **"period"** puts every asset of a contiguous run of periods in one batch. This
      keeps each batch within a single regime, which is what makes a batch-level
      time-series statistic meaningful, and it is the only mode that presents a complete
      cross-section. With one row per period it reproduces `TimeSeriesSampler` exactly.
    - **"asset"** puts every period of a subset of assets in one batch. It yields many
      more gradient steps per epoch when assets outnumber periods, at the cost of
      spanning every regime within each batch — the opposite of what "period" is for.
    - **"block"** intersects the two: a contiguous run of periods restricted to a subset
      of assets. This is the middle ground. It multiplies the number of distinct batches
      combinatorially without ever mixing regimes, and the cross-section it presents is
      partial but of controllable size. Where a cross-sectional statistic is being
      optimised, `batch_assets` should be kept well above the point at which that
      statistic is dominated by sampling noise.
    - **"mc"** holds the period block fixed and draws an *independent* random subset of
      the assets for each of `draws_per_block` passes over it. Unlike "block", the draws
      overlap: an asset may appear in several of them, or in none.

    In "period", "asset" and "block" every row appears in exactly one batch per epoch,
    because the periods are partitioned into contiguous blocks, the assets into disjoint
    groups, and a batch is one block, one group, or one (block, group) pair. **"mc" is
    deliberately not a partition**: each row appears a random number of times, with mean
    `draws_per_block * asset_fraction`.

    That resampling is the point of the mode, and what it buys is worth stating precisely,
    because it is not what it first looks like. Subsampling does **not** smooth
    idiosyncratic moves within a batch — a smaller cross-section is a worse estimate of
    the full-panel gradient, so each individual batch is noisier. What it does is prevent
    the fit from depending on any one asset's particular realisation, because that asset
    is missing from a fraction of the draws. The smoothing lives in the parameters
    averaged over draws, not inside any one batch. This is bagging, and structurally it is
    the resampled efficiency of the mean-variance literature.

    One consequence to expect: renormalising weights over a fraction `f` of the assets
    scales idiosyncratic portfolio volatility by `1 / sqrt(f)` while leaving systematic
    volatility unchanged, so any objective containing a realised portfolio volatility —
    `NegSharpeRatio`, or `RankingRiskLoss` with `risk="batch_vol"` — is biased by the
    subsampling, and biased more for a concentrated book. Pair "mc" with a per-period
    objective.
    """

    def __init__(
        self,
        period_index,
        asset_index=None,
        mode="period",
        batch_periods=32,
        batch_assets=None,
        asset_fraction=None,
        draws_per_block=None,
        shuffle=True,
        aggregate_last=True,
        drop_last=False,
        seed=None,
    ):
        self._check_init_params(
            period_index=period_index,
            asset_index=asset_index,
            mode=mode,
            batch_periods=batch_periods,
            batch_assets=batch_assets,
            asset_fraction=asset_fraction,
            draws_per_block=draws_per_block,
            shuffle=shuffle,
            aggregate_last=aggregate_last,
            drop_last=drop_last,
            seed=seed,
        )

        self.mode = mode
        self.batch_periods = batch_periods
        self.batch_assets = batch_assets
        self.asset_fraction = asset_fraction
        self.draws_per_block = draws_per_block
        self.shuffle = shuffle
        self.aggregate_last = aggregate_last
        self.drop_last = drop_last
        self.seed = seed

        # Map the labels onto contiguous integer codes, ordered by their sorted unique
        # values, so that "consecutive periods" means what it says whatever the dtype
        self.periods, period_codes = np.unique(np.asarray(period_index), return_inverse=True)
        self.n_periods = len(self.periods)

        if asset_index is not None:
            self.assets, asset_codes = np.unique(np.asarray(asset_index), return_inverse=True)
            self.n_assets = len(self.assets)
            self.asset_codes = asset_codes
        else:
            self.assets = None
            self.n_assets = 0
            self.asset_codes = None

        self.n_rows = len(period_codes)

        # Rows of each period, in dataset order. Row order within a batch is preserved so
        # that a positional penalty such as `reg_turnover` still sees adjacent periods
        # adjacent when there is one row per period
        order = np.argsort(period_codes, kind="stable")
        boundaries = np.searchsorted(period_codes[order], np.arange(self.n_periods + 1))
        self._rows_by_period = [
            np.sort(order[boundaries[i] : boundaries[i + 1]]) for i in range(self.n_periods)
        ]

        self.period_blocks = self._create_period_blocks(
            self.n_periods, self.batch_periods, self.aggregate_last, self.drop_last
        )

        if self.mode in ("asset", "block"):
            self._n_asset_groups = int(np.ceil(self.n_assets / self.batch_assets))
        else:
            self._n_asset_groups = 1

        if self.mode == "mc":
            if self.batch_assets is not None:
                self._n_draw = int(self.batch_assets)
                fraction = self._n_draw / self.n_assets
            else:
                fraction = float(self.asset_fraction)
                self._n_draw = max(1, int(round(fraction * self.n_assets)))
            self._n_draw = min(self._n_draw, self.n_assets)
            # Enough draws that every row is seen at least once per epoch in expectation
            self._draws = (
                int(self.draws_per_block)
                if self.draws_per_block is not None
                else int(np.ceil(1.0 / fraction))
            )
        else:
            self._n_draw = None
            self._draws = 1

    @staticmethod
    def _create_period_blocks(n_periods, batch_periods, aggregate_last, drop_last):
        """
        Partition the periods into contiguous blocks.

        Mirrors `TimeSeriesSampler._create_batches` so that "period" mode reproduces it
        exactly when the dataset carries one row per period.
        """
        blocks = [
            list(range(start, min(start + batch_periods, n_periods)))
            for start in range(0, n_periods, batch_periods)
        ]
        if aggregate_last:
            if len(blocks) > 1 and len(blocks[-1]) < batch_periods:
                blocks[-2].extend(blocks[-1])
                blocks = blocks[:-1]
        if drop_last:
            if len(blocks) > 1 and len(blocks[-1]) < batch_periods:
                blocks = blocks[:-1]
        return blocks

    def _asset_groups(self, generator):
        """Partition the assets into disjoint groups, regrouped per epoch if shuffling."""
        if self.shuffle:
            order = torch.randperm(self.n_assets, generator=generator).numpy()
        else:
            order = np.arange(self.n_assets)
        return [
            order[start : start + self.batch_assets]
            for start in range(0, self.n_assets, self.batch_assets)
        ]

    def __iter__(self):
        """Generator for batch row indices."""
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(self.seed))

        if self.mode == "period":
            batches = [
                np.concatenate([self._rows_by_period[p] for p in block])
                for block in self.period_blocks
            ]
        elif self.mode == "mc":
            batches = []
            in_draw = np.zeros(self.n_assets, dtype=bool)
            for block in self.period_blocks:
                block_rows = np.concatenate([self._rows_by_period[p] for p in block])
                block_assets = self.asset_codes[block_rows]
                for _ in range(self._draws):
                    drawn = torch.randperm(self.n_assets, generator=generator)[
                        : self._n_draw
                    ].numpy()
                    in_draw[:] = False
                    in_draw[drawn] = True
                    batches.append(block_rows[in_draw[block_assets]])
        else:
            groups = self._asset_groups(generator)
            in_group = np.zeros(self.n_assets, dtype=np.int64)
            for group_idx, group in enumerate(groups):
                in_group[group] = group_idx
            row_group = in_group[self.asset_codes]

            if self.mode == "asset":
                batches = [
                    np.sort(np.flatnonzero(row_group == group_idx))
                    for group_idx in range(len(groups))
                ]
            else:  # "block"
                batches = []
                for block in self.period_blocks:
                    block_rows = np.concatenate([self._rows_by_period[p] for p in block])
                    block_groups = row_group[block_rows]
                    for group_idx in range(len(groups)):
                        batches.append(block_rows[block_groups == group_idx])

        # An asset group can miss a period block entirely on an unbalanced panel
        batches = [b for b in batches if len(b) > 0]

        if self.shuffle:
            order = torch.randperm(len(batches), generator=generator).tolist()
        else:
            order = range(len(batches))

        for idx in order:
            yield batches[idx].tolist()

    def __len__(self):
        """Upper bound on the number of batches per epoch."""
        if self.mode == "mc":
            return len(self.period_blocks) * self._draws
        return len(self.period_blocks) * self._n_asset_groups

    def _check_init_params(
        self,
        period_index,
        asset_index,
        mode,
        batch_periods,
        batch_assets,
        asset_fraction,
        draws_per_block,
        shuffle,
        aggregate_last,
        drop_last,
        seed,
    ):
        # period_index
        period_index = np.asarray(period_index)
        if period_index.ndim != 1:
            raise ValueError("period_index must be one-dimensional.")
        if len(period_index) == 0:
            raise ValueError("period_index must not be empty.")

        # mode
        if not isinstance(mode, str):
            raise TypeError("mode must be a string.")
        if mode not in {"period", "asset", "block", "mc"}:
            raise ValueError("mode must be one of 'period', 'asset', 'block' or 'mc'.")

        # asset_index
        if mode in ("asset", "block", "mc"):
            if asset_index is None:
                raise ValueError(
                    "asset_index must be provided when mode is 'asset', 'block' or "
                    "'mc', since a batch is defined by which assets it contains."
                )
            if len(np.asarray(asset_index)) != len(period_index):
                raise ValueError("asset_index and period_index must have the same length.")
        elif asset_index is not None:
            if len(np.asarray(asset_index)) != len(period_index):
                raise ValueError("asset_index and period_index must have the same length.")

        # batch_periods
        if not isinstance(batch_periods, numbers.Integral):
            raise TypeError("batch_periods must be an integer.")
        if batch_periods < 1:
            raise ValueError("batch_periods must be at least 1.")

        # batch_assets
        if mode in ("asset", "block"):
            if batch_assets is None:
                raise ValueError(
                    "batch_assets must be provided when mode is 'asset' or 'block'."
                )
        if batch_assets is not None:
            if not isinstance(batch_assets, numbers.Integral):
                raise TypeError("batch_assets must be an integer.")
            if batch_assets < 1:
                raise ValueError("batch_assets must be at least 1.")

        # asset_fraction and draws_per_block
        if mode == "mc":
            if (batch_assets is None) == (asset_fraction is None):
                raise ValueError(
                    "mode 'mc' needs exactly one of batch_assets or asset_fraction, to "
                    "set how many assets each draw contains."
                )
        if asset_fraction is not None:
            if not isinstance(asset_fraction, numbers.Real):
                raise TypeError("asset_fraction must be a real number.")
            if not (0 < asset_fraction <= 1):
                raise ValueError("asset_fraction must be in (0, 1].")
        if draws_per_block is not None:
            if not isinstance(draws_per_block, numbers.Integral):
                raise TypeError("draws_per_block must be an integer.")
            if draws_per_block < 1:
                raise ValueError("draws_per_block must be at least 1.")

        # shuffle
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean.")

        # aggregate_last
        if not isinstance(aggregate_last, bool):
            raise TypeError("aggregate_last must be a boolean.")

        # drop_last
        if not isinstance(drop_last, bool):
            raise TypeError("drop_last must be a boolean.")
        if aggregate_last and drop_last:
            raise ValueError("aggregate_last and drop_last cannot both be True.")

        # seed
        if seed is not None:
            if not isinstance(seed, numbers.Integral):
                raise TypeError("seed must be an integer or None.")
            if seed < 0:
                raise ValueError("seed must be non-negative.")
