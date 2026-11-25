"""
Synthetic test of the portfolio losses in `macrosynergy.learning.forecasting.torch.losses`:
in-sample against out-of-sample loss, epoch by epoch.

The library's `MLPRegressor`, with `LongShortModule` as the book layer, is trained once on
a fixed window of T simulated months where the truth is known. At every epoch the script
records the loss on that window (in sample) and on further months simulated from the same
process, which the network never trains on (out of sample). The question is whether a
loss can reach a good in-sample value without the signal, in which case its in-sample
value says nothing about out-of-sample skill.

Usage
-----
Run from this folder. `python synthetic_loss_gap.py --help` lists every option.

    python synthetic_loss_gap.py --run smoke --epochs 50 --data-seeds 1 --n-fresh 5
    python synthetic_loss_gap.py --run real --n-jobs 8 --table
    python synthetic_loss_gap.py --run strong_signal --sigma 0.04 --ic 0.3 --n-jobs 8
    python synthetic_loss_gap.py --run real --plot-only --table

Without `--loss`, all seven losses are run. The write-up's main run used
`--loss NegSharpeRatio PortfolioVariance NegMeanPortfolioReturn MultiOutputMCR`.

A fit took about 35-80 seconds on one CPU worker; 320 fits (four losses) took about
35 minutes with `--n-jobs 8`. More workers ran out of memory with 467 names.

Requires torch, scikit-learn, numpy, pandas, matplotlib and pyarrow. When this script sits
two folders below the root of a macrosynergy source checkout, that source is imported
rather than an installed copy (see `use_local_macrosynergy`).

Outputs
-------
Next to this script, in logs/synthetic_loss_gap/<run>/:

    curves.parquet     in-sample and out-of-sample loss, per fit and epoch
    fits.csv           per fit: time, error, first non-finite epoch, source if copied
    references.csv     oracle and zero-skill loss, per loss, cell and data seed
    run_config.json    every setting, the script's hash and the package versions

and in figs/synthetic_loss_gap/<run>/:

    <loss>.png         loss by epoch, one panel per cell
    summary.png        final in-sample against final out-of-sample loss

A run folder is never overwritten. Each fit is also saved as it finishes, in
logs/synthetic_loss_gap/<run>.partial/, so a run that stops (for example out of memory)
resumes where it stopped when the same command is run again. A fit is fully determined by
its settings and seeds, so `--reuse RUN` copies the identical fits of earlier or stopped
runs instead of running them again; fits.csv names the source of each copied fit.

Notes
-----
**The four cells.** K = 55 features and T = 120 training months by default. Only the
number of names M and the link between features and returns change:

    cell                   M     features and returns       conditions
    random, data rich       30   independent                M < K, T > M
    random, data poor      467   independent                M > K, T < M
    known DGP, data rich    30   the oracle's IC is --ic    M < K, T > M
    known DGP, data poor   467   the oracle's IC is --ic    M > K, T < M

With T < M, one fixed book can earn the same return every month (T equations in M
unknowns), so a loss that sees only the book's return series can reach its best value
without using the features.

**Data.** One formula; the random cells have s = 0:

    x_t      ~ N(0, I_K)                    month-level features, the same for every name
    beta_i   ~ N(0, I_K / K)                name i's loadings, drawn once per dataset
    r_t,i    = s * beta_i . x_t + sigma * e_t,i,    e_t,i ~ N(0, 1), sigma = --sigma
    s        = sigma * ic / sqrt(1 - ic^2)  so that the oracle's cross-sectional IC is ic

x_t is known before the month whose returns are r_t. The same data seed gives the same x
and e in a random cell and in its known-DGP twin, so the two differ only by the signal.

**Two arms.** `real`: the network sees the features. `no_features`: every input is 0, so
the network gives the same output every month and can only hold one fixed book. This arm
shows how far a loss gets with no information at all.

**Training months and fresh months.** Each dataset is one simulation of
(1 + `--n-fresh`) x T months: 2,520 with the defaults. The network trains on the first T
months, all of them at every step, and is trained once: there is no walk-forward. The
other months, the fresh months, are never trained on; they are used only to score the
network out of sample.

- The fresh months share the training months' loadings beta, so the true link between
  features and returns is the same. The features and the noise are new draws.
- The features have no persistence and the process never changes, so later months are
  statistically the same as earlier ones: an idealised out-of-sample test.
- The out-of-sample loss is the mean loss over the `--n-fresh` windows of T fresh months
  (months 121-240, 241-360, ... with the defaults): the expected loss on a new window as
  long as the training window, like for like with the in-sample loss, and precise.
- The library computes it during training: after every epoch it runs the current network
  on the fresh months (the same forward pass as `predict`) and scores the books (see
  `TrainThenFresh`). The script reports the value at the last epoch, the network after
  full training.

**References**, per loss, cell and data seed, always with a book of gross 1 whatever
`--eps`:

    oracle       `LongShortModule()` applied to the true expected returns, scored in
                 sample and on the fresh windows (known-DGP cells only). It is the
                 true-signal book, not each loss's own optimum.
    zero skill   the mean out-of-sample loss of 100 random fixed books.

**A path that turns non-finite** is cut at that epoch (NaN from there on), and the epoch
is recorded in fits.csv. A fit that raises is recorded with its error; the run goes on.

Differences from the standard use of MLPRegressor
-------------------------------------------------
Every setting is in `MLP_KWARGS` and `build_estimator`. Those that differ from the
`MLPRegressor` defaults:

    setting             library default                   here
    loss_func           nn.MSELoss()                      one package loss, default arguments
    signal_modifier     None                              LongShortModule(eps=--eps),
                                                          zero_sum False: gross 1, net free
    n_latent            32                                8
    encoder_activation  "relu"                            "silu"
    batch_size          32                                T: one full batch per epoch
    learning_rate       3e-4                              1e-3
    weight_decay        1e-4                              0
    epochs              10,000                            2,000
    patience            1,000                             epochs + 1: never stops early
    train_splitter      0.7: the last 30% of the dates    TrainThenFresh(T): the fresh
                        validate                          months validate
    x_scaler, y_scaler  StandardScaler(with_mean=False)   None
    random_state        42                                8,000,000 + network seed

Everything else is at the library default: one hidden layer, intercepts in both layers,
identity head activation, no dropout, no normalisation, AdamW, no scheduler, no turnover
penalty, the time-series sampler, no refit.

What this means:

- **No regularisation.** No weight decay, dropout, mini-batches, learning-rate schedule
  or early stopping. The test measures what the loss alone rewards; how much these
  controls would reduce the in-sample inflation is not tested.
- **The whole training window in every step.** With batch_size = T, every gradient step
  and every recorded in-sample loss use all T months, the series the portfolio losses
  are defined on.
- **No scalers.** The losses see returns in decimals; the features are already N(0, 1).
- **Early stopping only records the losses.** `MLPRegressor` records the training and
  validation loss at every epoch only when `patience` is set. Patience above the number
  of epochs means training never stops early.
- **One network per fit.** `MLPRegressor` can train several networks in one fit (a list
  of optimizers, a list of random_state values, a splitter with several splits) and
  averages their books in `predict`. Here there is one of each, so one network. The 10
  fits behind each line of the figures are 10 separate `MLPRegressor` fits (5 datasets x
  2 network seeds), not an ensemble.
- **The restored network is not used.** During training, `MLPRegressor` keeps a copy of
  the network's weights whenever the validation loss is the best so far, and at the end
  it restores the weights of that epoch. Here that epoch is picked on the fresh months, so
  `predict` would score the network on the months used to pick it: an optimistic
  out-of-sample loss. The script never calls `predict`; it reads the two loss paths and
  reports their last epoch.
- **The input layout.** `MLPRegressor` needs X and y indexed by (cid, real_date). Here
  there is one row per month under the single label "SYN", with consecutive days as
  dates (labels only; see `model_frame`).

Call stack of one fit
---------------------
To step through training, run one short fit in a debugger, with a breakpoint in
`MLPRegressor.train_model` or `MLPRegressor._fit_one_batch`:

    import synthetic_loss_gap as S
    cell = S.cells()[0]                      # random, data rich
    job = S.Job("NegSharpeRatio", cell, S.DATA_SEED0, "real", S.INIT_SEED0, epochs=50)
    res = S.run_job(job)                     # res["insample"], res["oos"]: the paths

The calls, from this script into the library. Library paths are relative to
macrosynergy/learning/forecasting/:

    main                                    this script
      run_jobs                              parallel over fits; saves each finished fit
        run_job                             one fit
          make_dataset                      T training months, then the fresh months
          build_estimator                   MLPRegressor(..., TrainThenFresh(T), ...)
          MLPRegressor.fit                  nn/mlp.py
            create_train_valid_splits_      calls TrainThenFresh.split: the first T rows
                                            train, the rest validate
            scale_data_                     no scalers: the arrays are used as they are
            make_tensor_datasets_
            for each optimizer and random_state (one of each here):
              initialize_model              MultiLayerPerceptron, torch/models/mlps.py:
                                            Linear(K, 8) -> SiLU -> Dropout(0)
                                            -> Linear(8, M) -> Identity -> LongShortModule
              make_optimizer                torch.optim.AdamW
              make_dataloaders_             TimeSeriesSampler, torch/samplers/: blocks of
                                            batch_size consecutive months
              train_model                   the epoch loop
                _fit_one_batch              forward pass, loss_func(book, returns),
                                            backward pass, optimizer step
                _eval_loss                  on the training months -> train_loss_path
                _eval_loss                  on the fresh months -> valid_loss_path
                update_es_stats             copies the weights if the validation loss is
                                            the best so far
              model.load_state_dict         after the last epoch: restores those weights
          est.early_stopping_dynamics       run_job reads the two paths from here

`LongShortModule` is in torch/modules/portfolio_layers/long_short_layer.py. The losses are
in torch/losses/: portfolio_losses.py, sharpe_loss.py and mcr_loss.py.

Where to change things
----------------------
- **The loss:** `build_loss`. Any `torch.nn.Module` whose `forward(y_pred, y_true)`
  returns a scalar works. `references` scores the oracle and the random books with the
  same function.
- **The network and the optimiser:** `MLP_KWARGS` (n_latent, dropout_p, weight_decay,
  learning_rate, optimizer, scheduler, reg_turnover).
- **The batch size:** `build_estimator`. The library cuts both the training and the fresh
  months into blocks of batch_size for the recorded losses. With mini-batches, both paths
  become means over blocks of batch_size months, no longer the loss over a window of T
  months.
- **The book layer:** `build_estimator` for the network and `references` for the oracle
  and the random books. Change both together.
- **The data:** `make_dataset`.
- **What is recorded:** `run_job`. Besides the loss paths, `MLPRegressor` keeps gradient
  and input-sensitivity diagnostics in `early_stopping_inference`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import BaseCrossValidator

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parents[1]


def use_local_macrosynergy() -> None:
    """
    Import macrosynergy from the source checkout this script sits in, if there is one.

    If this script sits two folders below the root of a macrosynergy source checkout, that
    source is put first on `sys.path`, so it is imported rather than an installed copy,
    which may be older. Otherwise the installed package is used.

    Raises
    ------
    ImportError
        If macrosynergy was already imported from somewhere else. Run this script on its
        own, before anything imports macrosynergy.
    """
    if not (SOURCE_ROOT / "macrosynergy" / "__init__.py").is_file():
        return
    if sys.path[:1] != [str(SOURCE_ROOT)]:
        if str(SOURCE_ROOT) in sys.path:
            sys.path.remove(str(SOURCE_ROOT))
        sys.path.insert(0, str(SOURCE_ROOT))
    import macrosynergy

    if (
        Path(macrosynergy.__file__).resolve()
        != (SOURCE_ROOT / "macrosynergy" / "__init__.py").resolve()
    ):
        raise ImportError(
            f"macrosynergy was imported from {macrosynergy.__file__}, not from the source at "
            f"{SOURCE_ROOT}; run this script on its own, before anything imports macrosynergy"
        )


use_local_macrosynergy()

from macrosynergy.learning import MLPRegressor  # noqa: E402
from macrosynergy.learning.forecasting.torch import LongShortModule  # noqa: E402
from macrosynergy.learning.forecasting.torch import (  # noqa: E402
    losses as package_losses,
)
from macrosynergy.learning.forecasting.torch.losses import (  # noqa: E402
    portfolio_losses,
)

# Every loss of the library's learning.forecasting.torch.losses, and only those: the six
# exported ones and NegMeanVarianceSkewnessUtility, defined in portfolio_losses.py but not
# exported.
LOSSES = (
    "NegSharpeRatio",
    "MultiOutputSharpe",
    "PortfolioVariance",
    "NegMeanVarianceSkewnessUtility",
    "NegMeanPortfolioReturn",
    "NegMeanVarianceUtility",
    "MultiOutputMCR",
)
SYMLOG_ABOVE = 10.0  # an axis turns symlog (linear within +-1) once |loss| exceeds this

K = 55  # features
T = 120  # training months
NAMES = (30, 467)  # data rich, data poor
IC = 0.1  # the oracle's cross-sectional IC in the known-DGP cells
SIGMA = 0.08  # monthly volatility of each name's noise
EPS = 0.0  # LongShortModule's eps (the package default)
# Fresh windows of T months for the out-of-sample loss. It is evaluated every epoch and
# dominates the time: 467 names, 2,000 epochs ~ 50 s.
N_FRESH = 20
N_RANDOM = 100  # random fixed books in the zero-skill reference
EPOCHS = 2000
DATA_SEED0, INIT_SEED0 = 7_000_000, 8_000_000
ARMS = ("real", "no_features")
LOGS = HERE / "logs" / "synthetic_loss_gap"
FIGS = HERE / "figs" / "synthetic_loss_gap"

# The network and its training: a small multi-head network, full batch, no scalers, no
# weight decay. The differences from the MLPRegressor defaults are listed in the module
# docstring. batch_size, epochs, patience, train_splitter, random_state, loss_func and
# signal_modifier are set in build_estimator.
MLP_KWARGS = dict(
    n_latent=8,
    fit_encoder_intercept=True,
    fit_head_intercept=True,
    encoder_activation="silu",
    head_activation="identity",
    head_rank=None,
    dropout_p=0,
    normalization=None,
    optimizer="AdamW",
    scheduler=None,
    learning_rate=1e-3,
    weight_decay=0.0,
    reg_turnover=0,
    use_ts_sampler=True,
    aggregate_last=True,
    drop_last=False,
    x_scaler=None,
    y_scaler=None,
    refit=False,
    verbose=False,
    inverse_transform_preds=False,
    min_samples=36,
)


@dataclass(frozen=True)
class Cell:
    """
    One cell of the design: a kind of data and a number of names.

    Parameters
    ----------
    data : str
        "random" (features and returns independent) or "known DGP".
    regime : str
        "data rich" (more months than names) or "data poor" (more names than months).
    m : int
        Number of names M, which is also the number of network outputs.
    ic : float
        The oracle's cross-sectional IC; 0 in the random cells.
    """

    data: str
    regime: str
    m: int
    ic: float

    @property
    def label(self) -> str:
        """Name of the cell in the outputs, for example "random, data rich"."""
        return f"{self.data}, {self.regime}"


REGIMES = ("data rich", "data poor")


def cells(
    ic: float = IC, names: Sequence[int] = NAMES, regimes: Sequence[str] = REGIMES
) -> List[Cell]:
    """
    The cells of a run: random and known DGP, for each regime asked for.

    Parameters
    ----------
    ic : float
        The oracle's IC in the known-DGP cells.
    names : Sequence[int]
        Number of names in the data-rich and in the data-poor cells.
    regimes : Sequence[str]
        Regimes to keep, among "data rich" and "data poor".

    Returns
    -------
    List[Cell]
        The cells, random before known DGP.
    """
    rich, poor = names
    grid = [
        Cell("random", "data rich", rich, 0.0),
        Cell("random", "data poor", poor, 0.0),
        Cell("known DGP", "data rich", rich, ic),
        Cell("known DGP", "data poor", poor, ic),
    ]
    return [c for c in grid if c.regime in regimes]


# --------------------------------------------------------------------------- data


@dataclass
class Dataset:
    """
    One synthetic dataset: the training window and the fresh windows after it.

    Parameters
    ----------
    x_train : np.ndarray
        Features of the training months, shape (T, K).
    mu_train : np.ndarray
        True expected returns of the training months, shape (T, M).
    r_train : np.ndarray
        Realised returns of the training months, shape (T, M).
    x_fresh : np.ndarray
        Features of the fresh months, shape (n_fresh * T, K).
    mu_fresh : np.ndarray
        True expected returns of the fresh months, shape (n_fresh * T, M).
    r_fresh : np.ndarray
        Realised returns of the fresh months, shape (n_fresh * T, M).
    """

    x_train: np.ndarray
    mu_train: np.ndarray
    r_train: np.ndarray
    x_fresh: np.ndarray
    mu_fresh: np.ndarray
    r_fresh: np.ndarray


def signal_scale(ic: float, sigma: float = SIGMA) -> float:
    """
    Signal scale s that gives the oracle a cross-sectional IC of `ic`.

    Parameters
    ----------
    ic : float
        Target IC, s / sqrt(s^2 + sigma^2).
    sigma : float
        Monthly volatility of each name's noise.

    Returns
    -------
    float
        s = sigma * ic / sqrt(1 - ic^2).
    """
    return sigma * ic / np.sqrt(1.0 - ic**2)


def make_dataset(
    m: int,
    ic: float,
    seed: int,
    k: int = K,
    t: int = T,
    n_fresh: int = N_FRESH,
    sigma: float = SIGMA,
) -> Dataset:
    """
    Draw a training window and the fresh windows after it, from one process.

    Parameters
    ----------
    m : int
        Number of names.
    ic : float
        The oracle's IC; 0 gives returns independent of the features.
    seed : int
        Data seed. The draws do not depend on `ic`, so the same seed gives the same x and
        e at every IC: a random cell and its known-DGP twin differ only by the signal.
    k : int
        Number of features.
    t : int
        Training months, which is also the length of each fresh window.
    n_fresh : int
        Number of fresh windows.
    sigma : float
        Monthly volatility of each name's noise.

    Returns
    -------
    Dataset
        One simulation of (1 + n_fresh) * t months, cut in two: the first t months
        train, the other n_fresh * t months are the fresh ones, never trained on. The
        loadings beta are drawn once and shared by all months; the features and the noise
        are new draws every month.
    """
    rng = np.random.default_rng(seed)
    beta = rng.normal(0.0, 1.0 / np.sqrt(k), size=(m, k))
    n = (1 + n_fresh) * t
    x = rng.normal(size=(n, k))
    e = rng.normal(size=(n, m))
    mu = signal_scale(ic, sigma) * x @ beta.T
    r = mu + sigma * e
    return Dataset(x[:t], mu[:t], r[:t], x[t:], mu[t:], r[t:])


def mean_ic(mu: np.ndarray, r: np.ndarray) -> float:
    """
    Mean over months of the cross-sectional correlation of expected and realised returns.

    Parameters
    ----------
    mu : np.ndarray
        Expected returns, shape (months, names).
    r : np.ndarray
        Realised returns, same shape.

    Returns
    -------
    float
        The oracle's realised IC.
    """
    a, b = mu - mu.mean(axis=1, keepdims=True), r - r.mean(axis=1, keepdims=True)
    return float(
        np.mean((a * b).sum(axis=1) / np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1)))
    )


def model_frame(a: np.ndarray, prefix: str) -> pd.DataFrame:
    """
    Wrap an array in the layout `MLPRegressor` checks: a (cid, real_date) multi-index.

    Parameters
    ----------
    a : np.ndarray
        One row per month.
    prefix : str
        Column names are the prefix followed by the column number.

    Returns
    -------
    pd.DataFrame
        One row per month under the single cid "SYN". The dates are consecutive days,
        used as labels only: a run can have up to 6,120 months, and `MLPRegressor` does
        not use the date frequency with `TrainThenFresh`.
    """
    idx = pd.MultiIndex.from_arrays(
        [
            np.array(["SYN"] * len(a), dtype=object),
            pd.date_range("2000-01-01", periods=len(a), freq="D"),
        ],
        names=["cid", "real_date"],
    )
    return pd.DataFrame(
        a, index=idx, columns=[f"{prefix}{j}" for j in range(a.shape[1])]
    )


class TrainThenFresh(BaseCrossValidator):
    """
    One split: the first `n_train` rows train, the rest (the fresh windows) validate.

    Parameters
    ----------
    n_train : int
        Number of training rows, T.

    Notes
    -----
    This splitter is how the out-of-sample loss is recorded at every epoch by the
    library's own training loop, without custom training code.

    - `MLPRegressor` records the training and the validation loss at every epoch only on
      its early-stopping path (`patience` set), and it computes the validation loss on
      the rows its `train_splitter` marks as validation.
    - The default splitter (0.7) would validate on the last 30% of the dates: the training
      window would shrink from 120 to 84 months, and the out-of-sample loss would be
      measured on only 36 months.
    - With this splitter, fresh months drawn from the same process (the same loadings) are
      stacked after the T training months, and the library scores them every epoch.
      `run_job` stacks them in that order; the split works by row position.
    - The library cuts the validation rows into blocks of `batch_size` months and averages
      the loss over the blocks. With batch_size = T that is the mean loss over fresh
      windows as long as the training window, like for like with the in-sample loss.
    - Patience is set above the number of epochs, so the validation part never stops
      training. It does decide which epoch's weights `MLPRegressor` restores at the end
      (the epoch with the best validation loss). That epoch is picked on the fresh months,
      which is why the restored network is not used.
    """

    def __init__(self, n_train: int):
        self.n_train = int(n_train)

    def split(self, X, y=None, groups=None):
        """
        Yield the single split.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            All rows: the training months first, then the fresh months.
        y : ignored
        groups : ignored

        Yields
        ------
        Tuple[np.ndarray, np.ndarray]
            Positions of the training rows and of the validation rows.
        """
        yield np.arange(self.n_train), np.arange(self.n_train, len(X))

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of splits: always 1."""
        return 1


# --------------------------------------------------------------------------- losses and books


def build_loss(name: str):
    """
    A loss of the package, with its default arguments.

    Parameters
    ----------
    name : str
        One of `LOSSES`.

    Returns
    -------
    torch.nn.Module
        The loss; `forward(y_pred, y_true)` returns a scalar.

    Raises
    ------
    ValueError
        If `name` is not a loss of `macrosynergy.learning.forecasting.torch.losses`.
    """
    if name not in LOSSES:
        raise ValueError(
            f"{name!r} is not a loss of the package's learning.forecasting.torch.losses"
        )
    for module in (package_losses, portfolio_losses):
        if hasattr(module, name):
            return getattr(module, name)()
    raise ValueError(f"{name!r} not found in the package")


def window_mean_loss(loss, w: torch.Tensor, r: torch.Tensor, t: int) -> float:
    """
    Mean of a loss over consecutive windows of `t` months.

    This is how `MLPRegressor`'s validation loader computes the out-of-sample loss when
    batch_size is `t`; the references are scored the same way.

    Parameters
    ----------
    loss : torch.nn.Module
        The loss.
    w : torch.Tensor
        Book weights: one row (a fixed book, used every month) or one row per month.
    r : torch.Tensor
        Returns, one row per month.
    t : int
        Window length in months.

    Returns
    -------
    float
        The mean loss over the windows.
    """
    n = r.shape[0]
    w = w.expand(n, -1) if w.shape[0] == 1 else w
    with torch.no_grad():
        return float(
            np.mean([float(loss(w[i : i + t], r[i : i + t])) for i in range(0, n, t)])
        )


def references(
    loss_name: str, data: Dataset, cell: Cell, t: int, seed: int
) -> Dict[str, float]:
    """
    Reference levels of a loss on one dataset: the oracle and zero skill.

    Both use `LongShortModule()` with the package defaults (gross 1), whatever `--eps`.

    Parameters
    ----------
    loss_name : str
        One of `LOSSES`.
    data : Dataset
        The dataset.
    cell : Cell
        Its cell; the oracle exists only in the known-DGP cells.
    t : int
        Window length in months.
    seed : int
        Seed of the random books.

    Returns
    -------
    Dict[str, float]
        "oracle_insample" and "oracle_oos": the loss of the book made from the true
        expected returns, on the training window and averaged over the fresh windows (NaN
        in the random cells). "zero_skill_oos": the mean out-of-sample loss of `N_RANDOM`
        random fixed books.
    """
    loss, layer = build_loss(loss_name), LongShortModule()
    # float32, MLPRegressor's dtype
    f32 = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    r_tr, r_fr = f32(data.r_train), f32(data.r_fresh)
    out = {"oracle_insample": np.nan, "oracle_oos": np.nan}
    if cell.ic > 0:
        with torch.no_grad():
            out["oracle_insample"] = float(loss(layer(f32(data.mu_train)), r_tr))
        out["oracle_oos"] = window_mean_loss(loss, layer(f32(data.mu_fresh)), r_fr, t)
    g = torch.Generator().manual_seed(seed)
    books = layer(torch.randn((N_RANDOM, cell.m), generator=g))
    out["zero_skill_oos"] = float(
        np.mean([window_mean_loss(loss, b[None, :], r_fr, t) for b in books])
    )
    return out


# --------------------------------------------------------------------------- one fit


@dataclass
class Job:
    """
    The settings of one fit. Together they determine the fit exactly.

    Parameters
    ----------
    loss : str
        One of `LOSSES`.
    cell : Cell
        The cell.
    data_seed : int
        Seed of the dataset.
    arm : str
        "real" (the features) or "no_features" (every input 0).
    init_seed : int
        Seed of the network: `MLPRegressor`'s random_state.
    epochs : int
        Training epochs.
    k : int
        Number of features.
    t : int
        Training months.
    n_fresh : int
        Number of fresh windows of t months.
    sigma : float
        Monthly volatility of each name's noise.
    eps : float
        `LongShortModule`'s eps.
    """

    loss: str
    cell: Cell
    data_seed: int
    arm: str
    init_seed: int
    epochs: int = EPOCHS
    k: int = K
    t: int = T
    n_fresh: int = N_FRESH
    sigma: float = SIGMA
    eps: float = EPS


def build_estimator(
    loss_name: str, t: int, epochs: int, seed: int, eps: float = EPS
) -> MLPRegressor:
    """
    The estimator of one fit: `MLPRegressor` with `MLP_KWARGS` and the settings below.

    Parameters
    ----------
    loss_name : str
        One of `LOSSES`.
    t : int
        Training months. It is also the batch size (full batch) and the length of the
        validation blocks.
    epochs : int
        Training epochs. Patience is set one above, so training never stops early but the
        loss paths are recorded.
    seed : int
        The network's random_state.
    eps : float
        `LongShortModule`'s eps.

    Returns
    -------
    MLPRegressor
        Not fitted.
    """
    return MLPRegressor(
        signal_modifier=LongShortModule(eps=eps),
        loss_func=build_loss(loss_name),
        batch_size=t,
        epochs=epochs,
        patience=epochs + 1,
        train_splitter=TrainThenFresh(t),
        random_state=seed,
        **MLP_KWARGS,
    )


def book_layer_label(eps: float) -> str:
    """
    How run_config.json names the book layer.

    Parameters
    ----------
    eps : float
        `LongShortModule`'s eps.

    Returns
    -------
    str
        The label; unchanged from earlier runs for the package default.
    """
    return (
        "LongShortModule() (zero_sum False, eps 0)"
        if eps == 0
        else f"LongShortModule(eps={eps:g}) (zero_sum False)"
    )


def cut_nonfinite(path: Sequence[float]) -> Tuple[np.ndarray, int]:
    """
    Cut a loss path at its first non-finite value.

    Parameters
    ----------
    path : Sequence[float]
        Loss per epoch.

    Returns
    -------
    Tuple[np.ndarray, int]
        The path with every value from the first non-finite one on set to NaN, and that
        epoch (1-based; 0 if the path stays finite).
    """
    a = np.asarray(path, dtype=float).copy()
    bad = ~np.isfinite(a)
    if not bad.any():
        return a, 0
    i = int(np.argmax(bad))
    a[i:] = np.nan
    return a, i + 1


def run_job(job: Job) -> Dict[str, Any]:
    """
    Train one network and return its in-sample and out-of-sample loss paths.

    The entry point into the library: see "Call stack of one fit" in the module
    docstring.

    Parameters
    ----------
    job : Job
        The settings of the fit.

    Returns
    -------
    Dict[str, Any]
        "insample" and "oos": loss per epoch, cut at the first non-finite value;
        "first_nonfinite_insample" and "first_nonfinite_oos": that epoch (0 if none);
        "error": the exception if the fit raised, else ""; "seconds"; "job"; "source"
        (the run a copied fit came from, else "").
    """
    torch.set_num_threads(1)
    data = make_dataset(
        job.cell.m, job.cell.ic, job.data_seed, job.k, job.t, job.n_fresh, job.sigma
    )
    # The training months first, then the fresh ones: TrainThenFresh splits by position.
    x = np.vstack([data.x_train, data.x_fresh])
    if job.arm == "no_features":
        x = np.zeros_like(x)
    y = np.vstack([data.r_train, data.r_fresh])
    est = build_estimator(job.loss, job.t, job.epochs, job.init_seed, job.eps)
    start, error = time.perf_counter(), ""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(model_frame(x, "F"), model_frame(y, "N"))
        # One optimizer and one random_state: key (0, 0); one split: element [0].
        dyn = est.early_stopping_dynamics[(0, 0)]
        ins, oos = dyn["train_loss_path"][0], dyn["valid_loss_path"][0]
    except Exception as exc:
        # Recorded, not raised: one failed fit must not stop the grid.
        error, ins, oos = (
            f"{type(exc).__name__}: {exc}",
            [np.nan] * job.epochs,
            [np.nan] * job.epochs,
        )
    ins, bad_in = cut_nonfinite(ins)
    oos, bad_oos = cut_nonfinite(oos)
    return {
        "job": job,
        "insample": ins,
        "oos": oos,
        "first_nonfinite_insample": bad_in,
        "first_nonfinite_oos": bad_oos,
        "error": error,
        "seconds": time.perf_counter() - start,
        "source": "",
    }


def job_key(job: Job) -> str:
    """
    A stable name for a fit, used to resume runs and to copy identical fits.

    Parameters
    ----------
    job : Job
        The settings of the fit.

    Returns
    -------
    str
        16 hex characters; the same settings give the same key, whatever the number types.
    """
    c = job.cell
    fields = (
        job.loss,
        c.data,
        c.regime,
        int(c.m),
        float(c.ic),
        int(job.data_seed),
        job.arm,
        int(job.init_seed),
        int(job.epochs),
        int(job.k),
        int(job.t),
        int(job.n_fresh),
        float(job.sigma),
        float(job.eps),
    )
    return hashlib.sha1(repr(fields).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- the grid


def make_jobs(
    losses: Sequence[str],
    grid: Sequence[Cell],
    data_seeds: int,
    init_seeds: int,
    epochs: int,
    k: int,
    t: int,
    n_fresh: int,
    sigma: float = SIGMA,
    eps: float = EPS,
) -> List[Job]:
    """
    Every fit of a run: losses x cells x data seeds x arms x network seeds.

    Parameters
    ----------
    losses : Sequence[str]
        Losses to run.
    grid : Sequence[Cell]
        Cells to run.
    data_seeds : int
        Datasets per cell, seeded DATA_SEED0, DATA_SEED0 + 1, ...
    init_seeds : int
        Networks per dataset and arm, seeded INIT_SEED0, INIT_SEED0 + 1, ...
    epochs, k, t, n_fresh, sigma, eps
        As in `Job`.

    Returns
    -------
    List[Job]
        The fits, in a fixed order.
    """
    return [
        Job(
            loss,
            cell,
            DATA_SEED0 + i,
            arm,
            INIT_SEED0 + j,
            epochs,
            k,
            t,
            n_fresh,
            sigma,
            eps,
        )
        for loss in losses
        for cell in grid
        for i in range(data_seeds)
        for arm in ARMS
        for j in range(init_seeds)
    ]


def run_jobs(
    jobs: Sequence[Job], n_jobs: int = 1, store: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Run every fit, in parallel if asked, saving each one as it finishes.

    Parameters
    ----------
    jobs : Sequence[Job]
        The fits.
    n_jobs : int
        Fits run in parallel, one process each.
    store : Optional[Path]
        Folder where each finished fit is saved at once. Fits already saved there are not
        run again, so a run that stopped resumes where it stopped.

    Returns
    -------
    List[Dict[str, Any]]
        `run_job`'s result for every fit, in the order of `jobs`.

    Raises
    ------
    SystemExit
        If `store` holds fits of a different grid, or if some fits are missing at the end
        (for example a MemoryError while a worker returned one). Running the same command
        again runs only the missing fits.
    """
    index = {job_key(j): i for i, j in enumerate(jobs)}
    done: Dict[int, Dict[str, Any]] = {}
    if store is not None:
        store.mkdir(parents=True, exist_ok=True)
        for f in sorted(store.glob("fit_*.pkl")):
            res = pickle.loads(f.read_bytes())
            key = job_key(res["job"])
            if key not in index:
                raise SystemExit(
                    f"{store} holds fits of a different grid; remove it or change --run"
                )
            done[index[key]] = res
        if done:
            print(f"resuming: {len(done)} of {len(jobs)} fits already done", flush=True)

    def keep(i: int, res: Dict[str, Any]) -> None:
        done[i] = res
        if store is not None:
            (store / f"fit_{job_key(jobs[i])}.pkl").write_bytes(pickle.dumps(res))
        j = res["job"]
        print(
            f"fit {len(done)}/{len(jobs)}: {j.loss} | {j.cell.label} | {j.arm} | data {j.data_seed} "
            f"init {j.init_seed} ({res['seconds']:.1f}s){' ERROR ' + res['error'] if res['error'] else ''}",
            flush=True,
        )

    todo = [i for i in range(len(jobs)) if i not in done]
    if n_jobs <= 1:
        for i in todo:
            keep(i, run_job(jobs[i]))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(run_job, jobs[i]): i for i in todo}
            for fut in as_completed(futures):
                try:
                    keep(futures[fut], fut.result())
                except Exception as exc:  # the fit is lost, not the run
                    print(
                        f"fit of job {futures[fut]} lost: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
    missing = len(jobs) - len(done)
    if missing:
        raise SystemExit(
            f"{missing} of {len(jobs)} fits missing; run the same command again to resume"
        )
    return [done[i] for i in range(len(jobs))]


def stored_results(name: str) -> List[Dict[str, Any]]:
    """
    Every finished fit of an earlier run or of a stopped one.

    Parameters
    ----------
    name : str
        Run name: its run folder, or its .partial folder if it stopped.

    Returns
    -------
    List[Dict[str, Any]]
        In `run_job`'s form, with "source" set to the run. Failed fits are included, with
        their error.
    """
    run, partial = LOGS / name, LOGS / f"{name}.partial"
    if not run.is_dir() and not partial.is_dir():
        raise SystemExit(f"no run or stopped run named {name!r} in {LOGS}")
    out = (
        [pickle.loads(f.read_bytes()) for f in sorted(partial.glob("fit_*.pkl"))]
        if partial.is_dir()
        else []
    )
    if (run / "run_config.json").is_file():
        cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
        keys = ["loss", "cell", "arm", "data_seed", "init_seed"]
        paths = {
            k: g.sort_values("epoch")
            for k, g in pd.read_parquet(run / "curves.parquet").groupby(keys)
        }
        for row in pd.read_csv(run / "fits.csv", keep_default_na=False).itertuples(
            index=False
        ):
            cell = Cell(row.data, row.regime, int(row.m), float(row.ic))
            job = Job(
                row.loss,
                cell,
                int(row.data_seed),
                row.arm,
                int(row.init_seed),
                int(cfg["epochs"]),
                int(cfg["k"]),
                int(cfg["t"]),
                int(cfg["n_fresh"]),
                float(cfg.get("sigma", SIGMA)),
                float(cfg.get("eps", EPS)),
            )
            g = paths[(row.loss, row.cell, row.arm, row.data_seed, row.init_seed)]
            out.append(
                {
                    "job": job,
                    "insample": g["insample"].to_numpy(),
                    "oos": g["oos"].to_numpy(),
                    "first_nonfinite_insample": int(row.first_nonfinite_insample),
                    "first_nonfinite_oos": int(row.first_nonfinite_oos),
                    "error": row.error,
                    "seconds": float(row.seconds),
                    "source": name,
                }
            )
    return out


def seed_store(
    store: Path, jobs: Sequence[Job], sources: Sequence[str]
) -> Dict[str, int]:
    """
    Copy into `store` every fit of the source runs whose settings match one of `jobs`.

    A fit is fully determined by its settings and seeds, so a copy equals a re-run. Failed
    fits are not copied: they run again.

    Parameters
    ----------
    store : Path
        The .partial folder of the run being made.
    jobs : Sequence[Job]
        Its fits.
    sources : Sequence[str]
        Names of earlier or stopped runs.

    Returns
    -------
    Dict[str, int]
        Number of fits copied from each source.
    """
    index = {job_key(j): i for i, j in enumerate(jobs)}
    store.mkdir(parents=True, exist_ok=True)
    have = {
        job_key(pickle.loads(f.read_bytes())["job"]) for f in store.glob("fit_*.pkl")
    }
    counts = {}
    for name in sources:
        n = 0
        for res in stored_results(name):
            key = job_key(res["job"])
            if key in index and key not in have and not res["error"]:
                res = {
                    **res,
                    "job": jobs[index[key]],
                    "source": res.get("source") or name,
                }
                (store / f"fit_{key}.pkl").write_bytes(pickle.dumps(res))
                have.add(key)
                n += 1
        counts[name] = n
    return counts


def _keys(job: Job) -> Dict[str, Any]:
    """The columns that identify a fit in the output tables."""
    return {
        "loss": job.loss,
        "cell": job.cell.label,
        "data": job.cell.data,
        "regime": job.cell.regime,
        "m": job.cell.m,
        "ic": job.cell.ic,
        "arm": job.arm,
        "data_seed": job.data_seed,
        "init_seed": job.init_seed,
    }


def curves_frame(results: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """
    The loss paths of every fit, as written to curves.parquet.

    Parameters
    ----------
    results : Sequence[Dict[str, Any]]
        `run_job`'s results.

    Returns
    -------
    pd.DataFrame
        One row per fit and epoch: the fit's keys, "epoch" (1-based), "insample", "oos".
    """
    parts = []
    for res in results:
        n = len(res["insample"])
        part = pd.DataFrame(
            {
                "epoch": np.arange(1, n + 1),
                "insample": res["insample"],
                "oos": res["oos"],
            }
        )
        for key, value in _keys(res["job"]).items():
            part[key] = value
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def fits_frame(results: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """
    One row per fit, as written to fits.csv.

    Parameters
    ----------
    results : Sequence[Dict[str, Any]]
        `run_job`'s results.

    Returns
    -------
    pd.DataFrame
        The fit's keys, time, error, first non-finite epochs and the run it was copied
        from ("" if it was run here).
    """
    return pd.DataFrame(
        [
            {
                **_keys(r["job"]),
                "seconds": r["seconds"],
                "error": r["error"],
                "first_nonfinite_insample": r["first_nonfinite_insample"],
                "first_nonfinite_oos": r["first_nonfinite_oos"],
                "reused_from": r.get("source", ""),
            }
            for r in results
        ]
    )


def references_frame(
    losses: Sequence[str],
    grid: Sequence[Cell],
    data_seeds: int,
    k: int,
    t: int,
    n_fresh: int,
    sigma: float = SIGMA,
) -> pd.DataFrame:
    """
    The references of every loss, cell and dataset, as written to references.csv.

    Parameters
    ----------
    losses : Sequence[str]
        Losses of the run.
    grid : Sequence[Cell]
        Cells of the run.
    data_seeds : int
        Datasets per cell.
    k, t, n_fresh, sigma
        As in `make_dataset`.

    Returns
    -------
    pd.DataFrame
        One row per loss, cell and data seed: `references`' values and the oracle's
        realised in-sample IC.
    """
    rows = []
    for cell in grid:
        for i in range(data_seeds):
            data = make_dataset(cell.m, cell.ic, DATA_SEED0 + i, k, t, n_fresh, sigma)
            for loss in losses:
                rows.append(
                    {
                        "loss": loss,
                        "cell": cell.label,
                        "data_seed": DATA_SEED0 + i,
                        "oracle_ic_insample": (
                            mean_ic(data.mu_train, data.r_train)
                            if cell.ic > 0
                            else np.nan
                        ),
                        **references(loss, data, cell, t, seed=DATA_SEED0 + i),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures


ARM_STYLE = {"real": ("C0", "REAL"), "no_features": ("C1", "NO-FEATURES")}
SIDE_STYLE = {"insample": ("-", "in sample"), "oos": ("--", "out of sample")}


def final_values(curves: pd.DataFrame) -> pd.DataFrame:
    """
    The final loss of every fit.

    Parameters
    ----------
    curves : pd.DataFrame
        As written to curves.parquet.

    Returns
    -------
    pd.DataFrame
        One row per fit: the in-sample and out-of-sample loss at the last epoch where both
        are finite.
    """
    ok = curves[np.isfinite(curves["insample"]) & np.isfinite(curves["oos"])]
    last = (
        ok.sort_values("epoch")
        .groupby(["loss", "cell", "arm", "data_seed", "init_seed"])
        .tail(1)
    )
    return last.reset_index(drop=True)


def _band(sub: pd.DataFrame, col: str) -> pd.DataFrame:
    """Median, minimum and maximum of one column over fits, per epoch."""
    wide = sub.pivot_table(
        index="epoch", columns=["data_seed", "init_seed"], values=col, dropna=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return pd.DataFrame(
            {
                "median": wide.median(axis=1),
                "lo": wide.min(axis=1),
                "hi": wide.max(axis=1),
            }
        )


def _set_scale(ax, values: np.ndarray, axis: str = "y") -> None:
    """Symlog once the losses run beyond +-SYMLOG_ABOVE (losses without a floor), else linear."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size and np.abs(v).max() > SYMLOG_ABOVE:
        (ax.set_yscale if axis == "y" else ax.set_xscale)("symlog", linthresh=1.0)
    elif axis == "x":
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(4))


def plot_loss(
    curves: pd.DataFrame,
    refs: pd.DataFrame,
    fits: pd.DataFrame,
    loss: str,
    grid: Sequence[Cell],
    t: int,
    k: int,
    path: Path,
    setting: str = "",
) -> None:
    """
    Draw one loss by epoch: one panel per cell, both arms, in and out of sample.

    Each line is the median over fits, with a band from the minimum to the maximum. The
    oracle (green) and zero skill (grey) are horizontal lines.

    Parameters
    ----------
    curves, refs, fits : pd.DataFrame
        As written to curves.parquet, references.csv and fits.csv.
    loss : str
        The loss to draw.
    grid : Sequence[Cell]
        Cells of the run; panels of other cells are hidden.
    t, k : int
        Training months and features, for the panel titles.
    path : Path
        Output file.
    setting : str
        Added to the figure title when not empty (see `run_setting`).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows: List[str] = ["random", "known DGP"]
    cols: List[str] = list(REGIMES)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey="col")
    for ax in axes.flat:
        ax.set_visible(False)  # only the cells of the run are shown
    n_runs = 0
    for cell in grid:
        axes[rows.index(cell.data), cols.index(cell.regime)].set_visible(True)
        ax = axes[rows.index(cell.data), cols.index(cell.regime)]
        sub = curves[(curves["loss"] == loss) & (curves["cell"] == cell.label)]
        for arm, (color, _) in ARM_STYLE.items():
            arm_sub = sub[sub["arm"] == arm]
            if arm_sub.empty:
                continue
            n_runs = arm_sub.groupby(["data_seed", "init_seed"]).ngroups
            for side, (style, _) in SIDE_STYLE.items():
                b = _band(arm_sub, side)
                ax.plot(
                    b.index, b["median"], color=color, linestyle=style, linewidth=1.4
                )
                ax.fill_between(
                    b.index, b["lo"], b["hi"], color=color, alpha=0.12, linewidth=0
                )
        ref = refs[(refs["loss"] == loss) & (refs["cell"] == cell.label)].mean(
            numeric_only=True
        )
        if np.isfinite(ref.get("oracle_insample", np.nan)):
            ax.axhline(ref["oracle_insample"], color="C2", linestyle="-", linewidth=0.9)
            ax.axhline(ref["oracle_oos"], color="C2", linestyle="--", linewidth=0.9)
        ax.axhline(ref["zero_skill_oos"], color="0.5", linestyle=":", linewidth=1.0)
        f = fits[(fits["loss"] == loss) & (fits["cell"] == cell.label)]
        failed = int((f["error"] != "").sum())
        nonfinite = int(
            ((f["first_nonfinite_insample"] > 0) & (f["error"] == "")).sum()
        )
        title = f"{cell.label}\nM = {cell.m}, T = {t}, K = {k}" + (
            f"; IC {cell.ic:g}" if cell.ic > 0 else ""
        )
        if failed:
            title += f"\n{failed} of {len(f)} fits failed with an error (no path)"
        if nonfinite:
            title += f"\n{nonfinite} of {len(f)} fits non-finite (paths cut there)"
        ax.set_title(title, fontsize=9.5)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
    loss_values = (
        curves.loc[curves["loss"] == loss, ["insample", "oos"]].to_numpy().ravel()
    )
    for ax in axes.flat:
        _set_scale(ax, loss_values, "y")
    for ax in axes.flat:
        ax.set_xlabel("epoch")
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_ylabel("loss (lower is better)")
    handles = [
        Line2D([], [], color=c, linestyle=s, label=f"{a} {sl}")
        for _, (c, a) in ARM_STYLE.items()
        for _, (s, sl) in SIDE_STYLE.items()
    ]
    handles += [
        Line2D([], [], color="C2", linestyle="-", label="oracle in sample"),
        Line2D([], [], color="C2", linestyle="--", label="oracle out of sample"),
        Line2D(
            [],
            [],
            color="0.5",
            linestyle=":",
            label="zero skill: random fixed books, out of sample",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle(
        f"{loss}: loss by epoch, median of {n_runs} fits per line, band min to max"
        + (" | " + setting if setting else ""),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_summary(
    curves: pd.DataFrame, losses: Sequence[str], grid: Sequence[Cell], path: Path
) -> None:
    """
    Draw the final in-sample loss against the final out-of-sample loss, one panel per loss.

    Each point is the median over fits of one cell and arm. Above the diagonal, a fit does
    worse out of sample than in sample.

    Parameters
    ----------
    curves : pd.DataFrame
        As written to curves.parquet.
    losses : Sequence[str]
        Losses to draw.
    grid : Sequence[Cell]
        Cells of the run.
    path : Path
        Output file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fin = final_values(curves)
    med = (
        fin.groupby(["loss", "cell", "arm"])[["insample", "oos"]].median().reset_index()
    )
    colors = {c.label: f"C{i}" for i, c in enumerate(grid)}
    markers = {"real": "o", "no_features": "s"}
    ncols = 4
    # A free panel holds the legend; otherwise it goes below the panels.
    spare_panel = len(losses) % ncols != 0
    nrows = int(np.ceil(len(losses) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.8 * nrows + (0 if spare_panel else 0.6)),
        squeeze=False,
    )
    for ax, loss in zip(axes.flat, losses):
        sub = med[med["loss"] == loss]
        for _, row in sub.iterrows():
            ax.scatter(
                row["insample"],
                row["oos"],
                color=colors[row["cell"]],
                marker=markers[row["arm"]],
                s=36,
                facecolors=(
                    "none" if row["arm"] == "no_features" else colors[row["cell"]]
                ),
            )
        vals = np.concatenate([sub["insample"].to_numpy(), sub["oos"].to_numpy()])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = vals.min(), vals.max()
            ax.plot([lo, hi], [lo, hi], color="0.6", linestyle="--", linewidth=0.8)
        _set_scale(ax, vals, "x")
        _set_scale(ax, vals, "y")
        ax.set_title(loss, fontsize=9.5)
        ax.set_xlabel("final in-sample loss", fontsize=8.5)
        ax.set_ylabel("final out-of-sample loss", fontsize=8.5)
        ax.grid(True, alpha=0.3)
    for ax in list(axes.flat)[len(losses) :]:
        ax.axis("off")
    handles = [
        Line2D([], [], color=c, marker="o", linestyle="", label=lab)
        for lab, c in colors.items()
    ]
    handles += [
        Line2D([], [], color="k", marker="o", linestyle="", label="REAL"),
        Line2D(
            [],
            [],
            color="k",
            marker="s",
            markerfacecolor="none",
            linestyle="",
            label="NO-FEATURES",
        ),
        Line2D([], [], color="0.6", linestyle="--", label="out of sample = in sample"),
    ]
    if spare_panel:
        list(axes.flat)[-1].legend(
            handles=handles, loc="center", fontsize=8.5, frameon=False
        )
    else:
        fig.legend(
            handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False
        )
    fig.suptitle(
        "Final loss in sample against out of sample, median over fits (above the diagonal: "
        "worse out of sample)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0 if spare_panel else 0.12, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def final_table(
    curves: pd.DataFrame,
    refs: pd.DataFrame,
    losses: Sequence[str],
    grid: Sequence[Cell],
) -> str:
    """
    The final values of a run as a Markdown table.

    Parameters
    ----------
    curves, refs : pd.DataFrame
        As written to curves.parquet and references.csv.
    losses : Sequence[str]
        Losses of the run.
    grid : Sequence[Cell]
        Cells of the run.

    Returns
    -------
    str
        Per loss and cell: the median final loss of each arm, in and out of sample, and
        the references (mean over data seeds).
    """
    med = (
        final_values(curves)
        .groupby(["loss", "cell", "arm"])[["insample", "oos"]]
        .median()
    )
    ref = refs.groupby(["loss", "cell"])[
        ["oracle_insample", "oracle_oos", "zero_skill_oos"]
    ].mean()
    g = lambda v: "nan" if not np.isfinite(v) else f"{v:.4g}"  # noqa: E731
    lines = [
        "| loss | cell | REAL in | REAL out | NO-FEATURES in | NO-FEATURES out | oracle in | oracle out "
        "| zero skill out |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for loss in losses:
        for cell in grid:
            vals = []
            for arm in ARMS:
                key = (loss, cell.label, arm)
                vals += (
                    [med.loc[key, "insample"], med.loc[key, "oos"]]
                    if key in med.index
                    else [np.nan, np.nan]
                )
            r = ref.loc[(loss, cell.label)]
            vals += [r["oracle_insample"], r["oracle_oos"], r["zero_skill_oos"]]
            lines.append(
                f"| `{loss}` | {cell.label} | " + " | ".join(g(v) for v in vals) + " |"
            )
    return "\n".join(lines)


def run_setting(cfg: Dict[str, Any]) -> str:
    """
    The noise level and the book layer's eps, for the figure titles.

    Parameters
    ----------
    cfg : Dict[str, Any]
        The run's run_config.json. Runs from before these options existed have the
        defaults.

    Returns
    -------
    str
        "" when both are at their defaults.
    """
    sigma, eps = cfg.get("sigma", SIGMA), cfg.get("eps", EPS)
    if sigma == SIGMA and eps == EPS:
        return ""
    return f"noise {sigma:.0%} a month, LongShortModule eps {eps:g}"


def make_figures(run: str) -> Path:
    """
    Draw every figure of a run from its logs.

    Parameters
    ----------
    run : str
        Run name.

    Returns
    -------
    Path
        The run's figure folder.
    """
    logs, figs = LOGS / run, FIGS / run
    cfg = json.loads((logs / "run_config.json").read_text(encoding="utf-8"))
    curves = pd.read_parquet(logs / "curves.parquet")
    refs, fits = pd.read_csv(logs / "references.csv"), pd.read_csv(
        logs / "fits.csv", keep_default_na=False
    )
    grid = cells(cfg["ic"], cfg["names"], cfg.get("regimes", REGIMES))
    for loss in cfg["losses"]:
        plot_loss(
            curves,
            refs,
            fits,
            loss,
            grid,
            cfg["t"],
            cfg["k"],
            figs / f"{loss}.png",
            run_setting(cfg),
        )
    plot_summary(curves, cfg["losses"], grid, figs / "summary.png")
    return figs


# --------------------------------------------------------------------------- main


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the grid of fits, write the logs and draw the figures.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Command-line arguments; `python synthetic_loss_gap.py --help` lists them.

    Returns
    -------
    int
        0 on success.
    """
    p = argparse.ArgumentParser(
        description="In-sample and out-of-sample loss of each package loss on synthetic data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run", required=True, help="run name: logs/synthetic_loss_gap/<run>/"
    )
    p.add_argument("--loss", nargs="+", default=list(LOSSES), help="losses to run")
    p.add_argument(
        "--ic", type=float, default=IC, help="the oracle's IC in the known-DGP cells"
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=SIGMA,
        help="monthly volatility of each name's noise",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=EPS,
        help="LongShortModule's eps (0: the package default)",
    )
    p.add_argument(
        "--names",
        type=int,
        nargs=2,
        default=list(NAMES),
        help="M in the data-rich and data-poor cells",
    )
    p.add_argument(
        "--regimes",
        nargs="+",
        choices=REGIMES,
        default=list(REGIMES),
        help="the cells' columns to run",
    )
    p.add_argument("--k", type=int, default=K, help="number of features K")
    p.add_argument("--t", type=int, default=T, help="training months T")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="training epochs")
    p.add_argument("--data-seeds", type=int, default=5, help="datasets per cell")
    p.add_argument(
        "--init-seeds", type=int, default=2, help="network seeds per dataset and arm"
    )
    p.add_argument(
        "--n-fresh",
        type=int,
        default=N_FRESH,
        help="fresh windows of T months for the out-of-sample loss",
    )
    p.add_argument(
        "--n-jobs", type=int, default=1, help="fits run in parallel, one process each"
    )
    p.add_argument(
        "--reuse",
        nargs="+",
        default=[],
        metavar="RUN",
        help="earlier (or stopped) runs whose identical fits are copied instead of run again",
    )
    p.add_argument(
        "--plot-only", action="store_true", help="redraw the figures of an existing run"
    )
    p.add_argument(
        "--table", action="store_true", help="print the final values as Markdown"
    )
    args = p.parse_args(argv)
    for loss in args.loss:
        build_loss(loss)

    logs = LOGS / args.run
    if not args.plot_only:
        if logs.exists():
            raise SystemExit(
                f"{logs} exists; runs are never overwritten (use --plot-only to redraw)"
            )
        grid = cells(args.ic, args.names, args.regimes)
        jobs = make_jobs(
            args.loss,
            grid,
            args.data_seeds,
            args.init_seeds,
            args.epochs,
            args.k,
            args.t,
            args.n_fresh,
            args.sigma,
            args.eps,
        )
        print(
            f"{len(jobs)} fits: {len(args.loss)} losses x {len(grid)} cells x {args.data_seeds} data seeds x "
            f"{len(ARMS)} arms x {args.init_seeds} init seeds; {args.epochs} epochs",
            flush=True,
        )
        started = time.time()
        store = LOGS / f"{args.run}.partial"
        reused = seed_store(store, jobs, args.reuse) if args.reuse else {}
        if reused:
            print(f"reused: {reused}", flush=True)
        results = run_jobs(jobs, args.n_jobs, store)
        refs = references_frame(
            args.loss, grid, args.data_seeds, args.k, args.t, args.n_fresh, args.sigma
        )
        logs.mkdir(parents=True)
        curves_frame(results).to_parquet(logs / "curves.parquet", index=False)
        fits_frame(results).to_csv(logs / "fits.csv", index=False)
        refs.to_csv(logs / "references.csv", index=False)
        config = {
            **{
                k.replace("-", "_"): v
                for k, v in vars(args).items()
                if k not in ("plot_only", "table")
            },
            "losses": args.loss,
            "sigma": args.sigma,
            "n_random": N_RANDOM,
            "data_seed0": DATA_SEED0,
            "init_seed0": INIT_SEED0,
            "mlp_kwargs": {
                k: v for k, v in MLP_KWARGS.items() if not k.endswith("scaler")
            },
            "signal_modifier": book_layer_label(args.eps),
            "reused_fits": dict(
                Counter(r["source"] for r in results if r.get("source"))
            ),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
            "seconds": round(time.time() - started, 1),
        }
        (logs / "run_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        shutil.rmtree(store)  # every fit is now in the run folder
    figs = make_figures(args.run)
    print(f"logs: {logs}\nfigures: {figs}")
    if args.table:
        cfg = json.loads((logs / "run_config.json").read_text(encoding="utf-8"))
        print(
            final_table(
                pd.read_parquet(logs / "curves.parquet"),
                pd.read_csv(logs / "references.csv"),
                cfg["losses"],
                cells(cfg["ic"], cfg["names"], cfg.get("regimes", REGIMES)),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
