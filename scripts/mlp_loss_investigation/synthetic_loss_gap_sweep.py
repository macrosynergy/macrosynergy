"""
Final in-sample and out-of-sample loss against one setting of `synthetic_loss_gap.py`, the
other settings held fixed.

Each point of a sweep is one run of `synthetic_loss_gap.py` with every setting of run
`main` except the swept one. As there, the network is trained once on a fixed window of T
simulated months and scored on further months from the same process, which it never trains
on. This script starts the runs that are missing, checks that the runs differ only in the
swept setting, and draws each run's final loss (the last epoch) against the swept value.
The questions are whether a longer training window lets the in-sample loss tell noise from
signal, and whether `LongShortModule`'s eps changes the in-sample inflation or opens
another shortcut.

Usage
-----
Run from this folder, next to `synthetic_loss_gap.py`, which this script imports.
`python synthetic_loss_gap_sweep.py --help` lists every option.

    python synthetic_loss_gap_sweep.py --param eps --n-jobs 8 --table
    python synthetic_loss_gap_sweep.py --param t --n-jobs 8 --table
    python synthetic_loss_gap_sweep.py --param eps --plot-only --table

Both sweeps take their base point (T = 120, eps 0) from run `main` and make it if it is
missing. Run the eps sweep first, as above: it makes `main` with all four cells, which the
t sweep can use. The t sweep would make `main` with the data-rich cells only, and the eps
sweep then refuses it. `main` can also be made first on its own:

    python synthetic_loss_gap.py --run main --n-jobs 8 --loss NegSharpeRatio PortfolioVariance NegMeanPortfolioReturn MultiOutputMCR

Without `--loss`, the sweeps use these four losses.

With `--n-jobs 8`, each run of the eps sweep (320 fits) took about 35 minutes, and each run
of the t sweep (160 fits, 30 names) roughly 10 to 20 minutes. More workers ran out of
memory with 467 names.

Requires `synthetic_loss_gap.py` in the same folder, and the packages it requires.

Outputs
-------
The runs are written by `synthetic_loss_gap.py`, in logs/synthetic_loss_gap/<run>/ and
figs/synthetic_loss_gap/<run>/ (see its docstring). A run that exists is used as it is: it
is never run again or overwritten. A run that stopped resumes when the same command is run
again.

Derived from the runs and redrawn on every call:

    logs/synthetic_loss_gap/sweep_<param>/final.csv    final values per swept value, loss,
                                                       cell and arm
    figs/synthetic_loss_gap/sweep_<param>/<loss>.png   one figure per loss

Notes
-----
**The two sweeps.**

    t     training months T = 120, 240, 480, in the data-rich cells (30 names)
    eps   `LongShortModule`'s eps = 0 (the package default), 1e-8, 1, 100, in all four
          cells

**Training months and fresh months.** As in `synthetic_loss_gap.py`, each dataset is one
simulation. The network trains on its first T months, all of them at every step, and is
trained once: there is no walk-forward. The months after them, the fresh months, are never
trained on; they are used only to score the network out of sample.

- In the t sweep the fresh months stay at 2,400 at every T, cut into windows of T months:
  20 windows of 120, 10 of 240, 5 of 480. The out-of-sample loss is always the mean loss
  over windows as long as the training window, like for like with the in-sample loss.
- T stays below 500 months (`T_MAX`), about 40 years of monthly data. The largest value
  is 480 rather than 500 because 2,400 fresh months must cut into whole windows.
- In the eps sweep every run has the data of run `main`: 120 training months and 20
  windows of 120 fresh months.

**The runs.** Each swept value is one run of `synthetic_loss_gap.py`: `main` for the base
value, sweep_t<T> or eps_<eps> for the others (see `run_name`). A missing run is started
from here, with the options of this command and the swept setting (with `--reuse`,
identical fits of earlier runs are copied rather than run again).

**The check.** Before drawing, the runs' run_config.json files are compared: the runs
must differ only in the swept setting, or the script stops with an error.

**The figures.** For each loss: the final loss (last epoch) against the swept setting, in
sample on the left and out of sample on the right, one row of panels per regime. Blue is
REAL and orange NO-FEATURES; known DGP is solid with filled markers, random is dotted with
open markers. Each point is the median over fits, with a bar from the minimum to the
maximum. The oracle is green and zero skill grey.

Differences from the main run
-----------------------------
Every run of a sweep has the settings of run `main` except those below:

    setting                 main run            sweep t                  sweep eps
    training months T       120                 240, 480                 120
    fresh windows           20 of 120 months    10 of 240, 5 of 480      20 of 120 months
    cells                   all four            data rich (30 names)     all four
    LongShortModule eps     0                   0                        1e-8, 1, 100

Everything else is `synthetic_loss_gap.py`'s default in every run: K = 55 features, IC
0.1, noise 8% a month, 2,000 epochs, 5 data seeds and 2 network seeds, the network and its
training (`MLP_KWARGS`). `check_configs` refuses runs that differ in any of these.

What this means:

- **The batch stays full.** batch_size = T at every T, so each epoch is one gradient step
  on the whole training window, and 2,000 epochs are 2,000 steps at every T. A longer
  window gives each step more months, not more steps.
- **The same loadings, other months.** A data seed draws the same loadings beta at every
  T; the months differ.
- **Still fewer months than parameters.** The network has 718 parameters with 30 names,
  more than the months at every T. The t sweep shows how far memorisation weakens as T
  grows; it does not reach the point where the months outnumber the parameters.
- **eps changes only the network's book layer.** `LongShortModule` divides the raw outputs
  by their gross, the sum of their absolute values, but never by less than eps:
  book = raw / max(raw gross, eps). With eps above the raw gross, the book's gross falls
  below 1, so the network can hold less than full capital. The oracle and the random books
  keep `LongShortModule()` (gross 1) at every eps, so the references do not change.

Call stack of a sweep
---------------------
The calls, from this script into `synthetic_loss_gap.py` (imported as `S`):

    main                        this script
      run_name                  the run of each swept value
      S.main                    each run that does not exist yet: its fits, logs and
                                figures (see "Call stack of one fit" in
                                synthetic_loss_gap.py)
      check_configs             the runs differ only in the swept setting
        cfg_get                 a setting, with its default for older runs
      final_by_value            the values written to final.csv
        S.final_values          the loss at the last epoch of every fit, from each run's
                                curves.parquet; the references from references.csv
      plot_sweep                one figure per loss
      sweep_table               with --table

To step through the reading and the drawing once the runs exist, run this in a debugger
with a breakpoint in `final_by_value` or `plot_sweep`:

    import synthetic_loss_gap_sweep as W
    W.main(["--param", "eps", "--plot-only"])

To step through training, see synthetic_loss_gap.py.

Where to change things
----------------------
- **The swept values:** `--values`, or "values" in `PARAMS`. A value of T must cut the
  fresh months (`--fresh-months`) into whole windows and stay within `T_MAX`.
- **The losses:** `--loss`, or `LOSSES`. Every run of the sweep must hold them.
- **Another setting to sweep:** an entry in `PARAMS`, its run name in `run_name`, its tick
  label in `label`, and the options passed to `S.main` in `main` (`extra`). The setting
  must be an option of `synthetic_loss_gap.py` that it writes to run_config.json under
  the same name. Add it to `SAME`, and to `DEFAULTS` if earlier runs lack it.
- **The settings held fixed:** `synthetic_loss_gap.py`'s constants and `MLP_KWARGS`, or the
  options passed on from here (`--names`, `--k`, `--epochs`, `--data-seeds`,
  `--init-seeds`). The run names do not record them: with other values, the existing runs
  fail the check. Move those runs away first, or change `BASE_RUN` and `run_name`.
- **What is drawn:** `final_by_value` for the values of each point, `plot_sweep` for the
  figure.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

import synthetic_loss_gap as S

FRESH_MONTHS = 2400  # out-of-sample months at every T (the main run: 20 windows of 120)
T_MAX = 500  # about 40 years of monthly data; longer windows are not realistic
# The sweeps' default losses: the four of run `main`.
LOSSES = (
    "NegSharpeRatio",
    "PortfolioVariance",
    "NegMeanPortfolioReturn",
    "MultiOutputMCR",
)
BASE_RUN = "main"  # the run at each sweep's base value (T = 120, eps 0)
# Per sweep: its values, the value of the base run, the cells it needs, the run_config.json
# keys allowed to differ between its runs, and its axis label.
PARAMS: Dict[str, Dict[str, Any]] = {
    "t": {
        "values": (120, 240, 480),
        "base": 120,
        "regimes": ("data rich",),
        "differs": ("t", "n_fresh"),
        "label": "training months T",
    },
    "eps": {
        "values": (0.0, 1e-8, 1.0, 100.0),
        "base": 0.0,
        "regimes": S.REGIMES,
        "differs": ("eps", "signal_modifier"),
        "label": "LongShortModule eps",
    },
}
# run_config.json keys that must be equal across the runs of a sweep, except those the
# sweep lets differ ("differs" in PARAMS).
SAME = (
    "k",
    "ic",
    "sigma",
    "epochs",
    "data_seeds",
    "init_seeds",
    "names",
    "n_random",
    "data_seed0",
    "init_seed0",
    "mlp_kwargs",
    "t",
    "n_fresh",
    "eps",
    "signal_modifier",
)
# The value of a setting for runs made before the option existed.
DEFAULTS = {
    "sigma": S.SIGMA,
    "eps": S.EPS,
    "regimes": list(S.REGIMES),
}
DATA_STYLE = {"known DGP": ("-", True), "random": (":", False)}  # line, filled marker


# --------------------------------------------------------------------------- the runs


def cfg_get(cfg: Dict[str, Any], key: str) -> Any:
    """
    A setting of a run, with the default for runs made before the option existed.

    Parameters
    ----------
    cfg : Dict[str, Any]
        The run's run_config.json.
    key : str
        The setting.

    Returns
    -------
    Any
        Its value.
    """
    return cfg.get(key, DEFAULTS.get(key))


def n_fresh_for(t: int, fresh_months: int = FRESH_MONTHS) -> int:
    """
    Number of fresh windows of `t` months that make `fresh_months`.

    Parameters
    ----------
    t : int
        Training months, the window length.
    fresh_months : int
        Fresh months at every T.

    Returns
    -------
    int
        fresh_months / t.

    Raises
    ------
    ValueError
        If the fresh months do not cut into whole windows of `t` months.
    """
    if t <= 0 or fresh_months % t:
        raise ValueError(
            f"{fresh_months} fresh months do not cut into whole windows of T = {t}"
        )
    return fresh_months // t


def run_name(param: str, value: float) -> str:
    """
    Name of the run for one point of a sweep.

    Parameters
    ----------
    param : str
        "t" or "eps".
    value : float
        The swept value.

    Returns
    -------
    str
        `BASE_RUN` ("main") for the base value, else sweep_t<T> or eps_<eps>.
    """
    if value == PARAMS[param]["base"]:
        return BASE_RUN
    return f"sweep_t{int(value)}" if param == "t" else f"eps_{value:g}"


def label(param: str, value: float) -> str:
    """
    Tick label of a swept value, in the figures and the table.

    Parameters
    ----------
    param : str
        "t" or "eps".
    value : float
        The swept value.

    Returns
    -------
    str
        For example "480" or "0 (default)".
    """
    if param == "t":
        return f"{int(value):,}"
    return "0 (default)" if value == 0 else f"{value:g}"


def check_configs(
    cfgs: Dict[float, dict],
    param: str,
    losses: Sequence[str],
    fresh_months: int = FRESH_MONTHS,
) -> None:
    """
    Check that the runs of a sweep differ only in the swept setting.

    The runs are compared with the base run: run `main` if the sweep includes the base
    value, else the run of the smallest value.

    Parameters
    ----------
    cfgs : Dict[float, dict]
        run_config.json of each run, by swept value.
    param : str
        "t" or "eps".
    losses : Sequence[str]
        Losses the sweep needs.
    fresh_months : int
        Fresh months every run of the "t" sweep must have.

    Raises
    ------
    ValueError
        If a run has another swept value than the one it is filed under, lacks a loss or
        a cell, has other fresh months (sweep "t"), or differs from the base run in any
        other setting.
    """
    spec = PARAMS[param]
    base_value = spec["base"] if spec["base"] in cfgs else min(cfgs)
    base = cfgs[base_value]
    for value, cfg in cfgs.items():
        if cfg_get(cfg, param) != value:
            raise ValueError(
                f"the run filed under {param} = {value} has {param} = {cfg_get(cfg, param)}"
            )
        if param == "t" and cfg["t"] * cfg["n_fresh"] != fresh_months:
            raise ValueError(
                f"the run for T = {value} has {cfg['t'] * cfg['n_fresh']} fresh months, not {fresh_months}"
            )
        if not set(losses) <= set(cfg["losses"]):
            raise ValueError(
                f"the run for {param} = {value} lacks {sorted(set(losses) - set(cfg['losses']))}"
            )
        if not set(spec["regimes"]) <= set(cfg_get(cfg, "regimes")):
            raise ValueError(
                f"the run for {param} = {value} lacks the cells {spec['regimes']}"
            )
        for key in SAME:
            if key not in spec["differs"] and cfg_get(cfg, key) != cfg_get(base, key):
                raise ValueError(
                    f"the run for {param} = {value} differs from the base run in {key!r}"
                )


# --------------------------------------------------------------------------- final values


def final_by_value(
    runs: Dict[float, str], losses: Sequence[str], regimes: Sequence[str]
) -> pd.DataFrame:
    """
    The final values of every run of a sweep, as written to final.csv.

    The final value of a fit is its loss at the last epoch where both paths are finite
    (see `synthetic_loss_gap.final_values`).

    Parameters
    ----------
    runs : Dict[float, str]
        Run name by swept value.
    losses : Sequence[str]
        Losses to keep.
    regimes : Sequence[str]
        Regimes to keep.

    Returns
    -------
    pd.DataFrame
        One row per value, loss, cell and arm: the final in-sample and out-of-sample loss
        (median, min and max over fits) and the references (mean over data seeds).
    """
    rows = []
    for value, run in sorted(runs.items()):
        logs = S.LOGS / run
        curves = pd.read_parquet(logs / "curves.parquet")
        curves = curves[curves["loss"].isin(losses) & curves["regime"].isin(regimes)]
        fin = S.final_values(curves)
        refs = pd.read_csv(logs / "references.csv")
        for (loss, regime, data, arm), g in fin.groupby(
            ["loss", "regime", "data", "arm"]
        ):
            ref = refs[
                (refs["loss"] == loss) & (refs["cell"] == f"{data}, {regime}")
            ].mean(numeric_only=True)
            rows.append(
                {
                    "value": value,
                    "run": run,
                    "loss": loss,
                    "regime": regime,
                    "data": data,
                    "arm": arm,
                    "fits": len(g),
                    **{
                        f"{side}{stat}": getattr(g[side], fn)()
                        for side in ("insample", "oos")
                        for stat, fn in (("", "median"), ("_lo", "min"), ("_hi", "max"))
                    },
                    "oracle_insample": ref["oracle_insample"],
                    "oracle_oos": ref["oracle_oos"],
                    "zero_skill_oos": ref["zero_skill_oos"],
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figure and table


def plot_sweep(df: pd.DataFrame, loss: str, param: str, path) -> None:
    """
    Draw one loss against the swept setting.

    In sample on the left, out of sample on the right, one row of panels per regime. Each
    point is the median over fits, with a bar from the minimum to the maximum; the four
    lines (two arms, two kinds of data) are set slightly apart around each tick. The
    oracle (green) and zero skill (grey) are drawn at each value.

    Parameters
    ----------
    df : pd.DataFrame
        `final_by_value`'s output, with a column "m" (names per regime).
    loss : str
        The loss to draw.
    param : str
        "t" or "eps".
    path : Path
        Output file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sub = df[df["loss"] == loss]
    values = sorted(sub["value"].unique())
    pos = {v: i for i, v in enumerate(values)}
    regimes = [r for r in S.REGIMES if r in set(sub["regime"])]
    fig, axes = plt.subplots(
        len(regimes), 2, figsize=(12, 4.2 * len(regimes)), squeeze=False
    )
    # The four lines are set side by side around each tick, so the bars do not overlap.
    shift = {
        ("real", "known DGP"): -0.12,
        ("real", "random"): -0.04,
        ("no_features", "known DGP"): 0.04,
        ("no_features", "random"): 0.12,
    }
    for row, regime in zip(axes, regimes):
        reg = sub[sub["regime"] == regime]
        for ax, side in zip(row, ("insample", "oos")):
            for (arm, data), dx in shift.items():
                g = reg[(reg["arm"] == arm) & (reg["data"] == data)].sort_values(
                    "value"
                )
                if g.empty:
                    continue
                color = S.ARM_STYLE[arm][0]
                line, filled = DATA_STYLE[data]
                x = np.array([pos[v] for v in g["value"]]) + dx
                ax.errorbar(
                    x,
                    g[side],
                    yerr=[g[side] - g[f"{side}_lo"], g[f"{side}_hi"] - g[side]],
                    color=color,
                    linestyle=line,
                    marker="o",
                    markerfacecolor=color if filled else "white",
                    capsize=3,
                    linewidth=1.3,
                    markersize=5,
                )
            oracle = (
                reg[reg["data"] == "known DGP"]
                .groupby("value")[f"oracle_{side}"]
                .mean()
            )
            if oracle.notna().any():
                ax.plot(
                    [pos[v] for v in oracle.index],
                    oracle.to_numpy(),
                    color="C2",
                    linestyle="-" if side == "insample" else "--",
                    marker="^",
                    markersize=5,
                    linewidth=1.0,
                )
            if side == "oos":
                zero = reg.groupby("value")["zero_skill_oos"].mean()
                ax.plot(
                    [pos[v] for v in zero.index],
                    zero.to_numpy(),
                    color="0.5",
                    linestyle=":",
                    linewidth=1.0,
                )
            vals = np.concatenate(
                [
                    reg[c].to_numpy(dtype=float)
                    for c in (side, f"{side}_lo", f"{side}_hi")
                ]
            )
            S._set_scale(ax, vals, "y")
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels([label(param, v) for v in values])
            ax.set_xlabel(PARAMS[param]["label"])
            ax.set_ylabel("final loss (lower is better)")
            m = int(reg["m"].iloc[0]) if "m" in reg else None
            ax.set_title(
                f"{regime}{f' ({m} names)' if m else ''}: "
                + ("in sample" if side == "insample" else "out of sample"),
                fontsize=10,
            )
            ax.grid(True, alpha=0.3)
    handles = [
        Line2D(
            [],
            [],
            color=S.ARM_STYLE[arm][0],
            linestyle=DATA_STYLE[data][0],
            marker="o",
            markerfacecolor=S.ARM_STYLE[arm][0] if DATA_STYLE[data][1] else "white",
            label=f"{S.ARM_STYLE[arm][1]}, {data}",
        )
        for arm in S.ARMS
        for data in ("known DGP", "random")
    ]
    handles += [
        Line2D(
            [],
            [],
            color="C2",
            marker="^",
            label="oracle (solid in sample, dashed out of sample)",
        ),
        Line2D(
            [], [], color="0.5", linestyle=":", label="zero skill: random fixed books"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    n = int(sub["fits"].max()) if len(sub) else 0
    fig.suptitle(
        f"{loss}: final loss against {PARAMS[param]['label']}; median of up to {n} fits, bars min to max",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.1 if len(regimes) == 1 else 0.06, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def sweep_table(df: pd.DataFrame, losses: Sequence[str], param: str) -> str:
    """
    The final values of a sweep as a Markdown table.

    Parameters
    ----------
    df : pd.DataFrame
        `final_by_value`'s output.
    losses : Sequence[str]
        Losses to show.
    param : str
        "t" or "eps".

    Returns
    -------
    str
        Per loss, regime and value: the median final loss of each arm and kind of data,
        in and out of sample, and the references.
    """
    g = lambda v: "nan" if not np.isfinite(v) else f"{v:.4g}"  # noqa: E731
    cols = [
        ("real", "random", "insample"),
        ("real", "known DGP", "insample"),
        ("real", "random", "oos"),
        ("real", "known DGP", "oos"),
        ("no_features", "random", "insample"),
        ("no_features", "known DGP", "insample"),
        ("no_features", "random", "oos"),
        ("no_features", "known DGP", "oos"),
    ]
    head = [
        f"{S.ARM_STYLE[a][1]} {'in' if s == 'insample' else 'out'}, {d}"
        for a, d, s in cols
    ]
    lines = [
        f"| loss | cell | {param} | "
        + " | ".join(head)
        + " | oracle in | oracle out | zero skill out |",
        "|---|---|---|" + "---|" * (len(cols) + 3),
    ]
    idx = df.set_index(["loss", "regime", "value", "arm", "data"])
    for loss in losses:
        for regime in [r for r in S.REGIMES if r in set(df["regime"])]:
            for value in sorted(df["value"].unique()):
                vals = [
                    idx[side].get((loss, regime, value, arm, data), np.nan)
                    for arm, data, side in cols
                ]
                here = df[
                    (df["loss"] == loss)
                    & (df["regime"] == regime)
                    & (df["value"] == value)
                ]
                dgp = here[here["data"] == "known DGP"]
                vals += [
                    dgp["oracle_insample"].mean(),
                    dgp["oracle_oos"].mean(),
                    here["zero_skill_oos"].mean(),
                ]
                lines.append(
                    f"| `{loss}` | {regime} | {label(param, value)} | "
                    + " | ".join(g(float(v)) for v in vals)
                    + " |"
                )
    return "\n".join(lines)


# --------------------------------------------------------------------------- main


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the missing points of a sweep, check the runs, write final.csv and draw the
    figures.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Command-line arguments; `python synthetic_loss_gap_sweep.py --help` lists them.

    Returns
    -------
    int
        0 on success.
    """
    p = argparse.ArgumentParser(
        description="Final in-sample and out-of-sample loss against one setting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--param",
        choices=sorted(PARAMS),
        default="t",
        help="the setting to sweep: t (training months) or eps (LongShortModule's eps)",
    )
    p.add_argument(
        "--values",
        type=float,
        nargs="+",
        default=None,
        help="values of the setting; if not given, t 120 240 480 or eps 0 1e-8 1 100",
    )
    p.add_argument(
        "--fresh-months",
        type=int,
        default=FRESH_MONTHS,
        help="out-of-sample months at every T (sweep t only)",
    )
    p.add_argument("--loss", nargs="+", default=list(LOSSES), help="losses to run")
    p.add_argument(
        "--names",
        type=int,
        nargs=2,
        default=list(S.NAMES),
        help="M in the data-rich and data-poor cells",
    )
    p.add_argument("--k", type=int, default=S.K, help="number of features K")
    p.add_argument("--epochs", type=int, default=S.EPOCHS, help="training epochs")
    p.add_argument("--data-seeds", type=int, default=5, help="datasets per cell")
    p.add_argument(
        "--init-seeds", type=int, default=2, help="network seeds per dataset and arm"
    )
    p.add_argument(
        "--n-jobs", type=int, default=1, help="fits run in parallel, one process each"
    )
    p.add_argument(
        "--reuse",
        nargs="+",
        default=[],
        metavar="RUN",
        help="earlier (or stopped) runs whose identical fits are copied instead of run "
        "again, in every run started here",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="run nothing; redraw from the existing runs",
    )
    p.add_argument(
        "--table", action="store_true", help="print the final values as Markdown"
    )
    args = p.parse_args(argv)
    spec = PARAMS[args.param]
    values = sorted(args.values if args.values is not None else spec["values"])
    if args.param == "t":
        values = [int(v) for v in values]
        if max(values) > T_MAX:
            raise SystemExit(f"T = {max(values)} is above the cap of {T_MAX} months")

    # Start every run that does not exist yet, with this command's options and the swept
    # setting. A run that exists is used as it is.
    runs = {v: run_name(args.param, v) for v in values}
    for value, run in runs.items():
        if (S.LOGS / run).exists():
            continue
        if args.plot_only:
            raise SystemExit(f"run {run!r} for {args.param} = {value} does not exist")
        # Sweep t: the data-rich cells only, and the fresh months cut into windows of T.
        extra = (
            [
                "--regimes",
                *spec["regimes"],
                "--t",
                str(value),
                "--n-fresh",
                str(n_fresh_for(value, args.fresh_months)),
            ]
            if args.param == "t"
            else ["--eps", repr(float(value))]
        )
        print(f"{args.param} = {value}: run {run}", flush=True)
        S.main(
            [
                "--run",
                run,
                "--names",
                *map(str, args.names),
                "--k",
                str(args.k),
                "--epochs",
                str(args.epochs),
                "--data-seeds",
                str(args.data_seeds),
                "--init-seeds",
                str(args.init_seeds),
                "--n-jobs",
                str(args.n_jobs),
                "--loss",
                *args.loss,
                *extra,
            ]
            + (["--reuse", *args.reuse] if args.reuse else [])
        )

    # The runs must differ only in the swept setting; then final.csv and the figures.
    cfgs = {
        v: json.loads((S.LOGS / run / "run_config.json").read_text(encoding="utf-8"))
        for v, run in runs.items()
    }
    check_configs(cfgs, args.param, args.loss, args.fresh_months)
    df = final_by_value(runs, args.loss, spec["regimes"])
    m = {c.regime: c.m for c in S.cells(names=args.names)}
    df["m"] = df["regime"].map(m)
    out = S.LOGS / f"sweep_{args.param}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "final.csv", index=False)
    for loss in args.loss:
        plot_sweep(df, loss, args.param, S.FIGS / f"sweep_{args.param}" / f"{loss}.png")
    print(
        f"runs: {runs}\ntable: {out / 'final.csv'}\nfigures: {S.FIGS / f'sweep_{args.param}'}"
    )
    if args.table:
        print(sweep_table(df, args.loss, args.param))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
