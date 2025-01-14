import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Optional
from numbers import Number

from macrosynergy.download.transaction_costs import (
    download_transaction_costs,
    AVAIALBLE_COSTS,
    AVAILABLE_STATS,
)
from macrosynergy.management.utils import (
    reduce_df,
    get_cid,
    get_xcat,
)
from macrosynergy.management.types import QuantamentalDataFrame


def get_fids(df: QuantamentalDataFrame) -> list:
    def repl(x: str, yL: List[str]) -> str:
        for y in yL:
            x = x.replace(y, "")
        return x

    fid_endings = [f"{t}_{s}" for t in AVAIALBLE_COSTS for s in AVAILABLE_STATS]
    tickers = QuantamentalDataFrame(df).list_tickers()

    return list(set(map(lambda x: repl(x, fid_endings), tickers)))


def check_df_for_txn_stats(
    df: QuantamentalDataFrame,
    fids: List[str],
    tcost_n: str,
    rcost_n: str,
    size_n: str,
    tcost_l: str,
    rcost_l: str,
    size_l: str,
) -> None:
    expected_tickers = [
        f"{_fid}{txn_ticker}"
        for _fid in fids
        for txn_ticker in [tcost_n, rcost_n, size_n, tcost_l, rcost_l, size_l]
    ]
    found_tickers = QuantamentalDataFrame(df).list_tickers()
    if not set(expected_tickers).issubset(set(found_tickers)):
        raise ValueError(
            "The dataframe is missing the following tickers: "
            + ", ".join(set(expected_tickers) - set(found_tickers))
        )


def get_diff_index(df_wide: pd.DataFrame, freq: str = "D") -> pd.Index:
    df_diff = df_wide.diff(axis=0)
    change_index = df_diff.index[((df_diff.abs() > 0) | df_diff.isnull()).any(axis=1)]
    return change_index


def extrapolate_cost(
    trade_size: Number,
    median_size: Number,
    median_cost: Number,
    pct90_size: Number,
    pct90_cost: Number,
) -> Number:
    err_msg = "`{k}` must be a number > 0"
    if not isinstance(trade_size, Number):
        raise TypeError(err_msg.format(k="trade_size"))
    trade_size = abs(trade_size)

    for k, v in [
        ("trade_size", trade_size),
        ("median_size", median_size),
        ("median_cost", median_cost),
        ("pct90_size", pct90_size),
        ("pct90_cost", pct90_cost),
    ]:
        if not isinstance(v, Number):
            raise TypeError(err_msg.format(k=k))
        if v < 0:
            raise ValueError(err_msg.format(k=k))

    if trade_size <= median_size:
        cost = median_cost
    else:
        b = (pct90_cost - median_cost) / (pct90_size - median_size)
        cost = median_cost + b * (trade_size - median_size)
    return cost


def _plot_costs_func(
    tco: "TransactionCosts",
    fids: Optional[List[str]],
    cost_type: str,
    ncol: int,
    x_axis_label: str,
    y_axis_label: str,
    title: Optional[str] = None,
    title_fontsize: int = 28,
    facet_title_fontsize: int = 20,
    *args,
    **kwargs,
):
    tco.check_init()
    if fids is None:
        fids = tco.fids
    if not isinstance(fids, list) or not all(isinstance(fid, str) for fid in fids):
        raise ValueError("`fids` must be a list of strings")

    costfunc = tco.bidoffer if cost_type == "BIDOFFER" else tco.rollcost

    nrows = len(fids) // ncol + (len(fids) % ncol > 0)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncol, figsize=(5 * ncol, 5 * nrows), layout="constrained"
    )
    if title is None:
        fig.suptitle(f"{cost_type.capitalize()}", fontsize=28)
    else:
        fig.suptitle(title, fontsize=title_fontsize)

    # Define colors for each date range
    colors = sns.color_palette("viridis", n_colors=len(tco.change_index))

    idx_dates = tco.change_index.tolist()

    def label_fmt(x):
        return pd.Timestamp(x).strftime("%Y-%m-%d")

    labels = [
        f"{label_fmt(d1)} to {label_fmt(d2 - pd.offsets.BDay(1))}"
        for d1, d2 in zip(idx_dates[:-1], idx_dates[1:])
    ]
    labels.append(f"{label_fmt(idx_dates[-1])} to Present")
    color_map = dict(zip(labels, colors))

    legend_handles = {}
    ax: plt.Axes
    for i, fid in enumerate(sorted(fids)):
        r, c = divmod(i, ncol)
        ax = axes[r, c] if nrows > 1 else (axes[c] if ncol > 1 else axes)
        max_trade_size = tco.df_wide[fid + tco.size_l].max()
        trade_sizes = np.arange(1, max_trade_size + 101, 1)

        for dt, lb in zip(idx_dates, labels):
            trade_costs = [
                costfunc(fid=fid, trade_size=ts, real_date=dt) for ts in trade_sizes
            ]
            line = sns.lineplot(
                x=trade_sizes,
                y=trade_costs,
                ax=ax,
                color=color_map[lb],
                label=lb,
                zorder=10,
            )

            median_trade_size = tco.df_wide.loc[dt, fid + tco.size_n]
            large_trade_size = tco.df_wide.loc[dt, fid + tco.size_l]
            median_xcost = tco.df_wide.loc[
                dt, fid + (tco.tcost_n if cost_type == "BIDOFFER" else tco.rcost_n)
            ]
            large_xcost = tco.df_wide.loc[
                dt, fid + (tco.tcost_l if cost_type == "BIDOFFER" else tco.rcost_l)
            ]
            sns.scatterplot(
                x=[median_trade_size, large_trade_size],
                y=[median_xcost, large_xcost],
                ax=ax,
                color="red",
                zorder=20,
            )

            ax.set_xlim(left=0)
            ax.set_title(f"{fid}", fontsize=facet_title_fontsize)

            if lb not in legend_handles:
                legend_handles[lb] = line.lines[0]

    # Remove individual subplot legends
    if isinstance(axes, plt.Axes):
        axes.get_legend().remove()
    else:
        for ax in axes.flat[1:]:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    fig.supxlabel(x_axis_label)
    fig.supylabel(y_axis_label)
    plt.show()


class SparseCosts(object):
    def __init__(self, df):
        if not isinstance(df, QuantamentalDataFrame):
            raise TypeError("df must be a QuantamentalDataFrame")
        self.df: QuantamentalDataFrame = QuantamentalDataFrame(df)
        self.prepare_data()

    def prepare_data(self):
        """
        Prepares data for use within the class, including setting up the wide DataFrame
        and fids. This method can be called again to refresh the data and cache.
        """

        df_wide = QuantamentalDataFrame(self.df).to_wide()
        self._all_fids = get_fids(self.df)
        change_index = get_diff_index(df_wide)  # drop rows with no change
        self.change_index: pd.DatetimeIndex = change_index
        df_wide = df_wide.loc[change_index]
        self.df_wide = df_wide

    def get_costs(self, fid: str, real_date: str) -> pd.Series:
        """
        Returns the costs for a given FID and date.

        Parameters
        ----------
        fid : str
            The FID (financial contract identifier) to get costs for.
        real_date : str
            The date to get costs for.
        """
        if fid not in self._all_fids:
            return None
        cost_names = [col for col in self.df_wide.columns if col.startswith(fid)]
        if not cost_names:
            return None
        df_loc = self.df_wide.loc[:real_date, cost_names]
        last_valid_index = df_loc.last_valid_index()
        return df_loc.loc[last_valid_index] if last_valid_index is not None else None


class TransactionCosts(object):
    """
    Interface to query transaction statistics dataframe.
    """

    DEFAULT_ARGS = dict(
        tcost_n="BIDOFFER_MEDIAN",
        rcost_n="ROLLCOST_MEDIAN",
        size_n="SIZE_MEDIAN",
        tcost_l="BIDOFFER_90PCTL",
        rcost_l="ROLLCOST_90PCTL",
        size_l="SIZE_90PCTL",
    )

    def check_init(self) -> bool:
        if not hasattr(self, "sparse_costs") or not hasattr(
            self.sparse_costs, "df_wide"
        ):
            raise ValueError("The TransactionCosts object has not been initialised")
        return True

    def __init__(
        self,
        df: QuantamentalDataFrame,
        fids: List[str],
        tcost_n: str = "BIDOFFER_MEDIAN",
        rcost_n: str = "ROLLCOST_MEDIAN",
        size_n: str = "SIZE_MEDIAN",
        tcost_l: str = "BIDOFFER_90PCTL",
        rcost_l: str = "ROLLCOST_90PCTL",
        size_l: str = "SIZE_90PCTL",
    ) -> None:
        df = QuantamentalDataFrame(df)
        check_df_for_txn_stats(
            df=df,
            fids=fids,
            tcost_n=tcost_n,
            rcost_n=rcost_n,
            size_n=size_n,
            tcost_l=tcost_l,
            rcost_l=rcost_l,
            size_l=size_l,
        )
        self.fids = sorted(set(fids))
        self.tcost_n = tcost_n
        self.rcost_n = rcost_n
        self.size_n = size_n
        self.tcost_l = tcost_l
        self.rcost_l = rcost_l
        self.size_l = size_l
        self._txn_stats = [tcost_n, rcost_n, size_n, tcost_l, rcost_l, size_l]

        _cids = list(set(get_cid(fids)))
        _xcats = [f"{xc}{tc}" for xc in set(get_xcat(fids)) for tc in self._txn_stats]

        df = reduce_df(df=df, cids=_cids, xcats=_xcats)
        # drop all nan rows
        df = df.dropna(axis=0, how="any")
        self.sparse_costs = SparseCosts(df)

    @property
    def change_index(self) -> pd.DatetimeIndex:
        self.check_init()
        return self.sparse_costs.change_index

    @property
    def df_wide(self) -> pd.DataFrame:
        self.check_init()
        return self.sparse_costs.df_wide

    @property
    def qdf(self) -> QuantamentalDataFrame:
        self.check_init()
        return self.sparse_costs.df

    @staticmethod
    def from_qdf(qdf: QuantamentalDataFrame, fids: List[str]) -> "TransactionCosts":
        return TransactionCosts(df=qdf, fids=fids, **TransactionCosts.DEFAULT_ARGS)

    @classmethod
    def download(cls) -> "TransactionCosts":
        df = download_transaction_costs(categorical=True)
        return cls(df=df, fids=get_fids(df), **cls.DEFAULT_ARGS)

    def get_costs(self, fid: str, real_date: str) -> pd.Series:
        self.check_init()
        if fid not in self.fids:
            return None
        return self.sparse_costs.get_costs(fid=fid, real_date=real_date)

    @staticmethod
    def extrapolate_cost(
        trade_size: Number,
        median_size: Number,
        median_cost: Number,
        pct90_size: Number,
        pct90_cost: Number,
    ) -> Number:
        if (not isinstance(trade_size, Number)) or np.isnan(trade_size):
            return 0.0
        return extrapolate_cost(
            trade_size=trade_size,
            median_size=median_size,
            median_cost=median_cost,
            pct90_size=pct90_size,
            pct90_cost=pct90_cost,
        )

    def bidoffer(self, fid: str, trade_size: Number, real_date: str) -> Number:
        self.check_init()
        row = self.sparse_costs.get_costs(fid=fid, real_date=real_date)
        if row is None:
            return np.nan
        d = dict(
            trade_size=trade_size,
            median_size=row[fid + self.size_n],
            median_cost=row[fid + self.tcost_n],
            pct90_size=row[fid + self.size_l],
            pct90_cost=row[fid + self.tcost_l],
        )

        return self.extrapolate_cost(**d)

    def rollcost(self, fid: str, trade_size: Number, real_date: str) -> Number:
        self.check_init()
        row = self.sparse_costs.get_costs(fid=fid, real_date=real_date)
        if row is None:
            return np.nan
        d = dict(
            trade_size=trade_size,
            median_size=row[fid + self.size_n],
            median_cost=row[fid + self.rcost_n],
            pct90_size=row[fid + self.size_l],
            pct90_cost=row[fid + self.rcost_l],
        )
        return self.extrapolate_cost(**d)

    def plot_costs(
        self,
        fids: Optional[List[str]] = None,
        cost_type: str = "BIDOFFER",
        ncol: int = 8,
        x_axis_label: str = "Trade size (USD, millions)",
        y_axis_label: str = "Percent of outright forward",
        title: Optional[str] = None,
        title_fontsize: int = 28,
        facet_title_fontsize: int = 20,
        *args,
        **kwargs,
    ):
        _plot_costs_func(
            tco=self,
            fids=fids,
            cost_type=cost_type,
            ncol=ncol,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            title=title,
            facet_title_fontsize=facet_title_fontsize,
            title_fontsize=title_fontsize,
            *args,
            **kwargs,
        )


class ExampleAdapter(TransactionCosts):  # pragma: no cover
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def extrapolate_cost(
        trade_size: Number,
        median_size: Number,
        median_cost: Number,
        pct90_size: Number,
        pct90_cost: Number,
    ) -> Number:
        # just as an example
        u = median_cost / median_size
        v = pct90_cost / pct90_size
        avg_costs = (u + v) / 2
        return trade_size * avg_costs

    def bidoffer(self, fid: str, trade_size: Number, real_date: str) -> Number:
        return super().bidoffer(fid, trade_size, real_date)

    def somecalc(
        self, fid: str, trade_size: Number, real_date: str, factor=1
    ) -> Number:
        # some random computation
        row = self.sparse_costs.get_costs(fid=fid, real_date=real_date)
        d = dict(
            trade_size=trade_size,
            median_size=row[fid + self.size_n],
            median_cost=row[fid + self.rcost_n],
            pct90_size=row[fid + self.size_l],
            pct90_cost=row[fid + self.rcost_l],
        )
        d["roll_cost"] = d["roll_cost"] * factor
        return self.extrapolate_cost(**d)


if __name__ == "__main__":
    txn_costs_obj: TransactionCosts = TransactionCosts.download()

    test_dict = {
        "GBP_FXROLLCOST_MEDIAN": 0.0022470715369672,
        "GBP_FXSIZE_MEDIAN": 50.0,
        "GBP_FXSIZE_90PCTL": 200.0,
    }

    found_costs = txn_costs_obj.get_costs(
        fid="GBP_FX", real_date="2011-01-01"
    ).to_dict()

    for k, v in test_dict.items():
        assert np.isclose(found_costs[k], v)

    txn_costs_obj.plot_costs(cost_type="ROLLCOST", fids=txn_costs_obj.fids[:16], ncol=4)
