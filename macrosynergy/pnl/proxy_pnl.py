"""
Implementation of the ProxyPnL class.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numbers import Number
from typing import List, Union, Tuple, Optional, Dict

from macrosynergy.management.utils import (
    reduce_df,
    is_valid_iso_date,
    _map_to_business_day_frequency,
)
from macrosynergy.management.types import QuantamentalDataFrame, NoneType
import macrosynergy.visuals as msv
from macrosynergy.pnl import notional_positions, contract_signals, proxy_pnl_calc
from macrosynergy.pnl.sharpe_stability_ratio import sharpe_stability_ratio

from macrosynergy.pnl.transaction_costs import (
    TransactionCosts,
    TransactionCostsDictAdapter,
)


class ProxyPnL(object):
    """
    The purpose of this class is to facilitate PnL estimation under the consideration of
    AUM, volatility targeting or leverage, and transaction costs. The class is designed
    to be used in a step-by-step manner, where the user first contracts signals, then
    calculates notional positions, and finally calculates the proxy PnL.

    The steps for generating the PnL are as follows:
    - Contract signals: Contract signals for the given contracts and contract types.
    - Notional positions: Calculate notional (dollar) positions for the given contract
        signals.
    - Proxy PnL calculation: Calculate the proxy PnL and transaction costs for the given
        notional positions.

    Parameters
    ----------
    df : QuantamentalDataFrame
        DataFrame containing the data to be used in the PnL estimation. Initially, this
        DataFrame should contain the data used to contract signals (i.e. raw signals).
    transaction_costs_object : Optional[Union[TransactionCosts, TransactionCostsDictAdapter]]
        Object containing the transaction costs data.
    start : str, optional
        Start date for the PnL estimation. If not provided, the minimum date in the
        DataFrame is used.
    end : str, optional
        End date for the PnL estimation. If not provided, the maximum date in the
        DataFrame is used.
    blacklist : dict, optional
        The blacklist dictionary to be applied to the input data.
    rstring : str, optional
        A string used to specify the returns to be used in the PnL estimation.
    portfolio_name : str, optional
        The name given to the (current) portfolio. In the return outputs, the portfolio
        name is used to identify and aggregate the PnL and transaction costs.
    sname : str, optional
        The name given to the strategy,
    pname : str, optional
        The name given to the positions.

    """

    def __init__(
        self,
        df: QuantamentalDataFrame,
        transaction_costs_object: Optional[
            Union[TransactionCosts, TransactionCostsDictAdapter]
        ] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        blacklist: Optional[dict] = None,
        rstring: str = "XR",
        portfolio_name: str = "GLB",
        sname: str = "STRAT",
        pname: str = "POS",
    ):
        self.sname = sname
        self.portfolio_name = portfolio_name
        self.pname = pname
        self.blacklist = blacklist
        self.cs_df = None
        self.npos_df = None
        self.df = reduce_df(df=QuantamentalDataFrame(df), blacklist=blacklist)
        self.start = start or df["real_date"].min().strftime("%Y-%m-%d")
        self.end = end or df["real_date"].max().strftime("%Y-%m-%d")
        self.rstring = rstring
        self.transaction_costs_object: Optional[TransactionCosts] = None
        if not all(map(is_valid_iso_date, [self.start, self.end])):
            raise ValueError(f"Invalid date format: {self.start}, {self.end}")

        if transaction_costs_object is None:
            pass  # allowed for no-transaction-costs case
        elif isinstance(transaction_costs_object, TransactionCosts):
            transaction_costs_object.check_init()
            self.transaction_costs_object: TransactionCosts = transaction_costs_object
        elif isinstance(transaction_costs_object, TransactionCostsDictAdapter):
            transaction_costs_object.check_init()
            self.transaction_costs_object: TransactionCostsDictAdapter = (
                transaction_costs_object
            )
        else:
            raise ValueError(
                "Invalid type for `transaction_costs_object`."
                " Expected `TransactionCosts` or `TransactionCostsDictAdapter` object."
            )

        assert hasattr(
            self, "transaction_costs_object"
        ), "Failed to initialize `self.transaction_costs_object`"

    def contract_signals(
        self,
        sig: str,
        cids: List[str],
        ctypes: List[str],
        cscales: Optional[List[Union[Number, str]]] = None,
        csigns: Optional[List[int]] = None,
        basket_contracts: Optional[List[str]] = None,
        basket_weights: Optional[List[Union[Number, str]]] = None,
        hedge_xcat: Optional[str] = None,
        blacklist: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> QuantamentalDataFrame:
        """
        Contract signals for the given contracts and contract types.
        The method uses the same dataframe as the one used to initialize the class.
        The function stores the contract signals DataFrame as an attribute of the class
        (`self.cs_df`), and also returns the same DataFrame for convenience.

        See :func:`macrosynergy.pnl.contract_signals` for more information on the other
        parameters.

        Returns
        -------
        QuantamentalDataFrame
        """
        self.fids = [f"{cid}_{ctype}" for cid in cids for ctype in ctypes]
        cs_df: QuantamentalDataFrame = contract_signals(
            df=self.df,
            sig=sig,
            cids=cids,
            ctypes=ctypes,
            cscales=cscales,
            csigns=csigns,
            basket_contracts=basket_contracts,
            basket_weights=basket_weights,
            hedge_xcat=hedge_xcat,
            start=self.start,
            end=self.end,
            blacklist=blacklist or self.blacklist,
            sname=self.sname,
            *args,
            **kwargs,
        )
        self.cs_df: QuantamentalDataFrame = cs_df
        return cs_df

    def notional_positions(
        self,
        df: QuantamentalDataFrame = None,
        sname: str = None,
        fids: List[str] = None,
        aum: Number = 100,
        dollar_per_signal: Number = 1.0,
        slip: int = 1,
        leverage: Optional[Number] = None,
        vol_target: Optional[Number] = None,
        nan_tolerance: float = 0.25,
        remove_zeros: bool = True,
        rebal_freq: str = "m",
        lback_meth: str = "ma",
        est_freqs: Union[str, List[str]] = ["D", "W", "M"],
        est_weights: Union[Number, List[Number]] = [1, 1, 1],
        lback_periods: Union[int, List[int]] = [-1, -1, -1],
        half_life: Union[int, List[int]] = [11, 5, 6],
        rstring: str = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        blacklist: Optional[dict] = None,
        pname: str = "POS",
    ) -> Union[
        QuantamentalDataFrame,
        Tuple[QuantamentalDataFrame, QuantamentalDataFrame],
        Tuple[QuantamentalDataFrame, pd.DataFrame],
        Tuple[QuantamentalDataFrame, QuantamentalDataFrame, pd.DataFrame],
    ]:
        """
        Calculate notional positions for the given contract signals.
        The method uses the contract signals calculated in the previous step. The user
        may additionally provide more data that may be used as a new dataframe.

        The method stores the notional positions DataFrame, the portfolio volatility
        DataFrame, and the variance-covariance matrix DataFrame as attributes of the
        class (`self.npos_df`, `self.pvol_df`, and `self.vcv_df`, respectively). It also
        returns the notional positions DataFrame for convenience.

        See :func:`macrosynergy.pnl.notional_positions` for more information on the other
        parameters.

        Returns
        -------
        QuantamentalDataFrame
            The notional positions DataFrame
        """

        fids = fids or self.fids
        if df is None:
            if hasattr(self, "cs_df") and self.cs_df is not None:
                df = self.cs_df
            else:
                raise ValueError(
                    "Either pass a DataFrame with contract signals "
                    "or run `ProxyPnL.contract_signals` first."
                )
        sname = sname or self.sname
        start = start or self.start
        end = end or self.end
        blacklist = blacklist or self.blacklist
        rstring = rstring or self.rstring

        outs: Union[
            Tuple[QuantamentalDataFrame, QuantamentalDataFrame, pd.DataFrame],
            QuantamentalDataFrame,
        ] = notional_positions(
            df=pd.concat((self.df, df), axis=0),
            sname=sname,
            fids=fids,
            aum=aum,
            dollar_per_signal=dollar_per_signal,
            slip=slip,
            leverage=leverage,
            vol_target=vol_target,
            nan_tolerance=nan_tolerance,
            remove_zeros=remove_zeros,
            rebal_freq=rebal_freq,
            lback_meth=lback_meth,
            est_freqs=est_freqs,
            est_weights=est_weights,
            lback_periods=lback_periods,
            half_life=half_life,
            rstring=rstring,
            start=start,
            end=end,
            blacklist=blacklist,
            pname=pname,
            return_pvol=True,
            return_vcv=True,
        )
        if isinstance(outs, QuantamentalDataFrame):
            assert isinstance(outs, QuantamentalDataFrame)
            outs = (outs, None, None)  # to avoid multiple flow control
        assert len(outs) == 3
        assert isinstance(outs[0], QuantamentalDataFrame)
        assert isinstance(outs[1], (QuantamentalDataFrame, NoneType))
        assert isinstance(outs[2], (pd.DataFrame, NoneType))

        self.npos_df: QuantamentalDataFrame = outs[0]
        self.pvol_df: QuantamentalDataFrame = outs[1]
        self.vcv_df: QuantamentalDataFrame = outs[2]
        outs = None
        return self.npos_df

    def proxy_pnl_calc(
        self,
        spos: str = None,
        portfolio_name: str = None,
        df: QuantamentalDataFrame = None,
        roll_freq: Optional[Union[str, dict]] = None,
        rstring: str = None,
        pnl_name: str = "PNL",
        tc_name: str = "TCOST",
    ) -> Union[QuantamentalDataFrame, Tuple[QuantamentalDataFrame, ...]]:
        """
        Calculate the proxy PnL and transaction costs for the given notional positions.
        The method uses the notional positions calculated in the previous step. The user
        may additionally provide more data that may be used as a new dataframe.

        The method stores the proxy PnL DataFrame, the transaction costs DataFrame, and
        the proxy PnL excluding costs DataFrame as attributes of the class (`self.proxy_pnl`,
        `self.txn_costs_df`, and `self.pnl_excl_costs`, respectively). It also returns the
        proxy PnL DataFrame for convenience.

        See :func:`macrosynergy.pnl.proxy_pnl_calc` for more information on the other
        parameters.

        Returns
        -------
        QuantamentalDataFrame
            The proxy PnL DataFrame.
        """
        if df is None:
            if hasattr(self, "npos_df") and self.npos_df is not None:
                df = self.npos_df
            else:
                raise ValueError(
                    "Either pass a DataFrame with notional positions "
                    "or run `ProxyPnL.notional_positions` (and `contract_signals`) first."
                )
        spos: str = spos or (self.sname + "_" + self.pname)
        portfolio_name: str = portfolio_name or self.portfolio_name
        rstring: str = rstring or self.rstring

        outs: Tuple[QuantamentalDataFrame, ...] = proxy_pnl_calc(
            df=pd.concat((self.df, df), axis=0),
            transaction_costs_object=self.transaction_costs_object,
            spos=spos,
            rstring=rstring,
            portfolio_name=portfolio_name,
            roll_freq=roll_freq,
            start=self.start,
            end=self.end,
            blacklist=self.blacklist,
            pnl_name=pnl_name,
            tc_name=tc_name,
            return_pnl_excl_costs=True,
            return_costs=True,
        )
        assert len(outs) == 3
        assert all(map(lambda x: isinstance(x, QuantamentalDataFrame), outs))
        self.proxy_pnl: QuantamentalDataFrame = outs[0]
        self.pnl_excl_costs: QuantamentalDataFrame = outs[1]
        self.txn_costs_df: QuantamentalDataFrame = outs[2]
        outs = None

        return self.proxy_pnl

    def evaluate_pnl(
        self,
        aum: Number,
        include_pnle: bool = False,
        include_tcosts: bool = False,
        label_dict: Optional[Dict[str, str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute summary performance statistics for the proxy PnL.

        The PnL series is converted to a percentage return on AUM and annualized
        statistics are computed assuming 252 trading days per year. The method
        requires that `proxy_pnl_calc` has already been run; `pnl_excl_costs` and
        `txn_costs_df` are required only when the corresponding flags are set.

        Parameters
        ----------
        aum : Number
            Assets under management used to scale the PnL into percentage returns.
        include_pnle : bool
            If True, include the PnL excluding transaction costs (`self.pnl_excl_costs`)
            as an additional column in the output.
        include_tcosts : bool
            If True, include total transaction costs as a row in the output. Requires
            `self.txn_costs_df` to be available.
        label_dict : dict
            Mapping from raw column names (xcat values) to display labels used in the
            output columns.
        start : str
            Start date (ISO format) used to filter the PnL prior to computing statistics.
            If not provided, no lower bound is applied.
        end : str
            End date (ISO format) used to filter the PnL prior to computing statistics.
            If not provided, no upper bound is applied.
        benchmark_data : pd.DataFrame
            QuantamentalDataFrame of benchmark series. If provided, the correlation
            between each PnL column and each benchmark ticker (cid_xcat) is added as
            a row in the output.

        Returns
        -------
        pd.DataFrame
            Summary statistics with one column per PnL series. Rows include the
            annualized return and standard deviation (in %), Sharpe and Sortino
            ratios, Sharpe stability, maximum 21-day, 6-month and peak-to-trough
            drawdowns (in %), the share of total PnL contributed by the top 5%
            of months, optional benchmark correlations, optional transaction
            costs, and the number of traded months.
        """
        # Input validation
        for arg, value, types in [
            ("aum", aum, Number),
            ("include_pnle", include_pnle, bool),
            ("include_tcosts", include_tcosts, bool),
            ("label_dict", label_dict, (dict, type(None))),
            ("start", start, (str, type(None))),
            ("end", end, (str, type(None))),
            ("benchmark_data", benchmark_data, (pd.DataFrame, type(None))),
        ]:
            if not isinstance(value, types):
                raise TypeError(f"Argument {arg} must be one of: {types}")

        pnl_exists = hasattr(self, "proxy_pnl") and self.proxy_pnl is not None
        pnle_exists = hasattr(self, "pnl_excl_costs") and self.pnl_excl_costs is not None
        tcosts_exists = hasattr(self, "txn_costs_df") and self.txn_costs_df is not None

        missing_data_msg = "self.{} is missing"
        if not pnl_exists:
            raise ValueError(missing_data_msg.format("proxy_pnl"))
        if not pnle_exists and include_pnle:
            raise ValueError(missing_data_msg.format("pnl_excl_costs"))
        if not tcosts_exists and include_tcosts:
            raise ValueError(missing_data_msg.format("txn_costs_df"))


        # Data preparation
        df_pnl = self.proxy_pnl
        df_pnle = self.pnl_excl_costs if include_pnle else pd.DataFrame()

        df = pd.concat((df_pnl, df_pnle), ignore_index=True)
        df = reduce_df(df, cids=[self.portfolio_name], start=start, end=end)

        dfw = df.pivot(index="real_date", columns="xcat", values="value")
        dfw = 100 * dfw / aum  # percentage return instead of $
        dfw = dfw.rename(columns=label_dict if label_dict is not None else {})

        # Summary statistics
        ## Annualized mean and std
        mean = dfw.mean(axis=0) * 252
        std = dfw.std(axis=0) * np.sqrt(252)

        ## Sharpes and Sortino
        sharpe = mean / std
        sortino = np.divide(
            mean,
            dfw.apply(lambda x: np.sqrt(np.sum(x[x < 0] ** 2) / len(x))) * np.sqrt(252),
        )
        sharpe_stability = [
            sharpe_stability_ratio(
                dfw[col].dropna(),
                window=252,
                benchmark_sr=0.0,
                annualization_factor=252,
            )
            for col in dfw.columns
        ]

        ## Draws
        draw_21_day = dfw.rolling(21).sum().min()
        draw_6_month = dfw.rolling(6 * 21).sum().min()
        draw_peak_to_trough = -(dfw.cumsum().cummax() - dfw.cumsum()).max()

        ## PnL share
        mfreq = _map_to_business_day_frequency("M")
        monthly_pnl = dfw.resample(mfreq).sum()
        total_pnl = monthly_pnl.sum(axis=0)
        n_top = int(max(np.ceil(len(monthly_pnl) * 0.05), 1))
        n_top_pnl = -np.sort(-monthly_pnl.values, axis=0)[:n_top].sum(0)
        pnl_share = n_top_pnl / total_pnl

        ## Number of traded months
        n_traded_months = dfw.notna().resample(mfreq).sum().ne(0).sum()

        ## Benchmark correlations
        correlations = {}
        if benchmark_data is not None and not benchmark_data.empty:
            bm_data = benchmark_data.copy()
            bm_data["ticker"] = bm_data["cid"] + "_" + bm_data["xcat"]
            bm_data_w = bm_data.pivot(
                index="real_date", columns="ticker", values="value"
            )
            shared_idx = dfw.index.intersection(bm_data_w.index)
            correlations = {
                f"{bm} correl": dfw.loc[shared_idx].corrwith(
                    other=bm_data_w.loc[shared_idx][bm],
                    drop=True,
                )
                for bm in bm_data_w.columns
            }

        ## Transaction costs
        tcosts = {}
        if include_tcosts:
            txn_costs = reduce_df(
                df=self.txn_costs_df,
                cids=[self.portfolio_name],
                blacklist=self.blacklist,
            )
            total_txn_costs = txn_costs["value"].sum()
            total_txn_cost = [total_txn_costs, 0] if include_pnle else [total_txn_costs]
            tcosts["Transaction Cost"] = total_txn_cost

        # Format output
        summary_statistics = {
            "Return %": mean,
            "St. Dev. %": std,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Sharpe Stability": sharpe_stability,
            "Max 21-Day Draw %": draw_21_day,
            "Max 6-Month Draw %": draw_6_month,
            "Peak to Trough Draw %": draw_peak_to_trough,
            "Top 5% Monthly PnL Share": pnl_share,
            **correlations,
            **tcosts,
            "Traded Months": n_traded_months,
        }

        summary_statistics = pd.DataFrame(summary_statistics).T
        summary_statistics.columns = dfw.columns

        return summary_statistics

    def plot(
        self,
        cids: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        mark_cost_events: bool = True,
        cumsum: bool = True,
        title: str = "ProxyPnL summary",
        figsize: Tuple[float, float] = (16, 10),
        return_fig: bool = False,
        **kwargs,
    ) -> Optional[Figure]:
        """
        Render a composite 2x2 summary of the proxy PnL run.

        The method draws a single figure with four panels: contract signals
        (top-left), notional positions with optional cost-event markers
        (top-right), cumulative PnL excluding and including transaction costs
        (bottom-left), and a transaction-cost timeline split by bid-offer and
        roll cost (bottom-right). The method requires that `contract_signals`,
        `notional_positions`, and `proxy_pnl_calc` have all been run; if any
        of the underlying attributes are missing, a `RuntimeError` is raised
        naming the step the user needs to run first.

        Parameters
        ----------
        cids : List[str]
            Contracts (cids) to include in the signals and positions panels.
            If not provided, all contracts found in the positions frame are
            shown. The PnL and costs panels always show the portfolio-level
            aggregate and are not affected by this filter.
        start : str
            ISO-format lower bound applied to the time axis of every panel.
            If not provided, no lower bound is applied.
        end : str
            ISO-format upper bound applied to the time axis of every panel.
            If not provided, no upper bound is applied.
        mark_cost_events : bool
            If True, overlay markers on the positions panel at the dates on
            which a bid-offer or roll-cost charge was incurred. Bid-offer
            charges are drawn as upward triangles, roll-cost charges as open
            rings, with marker area scaled by charge magnitude within each
            cost type. Silently treated as False when no transaction costs
            are available.
        cumsum : bool
            If True (default), the PnL panel plots the cumulative PnL series;
            if False, the daily series is plotted instead.
        title : str
            Figure suptitle.
        figsize : tuple
            Figure size in inches, passed to `plt.subplots`.
        return_fig : bool
            If True, the `matplotlib.figure.Figure` is returned to the caller
            after rendering. If False (default), the figure is shown and None
            is returned.
        kwargs
            Additional keyword arguments forwarded to `plt.subplots`.

        Returns
        -------
        Optional[matplotlib.figure.Figure]
            The figure object if `return_fig` is True, otherwise None.
        """
        # Validate that the pipeline has been run end-to-end. Each missing
        # attribute is mapped back to the method the user needs to call.
        required = {
            "cs_df": "contract_signals",
            "npos_df": "notional_positions",
            "proxy_pnl": "proxy_pnl_calc",
            "pnl_excl_costs": "proxy_pnl_calc",
            "txn_costs_df": "proxy_pnl_calc",
        }
        missing = [a for a in required if getattr(self, a, None) is None]
        if missing:
            steps = sorted({required[a] for a in missing})
            steps_str = ", ".join(f"ProxyPnL.{s}" for s in steps)
            raise RuntimeError(
                f"Cannot plot: missing attribute(s) {missing}. "
                f"Run {steps_str} first."
            )

        for arg, value in (("start", start), ("end", end)):
            if value is not None and not is_valid_iso_date(value):
                raise ValueError(f"Invalid {arg} date format: {value!r}")

        # Determine which contracts to show in the per-contract panels. The
        # portfolio aggregate row is dropped from the contract universe so
        # the signals and positions panels only show real contracts.
        pos_wide = QuantamentalDataFrame(self.npos_df).to_wide()
        pos_suffix = f"_{self.sname}_{self.pname}"
        pos_cols_all = [c for c in pos_wide.columns if c.endswith(pos_suffix)]
        available_fids = [c[: -len(pos_suffix)] for c in pos_cols_all]
        available_cids = sorted({fid.split("_", 1)[0] for fid in available_fids})
        available_cids = [c for c in available_cids if c != self.portfolio_name]

        if cids is not None:
            plot_cids = [c for c in cids if c in available_cids]
            if not plot_cids:
                raise ValueError(
                    f"None of {cids} found in positions. "
                    f"Available contracts: {available_cids}"
                )
        else:
            plot_cids = available_cids

        # Time-window helper used by every panel so that all four panels share
        # the same date filter without duplicating the slicing logic.
        start_ts = pd.Timestamp(start) if start is not None else None
        end_ts = pd.Timestamp(end) if end is not None else None

        def _window(df):
            if start_ts is not None:
                df = df.loc[df.index >= start_ts]
            if end_ts is not None:
                df = df.loc[df.index <= end_ts]
            return df

        def _annotate_empty(ax, msg):
            ax.text(
                0.5,
                0.5,
                msg,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="gray",
            )
            ax.set_xticks([])
            ax.set_yticks([])

        # Determine whether transaction-cost data is actually available. An
        # empty txn_costs_df means proxy_pnl_calc was run without a cost
        # object; the cost panel renders an annotation and the PnL panel
        # falls back to a single ex-costs line.
        tc_available = (
            isinstance(self.txn_costs_df, pd.DataFrame) and not self.txn_costs_df.empty
        )

        # Fixed colour palette (matches the diagnostic script in PROXY_PNL.md
        # so the two figures are immediately recognisable as describing the
        # same pipeline).
        c_bo = "#1f77b4"
        c_rc = "#d62728"
        c_pnle = "#2ca02c"
        c_pnl = "#9467bd"
        c_glb = "#d62728"
        c_cum = "#404040"

        fig, axes = plt.subplots(
            2, 2, figsize=figsize, constrained_layout=True, **kwargs
        )
        ax_sig, ax_pos, ax_pnl, ax_cost = (
            axes[0, 0],
            axes[0, 1],
            axes[1, 0],
            axes[1, 1],
        )

        # --- Top-left: contract signals --------------------------------------
        cs_wide = QuantamentalDataFrame(self.cs_df).to_wide()
        csig_suffix = f"_CSIG_{self.sname}"
        sig_cols = [
            c
            for c in cs_wide.columns
            if c.endswith(csig_suffix)
            and any(c.startswith(f"{cid}_") for cid in plot_cids)
        ]
        sig_wide = _window(cs_wide[sig_cols]) if sig_cols else cs_wide.iloc[:0]

        if sig_wide.empty or sig_wide.dropna(how="all").empty:
            _annotate_empty(ax_sig, "No signal data in selected range")
        else:
            label_lines = len(sig_cols) <= 10
            for col in sig_cols:
                series = sig_wide[col].dropna()
                if series.empty:
                    continue
                label = col[: -len(csig_suffix)] if label_lines else None
                ax_sig.plot(series.index, series.values, lw=0.9, alpha=0.5, label=label)
            if not label_lines:
                ax_sig.text(
                    0.99,
                    0.97,
                    f"{len(sig_cols)} contracts overlaid",
                    transform=ax_sig.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    color="gray",
                )
            elif sig_cols:
                ax_sig.legend(fontsize="small", loc="best")
        ax_sig.set_title("Contract signals")
        ax_sig.set_ylabel("signal")
        ax_sig.grid(alpha=0.3)

        # --- Top-right: notional positions + cost-event markers --------------
        pos_cols = [
            c for c in pos_cols_all if any(c.startswith(f"{cid}_") for cid in plot_cids)
        ]
        pos_filtered = _window(pos_wide[pos_cols]) if pos_cols else pos_wide.iloc[:0]
        pos_title = "Notional positions"
        if mark_cost_events and tc_available:
            pos_title = "Notional positions with cost events"

        if pos_filtered.empty or pos_filtered.dropna(how="all").empty:
            _annotate_empty(ax_pos, "No position data in selected range")
        else:
            label_lines = len(pos_cols) <= 10
            for col in pos_cols:
                series = pos_filtered[col].dropna()
                if series.empty:
                    continue
                label = col[: -len(pos_suffix)] if label_lines else None
                ax_pos.plot(
                    series.index,
                    series.values,
                    drawstyle="steps-post",
                    lw=0.9,
                    alpha=0.5,
                    label=label,
                )

            # GLB gross exposure: the sum of absolute positions across every
            # contract in view. Drawn as a red dashed line to read against
            # the muted per-contract step lines underneath.
            gross = pos_filtered.abs().sum(axis=1, min_count=1).dropna()
            if not gross.empty:
                ax_pos.plot(
                    gross.index,
                    gross.values,
                    color=c_glb,
                    linestyle="--",
                    lw=1.6,
                    label="GLB gross exposure",
                )

            if mark_cost_events and tc_available:
                tc_wide = QuantamentalDataFrame(self.txn_costs_df).to_wide()
                bo_cols = [
                    c
                    for c in tc_wide.columns
                    if c.endswith("_BIDOFFER")
                    and any(c.startswith(f"{cid}_") for cid in plot_cids)
                ]
                rc_cols = [
                    c
                    for c in tc_wide.columns
                    if c.endswith("_ROLLCOST")
                    and any(c.startswith(f"{cid}_") for cid in plot_cids)
                ]

                def _scatter_events(cost_cols, suffix, **scatter_kwargs):
                    if not cost_cols:
                        return 0
                    sub = _window(tc_wide[cost_cols])
                    sub = sub.where(sub > 0)
                    if sub.dropna(how="all").empty:
                        return 0
                    max_val = float(np.nanmax(sub.values))
                    n_events = 0
                    for cost_col in cost_cols:
                        pos_col = cost_col[: -len(suffix)]
                        if pos_col not in pos_filtered.columns:
                            continue
                        charges = sub[cost_col].dropna()
                        if charges.empty:
                            continue
                        # Position level is taken from the contract's own step
                        # line, ffilled across any non-trading dates so that
                        # the marker sits on the line and not floating mid-axis.
                        pos_at_charge = (
                            pos_filtered[pos_col].reindex(charges.index).ffill()
                        )
                        valid = pos_at_charge.notna() & charges.notna()
                        if not valid.any():
                            continue
                        sizes = 20 + 200 * (charges[valid].values / max_val)
                        ax_pos.scatter(
                            charges.index[valid],
                            pos_at_charge[valid].values,
                            s=sizes,
                            **scatter_kwargs,
                        )
                        n_events += int(valid.sum())
                    return n_events

                n_bo = _scatter_events(
                    bo_cols,
                    "_TCOST_BIDOFFER",
                    marker="^",
                    color=c_bo,
                    alpha=0.7,
                    label=None,
                )
                n_rc = _scatter_events(
                    rc_cols,
                    "_TCOST_ROLLCOST",
                    marker="o",
                    facecolors="none",
                    edgecolors=c_rc,
                    lw=1.5,
                    label=None,
                )
                if n_bo > 0:
                    ax_pos.scatter(
                        [],
                        [],
                        marker="^",
                        color=c_bo,
                        alpha=0.7,
                        label=f"bid-offer events (n={n_bo})",
                    )
                if n_rc > 0:
                    ax_pos.scatter(
                        [],
                        [],
                        marker="o",
                        facecolors="none",
                        edgecolors=c_rc,
                        lw=1.5,
                        label=f"roll-cost events (n={n_rc})",
                    )

            if not label_lines:
                ax_pos.text(
                    0.99,
                    0.97,
                    f"{len(pos_cols)} contracts overlaid",
                    transform=ax_pos.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    color="gray",
                )
            ax_pos.legend(fontsize="small", loc="best")
            ax_pos.axhline(0, color="gray", lw=0.5, ls=":")
        ax_pos.set_title(pos_title)
        ax_pos.set_ylabel("position (USD mn)")
        ax_pos.grid(alpha=0.3)

        # --- Bottom-left: cumulative PnL excluding and including costs -------
        def _portfolio_series(qdf):
            sub = qdf.loc[qdf["cid"].astype(str) == self.portfolio_name]
            if sub.empty:
                return pd.Series(dtype=float)
            return (
                sub.sort_values("real_date")
                .set_index("real_date")["value"]
                .astype(float)
            )

        pnl_excl_series = _window(_portfolio_series(self.pnl_excl_costs))
        pnl_incl_series = _window(_portfolio_series(self.proxy_pnl)) if tc_available else pd.Series(dtype=float)

        if pnl_excl_series.empty and pnl_incl_series.empty:
            _annotate_empty(ax_pnl, "No PnL data in selected range")
        else:
            if cumsum:
                pnl_excl_plot = pnl_excl_series.cumsum()
                pnl_incl_plot = pnl_incl_series.cumsum() if not pnl_incl_series.empty else pnl_incl_series
                ax_pnl.set_title("Cumulative PnL")
            else:
                pnl_excl_plot = pnl_excl_series
                pnl_incl_plot = pnl_incl_series
                ax_pnl.set_title("Daily PnL")

            if not pnl_excl_plot.empty:
                total_excl = float(pnl_excl_series.sum())
                ax_pnl.plot(
                    pnl_excl_plot.index,
                    pnl_excl_plot.values,
                    color=c_pnle,
                    lw=1.6,
                    label=f"PnL ex-costs ({total_excl:.2f} USD mn)",
                )
            if not pnl_incl_plot.empty:
                total_incl = float(pnl_incl_series.sum())
                ax_pnl.plot(
                    pnl_incl_plot.index,
                    pnl_incl_plot.values,
                    color=c_pnl,
                    lw=1.6,
                    label=f"PnL incl-costs ({total_incl:.2f} USD mn)",
                )
                # Shade the cost-drag area: difference between ex- and
                # incl-costs paths over the common date range.
                common_idx = pnl_excl_plot.index.intersection(pnl_incl_plot.index)
                if len(common_idx) > 0:
                    total_drag = float(pnl_excl_series.sum() - pnl_incl_series.sum())
                    ax_pnl.fill_between(
                        common_idx,
                        pnl_incl_plot.loc[common_idx].values,
                        pnl_excl_plot.loc[common_idx].values,
                        color=c_rc,
                        alpha=0.15,
                        label=f"cost drag ({total_drag:.2f} USD mn)",
                    )
            ax_pnl.axhline(0, color="gray", lw=0.5, ls=":")
            ax_pnl.legend(fontsize="small", loc="best")
        ax_pnl.set_ylabel("USD mn")
        ax_pnl.grid(alpha=0.3)

        # --- Bottom-right: per-event transaction-cost bars -------------------
        if not tc_available:
            _annotate_empty(ax_cost, "No transaction costs available")
            ax_cost.set_title("Transaction costs")
        else:
            tc_wide = QuantamentalDataFrame(self.txn_costs_df).to_wide()
            bo_total_cols = [
                c
                for c in tc_wide.columns
                if c.endswith("_BIDOFFER")
                and not c.startswith(f"{self.portfolio_name}_")
            ]
            rc_total_cols = [
                c
                for c in tc_wide.columns
                if c.endswith("_ROLLCOST")
                and not c.startswith(f"{self.portfolio_name}_")
            ]
            bo_daily = (
                _window(tc_wide[bo_total_cols]).sum(axis=1)
                if bo_total_cols
                else pd.Series(dtype=float)
            )
            rc_daily = (
                _window(tc_wide[rc_total_cols]).sum(axis=1)
                if rc_total_cols
                else pd.Series(dtype=float)
            )
            bo_daily = bo_daily.loc[bo_daily.abs() > 0]
            rc_daily = rc_daily.loc[rc_daily.abs() > 0]

            charge_dates = bo_daily.index.union(rc_daily.index).sort_values()
            if len(charge_dates) == 0:
                _annotate_empty(ax_cost, "No transaction costs in selected range")
                ax_cost.set_title("Transaction costs")
            else:
                bo_at = bo_daily.reindex(charge_dates).fillna(0.0)
                rc_at = rc_daily.reindex(charge_dates).fillna(0.0)

                # Bar width on a date axis is in days. A width tied to the
                # median spacing between charge dates keeps sparse months
                # visible without overlapping dense ones.
                if len(charge_dates) > 1:
                    gaps = (
                        np.diff(charge_dates.values)
                        .astype("timedelta64[D]")
                        .astype(int)
                    )
                    bar_width = max(2, int(np.median(gaps)))
                else:
                    bar_width = 2

                total_bo = float(bo_at.sum())
                total_rc = float(rc_at.sum())
                ax_cost.bar(
                    charge_dates,
                    bo_at.values,
                    width=bar_width,
                    color=c_bo,
                    label=f"bid-offer ({total_bo:.2f} USD mn)",
                )
                ax_cost.bar(
                    charge_dates,
                    rc_at.values,
                    width=bar_width,
                    bottom=bo_at.values,
                    color=c_rc,
                    label=f"roll cost ({total_rc:.2f} USD mn)",
                )

                cum_total = (bo_at + rc_at).cumsum()
                ax_cost_twin = ax_cost.twinx()
                ax_cost_twin.plot(
                    cum_total.index,
                    cum_total.values,
                    color=c_cum,
                    linestyle="--",
                    lw=1.2,
                    label=f"cumulative total ({cum_total.iloc[-1]:.2f} USD mn)",
                )
                ax_cost_twin.set_ylabel("cumulative cost (USD mn)")

                handles_b, labels_b = ax_cost.get_legend_handles_labels()
                handles_t, labels_t = ax_cost_twin.get_legend_handles_labels()
                ax_cost.legend(
                    handles_b + handles_t,
                    labels_b + labels_t,
                    fontsize="small",
                    loc="upper left",
                )
                ax_cost.set_title("Transaction costs")
        ax_cost.set_ylabel("daily cost (USD mn)")
        ax_cost.grid(alpha=0.3)

        fig.suptitle(title, fontsize=13, fontweight="bold")
        plt.show()

        if return_fig:
            return fig
        return None

    def plot_pnl(self, title: str = "Proxy PnL", cumsum: bool = True, **kwargs):
        """
        Plot the proxy PnL DataFrame. The method uses the proxy PnL calculated in the
        previous step.

        Parameters
        ----------
        title : str, optional
            Title of the plot.
        cumsum : bool, optional
            Whether to plot the cumulative sum of the proxy PnL.
        kwargs
            Additional keyword arguments to be passed to the `timelines` function.
            See :func:`macrosynergy.visuals.timelines` for more information.
        """
        cdf = pd.concat((self.proxy_pnl, self.pnl_excl_costs), axis=0)
        rdf = reduce_df(cdf, cids=["GLB"])
        msv.timelines(rdf, title=title, cumsum=cumsum)


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_test_df

    cids_dmfx = ["CHF", "SEK", "NOK", "CAD", "GBP", "NZD", "JPY", "AUD"]
    fxblack = {"CHF": ("2011-10-03 00:00:00", "2015-01-30 00:00:00")}

    xcats = ["FX", "IRS", "CDS"]
    dfx = make_test_df(cids=cids_dmfx, xcats=xcats)
    txn_obj = TransactionCosts.download(verbose=True)

    p = ProxyPnL(
        df=dfx,
        transaction_costs_object=txn_obj,
        blacklist=fxblack,
        start="2001-01-01",
        end="2020-01-01",
        rstring="XR_NSA",
    )
    p.contract_signals(
        sig="CPIXFE_SJA_P6M6ML6ARvIETvBMZN",
        cids=cids_dmfx,
        ctypes=["FX"],
        cscales=["FXXRxLEV10_NSA"],
        relative_value=False,
        basket_contracts=["EUR_FX"],  # TODO invert asset class or returns?
        basket_weights=["FXXRxLEV10_NSA"],
        hedge_xcat="FXEURBETA",
    )
    p.notional_positions(
        aum=100,
        vol_target=10,
        rebal_freq="m",
        slip=1,
        est_freqs=["D", "W", "M"],
        est_weights=[1, 1, 1],
        lback_periods=[-1, -1, -1],
        lback_meth="xma",
        half_life=[11, 5, 6],
    )
    p.proxy_pnl_calc()
