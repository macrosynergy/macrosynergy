"""
Implementation of the ProxyPnL class.
"""
import numpy as np
import pandas as pd
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
