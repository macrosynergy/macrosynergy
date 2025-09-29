"""
Implementation of the ProxyPnL class.
"""

import pandas as pd
from numbers import Number
from typing import List, Union, Tuple, Optional

from macrosynergy.management.utils import (
    reduce_df,
    is_valid_iso_date,
)
from macrosynergy.management.types import QuantamentalDataFrame, NoneType
import macrosynergy.visuals as msv
from macrosynergy.pnl import (
    notional_positions,
    contract_signals,
    proxy_pnl_calc,
    TransactionCosts,
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
    transaction_costs_object : TransactionCosts
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
        transaction_costs_object: Optional[TransactionCosts],
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
        else:
            raise ValueError(
                "Invalid type for `transaction_costs_object`. Expected `TransactionCosts` object."
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
        hbasket: Optional[List[str]] = None,
        hscales: Optional[List[Union[Number, str]]] = None,
        hratios: Optional[str] = None,
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
            hbasket=hbasket,
            hscales=hscales,
            hratios=hratios,
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
        roll_freqs: Optional[dict] = None,
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
            roll_freqs=roll_freqs,
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
        self.txn_costs_df: QuantamentalDataFrame = outs[1]
        self.pnl_excl_costs: QuantamentalDataFrame = outs[2]
        outs = None

        return self.proxy_pnl

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
        hbasket=["EUR_FX"],  # TODO invert asset class or returns?
        hscales=["FXXRxLEV10_NSA"],
        hratios="FXEURBETA",
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
