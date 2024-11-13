import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numbers import Number
from typing import List, Union, Tuple, Optional, Dict, Any

from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import (
    reduce_df,
    standardise_dataframe,
    is_valid_iso_date,
    ticker_df_to_qdf,
    qdf_to_ticker_df,
)
from macrosynergy.management.types import QuantamentalDataFrame, NoneType
import macrosynergy.visuals as msv
from macrosynergy.pnl import (
    notional_positions,
    contract_signals,
    proxy_pnl_calc,
    TransactionCosts,
)


def _plot_strategy(
    df: QuantamentalDataFrame,
    sig: str,
    sname: str,
    pname: str,
    portfolio_name: str,
    pnl_name: str,
    tc_name: str,
    start: str,
    end: str,
):
    df_wide = qdf_to_ticker_df(df)
    df_wide = df_wide.loc[start:end]
    sel_tickers = [col for col in df_wide.columns]


class ProxyPnL(object):
    def __init__(
        self,
        df: QuantamentalDataFrame,
        transaction_costs_object: TransactionCosts,
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
        self.df = reduce_df(df=standardise_dataframe(df), blacklist=blacklist)
        self.start = start or df["real_date"].min().strftime("%Y-%m-%d")
        self.end = end or df["real_date"].max().strftime("%Y-%m-%d")
        self.rstring = rstring
        if not all(map(is_valid_iso_date, [self.start, self.end])):
            raise ValueError(f"Invalid date format: {self.start}, {self.end}")

        if not isinstance(transaction_costs_object, TransactionCosts):
            raise ValueError("Invalid transaction costs object.")
        else:
            transaction_costs_object.check_init()
            self.transaction_costs_object: TransactionCosts = transaction_costs_object

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
        est_weights: Union[Number, List[Number]] = [1, 2, 3],
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
        if df is None:
            if hasattr(self, "npos_df") and self.npos_df is not None:
                df = self.npos_df
            else:
                raise ValueError(
                    "Either pass a DataFrame with notional positions "
                    "or run `ProxyPnL.notional_positions` (and `contract_signals`) first."
                )
        spos: str = spos or self.sname + "_" + self.pname
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

    def plot_strategy(self):
        csigs = self.cs_df.columns
        poss = self.npos_df.columns

        msv.FacetPlot(df=pd.concat((self.cs_df, self.npos_df), axis=0)).plot(
            cols=csigs + poss, title="Contract Signals and Notional Positions"
        )


if __name__ == "__main__":
    import pickle, os

    # from macrosynergy.pnl import

    cids_dmfx = ["CHF", "SEK", "NOK", "CAD", "GBP", "NZD", "JPY", "AUD"]
    fxblack = {
        "CHF": (
            pd.Timestamp("2011-10-03 00:00:00"),
            pd.Timestamp("2015-01-30 00:00:00"),
        )
    }
    dfx = pd.read_pickle("data/dfx.pkl")

    if not os.path.exists("data/txn.obj.pkl"):
        txn = TransactionCosts()
        txn.download(verbose=True)
        with open("data/txn.obj.pkl", "wb") as f:
            pickle.dump(txn, f)

    with open("data/txn.obj.pkl", "rb") as f:
        txn_obj = pickle.load(f)

    pobjpath = "data/proxy_obj.pkl"
    if not os.path.exists(pobjpath):
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
        with open(pobjpath, "wb") as f:
            pickle.dump(p, f)

    with open(pobjpath, "rb") as f:
        p: ProxyPnL = pickle.load(f)

    p.proxy_pnl_calc()
    p.plot()
