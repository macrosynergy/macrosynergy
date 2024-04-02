import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Union, Tuple, Optional

from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import (
    reduce_df,
    standardise_dataframe,
    is_valid_iso_date,
)
from macrosynergy.management.types import Numeric, QuantamentalDataFrame
from macrosynergy.pnl import notional_positions, contract_signals, proxy_pnl_calc
from macrosynergy.download.transaction_costs import download_transaction_costs


class ProxyPnL(object):
    def __init__(
        self,
        df: QuantamentalDataFrame,
        start: Optional[str] = None,
        end: Optional[str] = None,
        blacklist: Optional[dict] = None,
        sname: str = "STRAT",
        pname: str = "POS",
        # TODO roll costs
        # TODO bid-ask spread
        # TODO size
        # TODO slippage? In notional?
    ):
        self.sname = sname
        self.pname = pname
        self.blacklist = blacklist
        self.cs_df = None
        self.npos_df = None
        self.df = reduce_df(df=standardise_dataframe(df), blacklist=blacklist)
        self.start = start or df["real_date"].min().strftime("%Y-%m-%d")
        self.end = end or df["real_date"].max().strftime("%Y-%m-%d")
        if not all(map(is_valid_iso_date, [self.start, self.end])):
            raise ValueError(f"Invalid date format: {self.start}, {self.end}")

    def contract_signals(
        self,
        sig: str,
        cids: List[str],
        ctypes: List[str],
        cscales: Optional[List[Union[Numeric, str]]] = None,
        csigns: Optional[List[int]] = None,
        hbasket: Optional[List[str]] = None,
        hscales: Optional[List[Union[Numeric, str]]] = None,
        hratios: Optional[str] = None,
        *args,
        **kwargs,
    ) -> QuantamentalDataFrame:
        cs_args = dict(
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
            blacklist=self.blacklist,
            sname=self.sname,
        )
        cs_args.update(kwargs)
        self.fids = [f"{cid}_{ctype}" for cid in cids for ctype in ctypes]

        cs_df: QuantamentalDataFrame = contract_signals(**cs_args)
        self.cs_df = cs_df
        return cs_df

    def notional_positions(
        self,
        df: QuantamentalDataFrame = None,
        sname: str = None,
        fids: List[str] = None,
        aum: Numeric = 100,
        dollar_per_signal: Numeric = 1.0,
        leverage: Optional[Numeric] = None,
        vol_target: Optional[Numeric] = None,
        rebal_freq: str = "m",
        slip: int = 1,
        lback_periods: int = 21,
        lback_meth: str = "ma",
        half_life=11,
        rstring: str = "XR",
        start: Optional[str] = None,
        end: Optional[str] = None,
        blacklist: Optional[dict] = None,
        pname: str = "POS",
    ):
        fids = fids or self.fids
        df = df or self.cs_df or self.contract_signals()
        np_args = dict(
            df=df,
            sname=sname or self.sname,
            fids=fids,
            aum=aum,
            dollar_per_signal=dollar_per_signal,
            leverage=leverage,
            vol_target=vol_target,
            rebal_freq=rebal_freq,
            slip=slip,
            lback_periods=lback_periods,
            lback_meth=lback_meth,
            half_life=half_life,
            rstring=rstring,
            start=start or self.start,
            end=end or self.end,
            blacklist=blacklist or self.blacklist,
            pname=pname,
        )
        np_df: QuantamentalDataFrame = notional_positions(**np_args)
        self.npos_df = np_df
        return np_df

    def proxy_pnl_calc(
        self,
        spos: str,
        df: QuantamentalDataFrame = None,
        fids: List[str] = None,
        tcost_n: Optional[str] = None,
        rcost_n: Optional[str] = None,
        size_n: Optional[str] = None,
        tcost_l: Optional[str] = None,
        rcost_l: Optional[str] = None,
        size_l: Optional[str] = None,
        roll_freqs: Optional[dict] = None,
    ):
        df = df or self.npos_df or self.notional_positions()
        spos = spos or self.sname + "_" + self.pname
        fids = fids or self.fids
        pp_args = dict(
            df=df,
            spos=spos,
            fids=fids,
            tcost_n=tcost_n,
            rcost_n=rcost_n,
            size_n=size_n,
            tcost_l=tcost_l,
            rcost_l=rcost_l,
            size_l=size_l,
            roll_freqs=roll_freqs,
            start=self.start,
            end=self.end,
            blacklist=self.blacklist,
        )

        return proxy_pnl_calc(**pp_args)
