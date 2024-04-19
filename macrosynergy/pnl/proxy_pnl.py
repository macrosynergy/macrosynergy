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
)
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.pnl import (
    notional_positions,
    contract_signals,
    proxy_pnl_calc,
    TransactionCosts,
)


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
        cscales: Optional[List[Union[Number, str]]] = None,
        csigns: Optional[List[int]] = None,
        hbasket: Optional[List[str]] = None,
        hscales: Optional[List[Union[Number, str]]] = None,
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
        rstring: str = "XR",
        start: Optional[str] = None,
        end: Optional[str] = None,
        blacklist: Optional[dict] = None,
        pname: str = "POS",
        return_pvol: bool = False,
        return_vcv: bool = False,
    ) -> Union[
        QuantamentalDataFrame,
        Tuple[QuantamentalDataFrame, QuantamentalDataFrame],
        Tuple[QuantamentalDataFrame, pd.DataFrame],
        Tuple[QuantamentalDataFrame, QuantamentalDataFrame, pd.DataFrame],
    ]:
        fids = fids or self.fids
        df = df or self.cs_df or self.contract_signals()
        sname = sname or self.sname
        start = start or self.start
        end = end or self.end
        blacklist = blacklist or self.blacklist
        np_args = dict(
            df=df,
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
            return_pvol=return_pvol,
            return_vcv=return_vcv,
        )
        outs: Union[QuantamentalDataFrame, Tuple[QuantamentalDataFrame, ...]] = (
            notional_positions(**np_args)
        )
        assert isinstance(outs, (tuple, QuantamentalDataFrame))
        if isinstance(outs, QuantamentalDataFrame):
            outs = (outs,)
        assert len(outs) in [1, 2, 3]
        assert isinstance(outs[0], QuantamentalDataFrame)

        np_df: QuantamentalDataFrame = outs[0]
        if return_pvol:
            pvol_df = outs[1]
            self.pvol_df: QuantamentalDataFrame = pvol_df
        if return_vcv:
            vcv_df = outs[int(return_pvol) + 1]
            self.vcv_df: QuantamentalDataFrame = vcv_df

        self.npos_df: QuantamentalDataFrame = np_df
        return np_df

    def proxy_pnl_calc(
        self,
        spos: str = None,
        df: QuantamentalDataFrame = None,
        fids: List[str] = None,
        tcost_n: Optional[str] = None,
        rcost_n: Optional[str] = None,
        size_n: Optional[str] = None,
        tcost_l: Optional[str] = None,
        rcost_l: Optional[str] = None,
        size_l: Optional[str] = None,
        roll_freqs: Optional[dict] = None,
        pnl_name: str = "PNL",
        tc_name: str = "TCOST",
        return_pnl_excl_costs: bool = False,
        return_costs: bool = False,
    ) -> Union[
        QuantamentalDataFrame,
        Tuple[QuantamentalDataFrame, pd.DataFrame],
        Tuple[QuantamentalDataFrame, pd.DataFrame, pd.DataFrame],
    ]:
        df: QuantamentalDataFrame = df or self.npos_df or self.notional_positions()
        spos: str = spos or self.sname + "_" + self.pname
        fids: List[str] = fids or self.fids
        txn_obj_args = dict(
            tcost_n=tcost_n,
            rcost_n=rcost_n,
            size_n=size_n,
            tcost_l=tcost_l,
            rcost_l=rcost_l,
            size_l=size_l,
        )
        txn_obj_args = {
            k: v if v is not None else TransactionCosts.DEFAULT_ARGS[k]
            for k, v in txn_obj_args.items()
        }

        pp_args: Dict[str, Union[QuantamentalDataFrame, str, List[str], dict]] = dict(
            df=df,
            spos=spos,
            fids=fids,
            tcost_n=tcost_n,
            rcost_n=rcost_n,
            size_n=size_n,
            tcost_l=tcost_l,
            rcost_l=rcost_l,
            size_l=size_l,  # TODO what happens if None?
            roll_freqs=roll_freqs,
            start=self.start,
            end=self.end,
            blacklist=self.blacklist,
            pnl_name=pnl_name,
            tc_name=tc_name,
            return_pnl_excl_costs=return_pnl_excl_costs,
            return_costs=return_costs,
        )
        outs = proxy_pnl_calc(**pp_args)
        assert isinstance(outs, (tuple, QuantamentalDataFrame))
        if isinstance(outs, QuantamentalDataFrame):
            outs = (outs,)
        assert len(outs) in [1, 2, 3]
        assert isinstance(outs[0], QuantamentalDataFrame)

        proxy_pnl: QuantamentalDataFrame = outs[0]
        if return_pnl_excl_costs:
            pnl_excl_costs = outs[1]
            self.pnl_excl_costs: QuantamentalDataFrame = pnl_excl_costs
        if return_costs:
            txn_costs_df = outs[int(return_pnl_excl_costs) + 1]
            self.txn_costs_df: pd.DataFrame = txn_costs_df

        self.proxy_pnl: QuantamentalDataFrame = proxy_pnl

        return proxy_pnl
