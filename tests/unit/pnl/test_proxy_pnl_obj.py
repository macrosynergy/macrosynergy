import unittest
import numpy as np
import pandas as pd
from typing import List
from macrosynergy.management.utils import _map_to_business_day_frequency, get_cid
from macrosynergy.pnl import (
    notional_positions,
    contract_signals,
    proxy_pnl_calc,
    ProxyPnL,
)
from macrosynergy.pnl.transaction_costs import TransactionCosts, get_fids
from macrosynergy.download.transaction_costs import AVAILABLE_CATS, AVAILABLE_CTYPES
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.simulate import make_test_df, simulate_returns_and_signals


def make_tx_cost_df(
    cids: List[str] = None,
    tickers: List[str] = None,
    start="2020-01-01",
    end="2025-01-01",
) -> pd.DataFrame:
    err = "Either cids or tickers must be provided (not both)"
    assert bool(cids) or bool(tickers), err
    assert bool(cids) ^ bool(tickers), err

    if cids is None:
        tiks = tickers
    else:
        tiks = [f"{c}_{k}" for c in cids for k in AVAILABLE_CATS]

    freq = _map_to_business_day_frequency("M")
    date_range = pd.bdate_range(start=start, end=end, freq=freq)

    val_dict = {
        "BIDOFFER_MEDIAN": (0.1, 0.2),
        "BIDOFFER_90PCTL": (0.5, 2),
        "ROLLCOST_MEDIAN": (0.001, 0.006),
        "ROLLCOST_90PCTL": (0.007, 0.01),
        "SIZE_MEDIAN": (10, 20),
        "SIZE_90PCTL": (50, 70),
    }

    ct_map = {
        "FX": (10, 20),
        "IRS": (100, 150),
        "CDS": (1000, 1500),
    }

    out_dict = dict()
    u_cids = sorted(set(map(get_cid, tiks)))
    for cid in u_cids:
        for fid_type, (rn, rx) in ct_map.items():
            for cost_type, (mn, mx) in val_dict.items():
                ra = np.random.uniform(mx, mn, len(date_range))
                rb = np.random.uniform(rx, rn, len(date_range))
                name = f"{cid}_{fid_type}{cost_type}"
                data = ra * rb
                s = pd.Series(data, index=date_range, name=name)
                if name in out_dict:
                    raise ValueError(f"Duplicate name: {name}")
                out_dict[name] = s

    df = pd.concat(out_dict.values(), axis=1)

    # forward will this to complete for every day
    new_index = pd.bdate_range(start=start, end=end, freq="B")
    df = df.reindex(new_index).ffill().bfill()
    df.index.name = "real_date"

    return QuantamentalDataFrame.from_wide(df)


class TestProxyPNLObject(unittest.TestCase):
    def setUp(self):
        self.cids = ["USD", "EUR", "JPY", "GBP", "AUD", "CAD"]
        self.xcat = "EQ"
        n_years = 5

        self.tc_df = make_tx_cost_df(cids=self.cids)
        self.fids = get_fids(self.tc_df)
        self.sname = "STRAT"
        self.pname = "POS"
        self.rstring = "XR"
        self.portfolio_name = "GLB"
        self.ctypes = AVAILABLE_CTYPES
        self.df = pd.concat(
            simulate_returns_and_signals(
                cids=self.cids,
                xcat=_xc,
                return_suffix="XR",
                signal_suffix="CSIG_STRAT",
                years=n_years,
                end="2025-01-01",
            )
            for _xc in self.ctypes
        )

    def get_proxy_pnl_args(self):
        return dict(
            df=self.df,
            transaction_costs_object=TransactionCosts(self.tc_df, fids=self.fids),
            sname=self.sname,
            pname=self.pname,
            rstring=self.rstring,
            portfolio_name=self.portfolio_name,
        )

    def get_contract_signals_args(self):
        return dict(
            cids=self.cids,
            xcats=self.ctypes,
            ctypes=self.ctypes,
            cscales=[1.0, 0.5, 0.1],
            csigns=[1, -1, 1],
            hbasket=["USD_FX", "EUR_FX"],
            hscales=[0.7, 0.3],
            sig="SIG",
            hratios="HR",
        )

    def get_notional_positions_args(self):
        return dict(
            fids=self.fids,
            leverage=1.1,
            sname=self.sname,
            aum=1000,
            lback_meth="xma",
        )

    def get_proxy_pnl_calc_args(self):
        return dict(
            spos=self.sname + "_" + self.pname,
            portfolio_name=self.portfolio_name,
            rstring=self.rstring,
            pnl_name="PNL",
            tc_name="TCOST",
        )

    def test_init(self):
        base_args = self.get_proxy_pnl_args()
        proxy_pnl = ProxyPnL(**base_args)

        for k, v in base_args.items():
            in_val = getattr(proxy_pnl, k)
            if isinstance(v, pd.DataFrame):
                result = (
                    QuantamentalDataFrame(in_val)
                    .sort_values(by=QuantamentalDataFrame.IndexColsSortOrder)
                    .reset_index(drop=True)
                    .eq(
                        QuantamentalDataFrame(v)
                        .sort_values(by=QuantamentalDataFrame.IndexColsSortOrder)
                        .reset_index(drop=True)
                    )
                    .all()
                    .all()
                )
                self.assertTrue(result)
            else:
                self.assertEqual(in_val, v)

    def test_flow(self):

        args = self.get_contract_signals_args()
        xcats = args["xcats"].copy()
        xcats += [f"{xc}XR" for xc in xcats]
        xcats += [args["sig"]] + [args["hratios"]]
        df = make_test_df(
            cids=args["cids"],
            xcats=xcats,
            start=self.df["real_date"].min().strftime("%Y-%m-%d"),
            end=self.df["real_date"].max().strftime("%Y-%m-%d"),
        )

        expected_cs_df = contract_signals(
            df=df,
            **args,
        )
        proxy_pnl_args = self.get_proxy_pnl_args()
        proxy_pnl_args["df"] = df
        proxy_pnl_obj = ProxyPnL(**proxy_pnl_args)
        cs_df = proxy_pnl_obj.contract_signals(**args)

        pd.testing.assert_frame_equal(cs_df, expected_cs_df)

        # testing notional positions
        expected_notional_df = notional_positions(
            df=cs_df,
            **self.get_notional_positions_args(),
        )

        notional_df = proxy_pnl_obj.notional_positions(
            **self.get_notional_positions_args()
        )

        pd.testing.assert_frame_equal(notional_df, expected_notional_df)

        # testing proxy_pnl_calc
        tco = TransactionCosts(self.tc_df, fids=self.fids)

        dfx = pd.concat([df, cs_df, notional_df], axis=0)

        proxy_pnl_df = proxy_pnl_obj.proxy_pnl_calc(**self.get_proxy_pnl_calc_args())

        expected_proxy_pnl_df = proxy_pnl_calc(
            df=dfx, transaction_costs_object=tco, **self.get_proxy_pnl_calc_args()
        )

        pd.testing.assert_frame_equal(proxy_pnl_df, expected_proxy_pnl_df)


if __name__ == "__main__":
    unittest.main()
