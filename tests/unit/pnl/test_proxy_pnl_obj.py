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
from macrosynergy.pnl.transaction_costs import (
    TransactionCosts,
    TransactionCostsDictAdapter,
    get_fids,
)
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
            basket_contracts=["USD_FX", "EUR_FX"],
            basket_weights=[0.7, 0.3],
            sig="SIG",
            hedge_xcat="HR",
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
        xcats += [args["sig"]] + [args["hedge_xcat"]]
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

        # Align the date window with what ProxyPnL.proxy_pnl_calc uses internally
        # (its self.start / self.end captured at __init__ from the original df),
        # otherwise the functional path infers a different range from dfx because
        # notional_df extends one business day past the original df due to slip.
        expected_proxy_pnl_df = proxy_pnl_calc(
            df=dfx,
            transaction_costs_object=tco,
            start=proxy_pnl_obj.start,
            end=proxy_pnl_obj.end,
            **self.get_proxy_pnl_calc_args(),
        )

        pd.testing.assert_frame_equal(proxy_pnl_df, expected_proxy_pnl_df)

        # The three attributes assigned on the ProxyPnL instance must line up
        # with the functional proxy_pnl_calc return order (incl_costs, excl_costs,
        # tc_wide). A regression here would silently swap pnl_excl_costs and
        # txn_costs_df: the return value (self.proxy_pnl) would still be correct
        # but downstream consumers of the other two attributes would get the
        # wrong frame.
        expected_incl, expected_excl, expected_tc = proxy_pnl_calc(
            df=dfx,
            transaction_costs_object=tco,
            start=proxy_pnl_obj.start,
            end=proxy_pnl_obj.end,
            return_pnl_excl_costs=True,
            return_costs=True,
            **self.get_proxy_pnl_calc_args(),
        )
        pd.testing.assert_frame_equal(proxy_pnl_obj.proxy_pnl, expected_incl)
        pd.testing.assert_frame_equal(proxy_pnl_obj.pnl_excl_costs, expected_excl)
        pd.testing.assert_frame_equal(proxy_pnl_obj.txn_costs_df, expected_tc)

    def _build_cost_dict(self) -> dict:
        # Nested TransactionCostsDictAdapter schema: per fid, one entry per
        # cost type, each with size and cost anchors at median / pct90.
        def anchors(median, pct90):
            return {"median": median, "pct90": pct90}

        return {
            fid: {
                "bid_offer": {
                    "size": anchors(50.0, 200.0),
                    "cost": anchors(0.1, 0.5),
                },
                "rollcost": {
                    "size": anchors(50.0, 200.0),
                    "cost": anchors(0.05, 0.25),
                },
            }
            for fid in self.fids
        }

    def test_init_with_dict_adapter(self):
        adapter = TransactionCostsDictAdapter(
            cost_dict=self._build_cost_dict(), fids=self.fids
        )
        proxy_pnl = ProxyPnL(
            df=self.df,
            transaction_costs_object=adapter,
            sname=self.sname,
            pname=self.pname,
            rstring=self.rstring,
            portfolio_name=self.portfolio_name,
        )
        self.assertIs(proxy_pnl.transaction_costs_object, adapter)

    def test_proxy_pnl_calc_with_dict_adapter(self):
        cost_dict = self._build_cost_dict()
        adapter = TransactionCostsDictAdapter(cost_dict=cost_dict, fids=self.fids)

        cs_args = self.get_contract_signals_args()
        xcats = cs_args["xcats"].copy()
        xcats += [f"{xc}XR" for xc in xcats]
        xcats += [cs_args["sig"], cs_args["hedge_xcat"]]
        df = make_test_df(
            cids=cs_args["cids"],
            xcats=xcats,
            start=self.df["real_date"].min().strftime("%Y-%m-%d"),
            end=self.df["real_date"].max().strftime("%Y-%m-%d"),
        )

        proxy_pnl_obj = ProxyPnL(
            df=df,
            transaction_costs_object=adapter,
            sname=self.sname,
            pname=self.pname,
            rstring=self.rstring,
            portfolio_name=self.portfolio_name,
        )
        cs_df = proxy_pnl_obj.contract_signals(**cs_args)
        notional_df = proxy_pnl_obj.notional_positions(
            **self.get_notional_positions_args()
        )
        proxy_pnl_df = proxy_pnl_obj.proxy_pnl_calc(**self.get_proxy_pnl_calc_args())

        dfx = pd.concat([df, cs_df, notional_df], axis=0)
        # See test_flow for the start/end alignment rationale: the functional
        # path otherwise infers a wider window from dfx than ProxyPnL uses
        # internally.
        expected_proxy_pnl_df = proxy_pnl_calc(
            df=dfx,
            transaction_costs_object=adapter,
            start=proxy_pnl_obj.start,
            end=proxy_pnl_obj.end,
            **self.get_proxy_pnl_calc_args(),
        )
        pd.testing.assert_frame_equal(proxy_pnl_df, expected_proxy_pnl_df)


class TestEvaluatePnL(unittest.TestCase):
    """Tests for ProxyPnL.evaluate_pnl."""

    portfolio_name = "GLB"
    sname = "STRAT"
    pname = "POS"
    rstring = "XR"

    def _make_obj(self) -> ProxyPnL:
        init_df = make_test_df(
            cids=[self.portfolio_name],
            xcats=["DUMMY"],
            start="2018-01-01",
            end="2024-12-31",
        )
        return ProxyPnL(
            df=init_df,
            sname=self.sname,
            pname=self.pname,
            rstring=self.rstring,
            portfolio_name=self.portfolio_name,
        )

    def _make_pnl_qdf(
        self,
        xcat: str,
        values: np.ndarray,
        cid: str = None,
        start: str = "2020-01-01",
    ) -> QuantamentalDataFrame:
        cid = cid or self.portfolio_name
        dates = pd.bdate_range(start=start, periods=len(values))
        return QuantamentalDataFrame(
            pd.DataFrame(
                {
                    "real_date": dates,
                    "cid": cid,
                    "xcat": xcat,
                    "value": np.asarray(values, dtype=float),
                }
            )
        )

    def test_invalid_argument_types(self):
        obj = self._make_obj()
        with self.assertRaisesRegex(TypeError, "Argument aum"):
            obj.evaluate_pnl(aum="100")

        with self.assertRaisesRegex(TypeError, "Argument include_pnle"):
            obj.evaluate_pnl(aum=100, include_pnle="True")

        with self.assertRaisesRegex(TypeError, "Argument include_tcosts"):
            obj.evaluate_pnl(aum=100, include_tcosts="False")

        with self.assertRaisesRegex(TypeError, "Argument label_dict"):
            obj.evaluate_pnl(aum=100, label_dict=[])

        with self.assertRaisesRegex(TypeError, "Argument start"):
            obj.evaluate_pnl(aum=100, start=123)

        with self.assertRaisesRegex(TypeError, "Argument end"):
            obj.evaluate_pnl(aum=100, end=567)

        with self.assertRaisesRegex(TypeError, "Argument benchmark_data"):
            obj.evaluate_pnl(aum=100, benchmark_data=np.ndarray([]))

    def test_raises_when_proxy_pnl_missing(self):
        obj = self._make_obj()
        with self.assertRaisesRegex(ValueError, "self.proxy_pnl is missing"):
            obj.evaluate_pnl(aum=100)

    def test_raises_when_pnle_required_but_missing(self):
        obj = self._make_obj()
        obj.proxy_pnl = self._make_pnl_qdf("PNL", np.ones(252))
        with self.assertRaisesRegex(ValueError, "self.pnl_excl_costs is missing"):
            obj.evaluate_pnl(aum=100, include_pnle=True)

    def test_raises_when_tcosts_required_but_missing(self):
        obj = self._make_obj()
        obj.proxy_pnl = self._make_pnl_qdf("PNL", np.ones(252))
        with self.assertRaisesRegex(ValueError, "self.txn_costs_df is missing"):
            obj.evaluate_pnl(aum=100, include_tcosts=True)

    def test_output_columns_match_input_xcat(self):
        obj = self._make_obj()
        obj.proxy_pnl = self._make_pnl_qdf("PNL", np.arange(252, dtype=float))
        out = obj.evaluate_pnl(aum=100)
        self.assertEqual(list(out.columns), ["PNL"])

    def test_output_index_has_expected_summary_rows(self):
        obj = self._make_obj()
        obj.proxy_pnl = self._make_pnl_qdf("PNL", np.arange(252, dtype=float))
        out = obj.evaluate_pnl(aum=100)
        expected_rows = {
            "Return %",
            "St. Dev. %",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Sharpe Stability",
            "Max 21-Day Draw %",
            "Max 6-Month Draw %",
            "Peak to Trough Draw %",
            "Top 5% Monthly PnL Share",
            "Traded Months",
        }
        self.assertTrue(expected_rows.issubset(set(out.index)))

    def test_aum_scales_return_inversely(self):
        n_days = 252
        obj = self._make_obj()
        obj.proxy_pnl = self._make_pnl_qdf("PNL", 0.01 * np.ones(n_days))

        out_100 = obj.evaluate_pnl(aum=100)
        out_200 = obj.evaluate_pnl(aum=200)

        self.assertAlmostEqual(out_100.loc["Return %", "PNL"], 2.52)
        self.assertAlmostEqual(out_200.loc["Return %", "PNL"], 2.52 / 2)
        self.assertAlmostEqual(out_100.loc["St. Dev. %", "PNL"], 0.0)
        self.assertAlmostEqual(out_200.loc["St. Dev. %", "PNL"], 0.0)

    def test_mean_zero_alternating_series(self):
        # Alternating +/-1 PnL with aum=100 => daily returns +/-1%.
        # daily mean = 0, sample std = 1 * sqrt(n / (n - 1)).
        n_days = 600
        values = np.tile([1.0, -1.0], n_days // 2)
        obj = self._make_obj()
        obj.proxy_pnl = self._make_pnl_qdf("PNL", values)

        out = obj.evaluate_pnl(aum=100)

        self.assertAlmostEqual(out.loc["Return %", "PNL"], 0.0)
        expected_std = 1 * np.sqrt(n_days / (n_days - 1)) * np.sqrt(252)
        self.assertAlmostEqual(out.loc["St. Dev. %", "PNL"], expected_std)
        self.assertAlmostEqual(out.loc["Sharpe Ratio", "PNL"], 0.0)


    def test_include_pnle_adds_pnle_column(self):
        obj = self._make_obj()
        obj.proxy_pnl = self._make_pnl_qdf("PNL", 0.01 * np.ones(252))
        obj.pnl_excl_costs = self._make_pnl_qdf("PNLe", np.full(252, 0.02))

        out = obj.evaluate_pnl(aum=100, include_pnle=True)

        self.assertEqual(sorted(out.columns.tolist()), ["PNL", "PNLe"])
        self.assertAlmostEqual(out.loc["Return %", "PNL"], 2.52)
        self.assertAlmostEqual(out.loc["Return %", "PNLe"], 2.52 * 2)


    def test_include_tcosts_adds_transaction_cost_row(self):
        obj = self._make_obj()
        obj.proxy_pnl = self._make_pnl_qdf("PNL", np.ones(252))
        obj.txn_costs_df = self._make_pnl_qdf("TCOST", np.full(252, 0.5))

        out = obj.evaluate_pnl(aum=100, include_tcosts=True)

        self.assertIn("Transaction Cost", out.index)
        self.assertAlmostEqual(out.loc["Transaction Cost", "PNL"], 252 * 0.5)

    def test_include_tcosts_zero_for_pnle_column(self):
        obj = self._make_obj()
        obj.proxy_pnl = self._make_pnl_qdf("PNL", np.ones(252))
        obj.pnl_excl_costs = self._make_pnl_qdf("PNLe", np.full(252, 2.0))
        obj.txn_costs_df = self._make_pnl_qdf("TCOST", np.full(252, 0.5))

        out = obj.evaluate_pnl(aum=100, include_pnle=True, include_tcosts=True)

        self.assertAlmostEqual(out.loc["Transaction Cost", "PNL"], 252 * 0.5)
        self.assertAlmostEqual(out.loc["Transaction Cost", "PNLe"], 0.0)

    def test_label_dict_renames_columns(self):
        obj = self._make_obj()
        obj.proxy_pnl = self._make_pnl_qdf("PNL", np.ones(252))
        out = obj.evaluate_pnl(aum=100, label_dict={"PNL": "Strategy A"})
        self.assertEqual(list(out.columns), ["Strategy A"])

    def test_end_filters_dates(self):
        n_days = 252
        obj = self._make_obj()
        obj.proxy_pnl = self._make_pnl_qdf(
            "PNL", np.arange(n_days, dtype=float), start="2020-01-01"
        )

        end_short = pd.bdate_range("2020-01-01", periods=21)[-1].strftime("%Y-%m-%d")

        full = obj.evaluate_pnl(aum=100)
        clipped = obj.evaluate_pnl(aum=100, end=end_short)

        self.assertGreater(
            full.loc["Traded Months", "PNL"],
            clipped.loc["Traded Months", "PNL"],
        )

    def test_benchmark_data_adds_correlation_row(self):
        n_days = 252
        values = np.arange(n_days, dtype=float)
        obj = self._make_obj()
        obj.proxy_pnl = self._make_pnl_qdf("PNL", values)

        bm = pd.DataFrame(
            {
                "real_date": pd.bdate_range("2020-01-01", periods=n_days),
                "cid": "USD",
                "xcat": "EQ",
                "value": values,
            }
        )

        out = obj.evaluate_pnl(aum=100, benchmark_data=bm)

        self.assertIn("USD_EQ correl", out.index)
        self.assertAlmostEqual(out.loc["USD_EQ correl", "PNL"], 1.0)

    def test_benchmark_data_not_mutated(self):
        n_days = 100
        obj = self._make_obj()
        obj.proxy_pnl = self._make_pnl_qdf("PNL", np.arange(n_days, dtype=float))

        bm = pd.DataFrame(
            {
                "real_date": pd.bdate_range("2020-01-01", periods=n_days),
                "cid": "USD",
                "xcat": "EQ",
                "value": np.arange(n_days, dtype=float),
            }
        )
        original_cols = list(bm.columns)

        obj.evaluate_pnl(aum=100, benchmark_data=bm)

        self.assertEqual(list(bm.columns), original_cols)

    def test_end_to_end(self):
        cids = ["USD", "EUR", "JPY", "GBP"]
        ctypes = ["FX"]
        fids = [f"{c}_{x}" for c in cids for x in ctypes]
        xcats = ctypes + [f"{xc}XR" for xc in ctypes] + ["SIG"]

        df = make_test_df(
            cids=cids,
            xcats=xcats,
            start="2018-01-01",
            end="2023-12-31",
        )

        cost_dict = {
            fid: {
                "bid_offer": {"size": {"median": 50, "pct90": 200}, "cost": {"median": 0.1, "pct90": 0.3}},
                "rollcost": {"size": {"median": 50, "pct90": 200}, "cost": {"median": 0.1, "pct90": 0.3}},
            }
            for fid in fids
        }
        adapter = TransactionCostsDictAdapter(cost_dict=cost_dict, fids=fids)

        obj = ProxyPnL(
            df=df,
            transaction_costs_object=adapter,
            sname=self.sname,
            pname=self.pname,
            rstring="XR",
            portfolio_name=self.portfolio_name,
        )
        obj.contract_signals(sig="SIG", cids=cids, ctypes=ctypes)
        obj.notional_positions(fids=fids, aum=1000, leverage=1.0, lback_meth="xma")
        obj.proxy_pnl_calc()

        out = obj.evaluate_pnl(aum=1000, include_pnle=True, include_tcosts=True)

        self.assertIn("Return %", out.index)
        self.assertIn("Transaction Cost", out.index)
        self.assertGreaterEqual(len(out.columns), 2)
        self.assertFalse(out.isna().all().any())


if __name__ == "__main__":
    unittest.main()
