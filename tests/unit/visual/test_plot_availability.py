import unittest
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch

from macrosynergy.visuals.view_availability import view_availability


def _make_binary_qdf(
    start: str = "2020-01-01",
    end: str = "2022-12-31",
    tickers: Dict[Tuple[str, str], str] = None,
) -> pd.DataFrame:
    """Build a binary-availability QDF for testing.

    ``tickers`` maps (cid, xcat) pairs to the date from which that ticker is
    available (value=1 from that date onward, 0 before).
    """
    idx = pd.date_range(start, end, freq="D")
    if tickers is None:
        tickers = {
            ("USD", "A"): "2020-01-01",
            ("USD", "B"): "2021-01-01",
            ("USD", "C"): "2022-01-01",
        }
    frames = []
    for (cid, xcat), from_date in tickers.items():
        values = (idx >= pd.Timestamp(from_date)).astype(int)
        frames.append(
            pd.DataFrame(
                {
                    "real_date": idx,
                    "cid": cid,
                    "xcat": xcat,
                    "value": values,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class TestPlotAvailability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")
        cls.mock_show = patch("matplotlib.pyplot.show").start()

        cls.df = _make_binary_qdf()

    @classmethod
    def tearDownClass(cls) -> None:
        plt.close("all")
        patch.stopall()
        matplotlib.use(cls.mpl_backend)

    def setUp(self):
        self.valid_args: Dict[str, Any] = {
            "df": self.df,
            "title": "Test Availability",
            "n_ticks": 5,
            "fig_kw": None,
            "heatmap_kw": None,
            "xticklabel_kw": None,
            "yticklabel_kw": None,
            "title_kw": None,
            "return_fig": False,
        }

    def tearDown(self) -> None:
        plt.close("all")

    # ------------------------------------------------------------------
    # Happy-path
    # ------------------------------------------------------------------

    def test_no_error_default_args(self):
        try:
            view_availability(self.df)
        except Exception as e:
            self.fail(f"view_availability raised {e} unexpectedly")

    def test_no_error_all_valid_args(self):
        try:
            view_availability(**self.valid_args)
        except Exception as e:
            self.fail(f"view_availability raised {e} unexpectedly")

    def test_return_fig_returns_figure(self):
        args = self.valid_args.copy()
        args["return_fig"] = True
        result = view_availability(**args)
        self.assertIsInstance(result, plt.Figure)

    def test_return_fig_false_returns_none(self):
        args = self.valid_args.copy()
        args["return_fig"] = False
        result = view_availability(**args)
        self.assertIsNone(result)

    def test_kwargs_dicts_are_applied(self):
        args = self.valid_args.copy()
        args["fig_kw"] = {"figsize": (20, 6)}
        args["heatmap_kw"] = {"linewidths": 0.5}
        args["xticklabel_kw"] = {"rotation": 30, "fontsize": 8}
        args["yticklabel_kw"] = {"fontsize": 10}
        args["title_kw"] = {"fontsize": 14, "pad": 15}
        try:
            view_availability(**args)
        except Exception as e:
            self.fail(f"view_availability raised {e} with valid kwargs dicts")

    def test_n_ticks_respected(self):
        for n in [1, 5, 20]:
            args = self.valid_args.copy()
            args["n_ticks"] = n
            args["return_fig"] = True
            fig = view_availability(**args)
            ax = fig.axes[0]
            self.assertLessEqual(len(ax.get_xticklabels()), n + 1)

    def test_column_sort_order(self):
        # A available throughout; B available only in 2022; C never.
        df = _make_binary_qdf(
            start="2020-01-01",
            end="2022-12-31",
            tickers={
                ("USD", "A"): "2020-01-01",
                ("USD", "B"): "2022-01-01",
                ("USD", "C"): "2099-01-01",  # never available within range
            },
        )
        args = self.valid_args.copy()
        args["df"] = df
        args["return_fig"] = True
        fig = view_availability(**args)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        a_pos = labels.index("USD_A")
        b_pos = labels.index("USD_B")
        c_pos = labels.index("USD_C")
        # A has more observations than B; both precede C (zero count)
        self.assertLess(a_pos, c_pos)
        self.assertLess(b_pos, c_pos)

    def test_single_ticker(self):
        df = _make_binary_qdf(tickers={("USD", "OnlyCol"): "2020-06-01"})
        try:
            view_availability(df)
        except Exception as e:
            self.fail(f"view_availability raised {e} for single-ticker QDF")

    def test_all_zeros_ticker(self):
        df = _make_binary_qdf(
            start="2020-01-01",
            end="2021-12-31",
            tickers={
                ("USD", "Active"): "2020-01-01",
                ("USD", "Inactive"): "2099-01-01",
            },
        )
        try:
            view_availability(df)
        except Exception as e:
            self.fail(f"view_availability raised {e} for all-zeros ticker")

    # ------------------------------------------------------------------
    # Validation — df
    # ------------------------------------------------------------------

    def test_invalid_df_not_qdf(self):
        for bad in ["string", 42, [1, 2, 3], None, pd.DataFrame({"x": [1, 2]})]:
            with self.assertRaises(
                TypeError, msg=f"Expected TypeError for df={bad!r}"
            ):
                view_availability(bad)

    def test_invalid_df_empty(self):
        with self.assertRaises((TypeError, ValueError)):
            view_availability(pd.DataFrame())

    def test_invalid_df_non_binary_values(self):
        df = self.df.copy()
        df.loc[0, "value"] = 0.5
        with self.assertRaises(ValueError):
            view_availability(df)

    def test_invalid_df_continuous_values(self):
        df = self.df.copy().astype({"value": float})
        rng = np.random.default_rng(0)
        df["value"] = rng.standard_normal(len(df))
        with self.assertRaises(ValueError):
            view_availability(df)

    # ------------------------------------------------------------------
    # Validation — n_ticks
    # ------------------------------------------------------------------

    def test_invalid_n_ticks_zero(self):
        with self.assertRaises(ValueError):
            view_availability(self.df, n_ticks=0)

    def test_invalid_n_ticks_negative(self):
        with self.assertRaises(ValueError):
            view_availability(self.df, n_ticks=-3)

    def test_invalid_n_ticks_float(self):
        with self.assertRaises(ValueError):
            view_availability(self.df, n_ticks=5.5)

    def test_invalid_n_ticks_string(self):
        with self.assertRaises(ValueError):
            view_availability(self.df, n_ticks="ten")

    # ------------------------------------------------------------------
    # Validation — kwargs dicts
    # ------------------------------------------------------------------

    def test_invalid_kwargs_dict_type(self):
        for param in (
            "fig_kw",
            "heatmap_kw",
            "xticklabel_kw",
            "yticklabel_kw",
            "title_kw",
        ):
            with self.assertRaises(
                TypeError, msg=f"Expected TypeError for {param}='bad'"
            ):
                view_availability(self.df, **{param: "bad"})

            with self.assertRaises(
                TypeError, msg=f"Expected TypeError for {param}=123"
            ):
                view_availability(self.df, **{param: 123})


if __name__ == "__main__":
    unittest.main()
