import unittest
import pandas as pd
import numpy as np
from typing import Dict, Any
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch

from macrosynergy.visuals.table import view_availability


def _make_binary_df(
    start: str = "2020-01-01",
    end: str = "2022-12-31",
    columns: Dict[str, str] = None,
) -> pd.DataFrame:
    """Build a binary availability DataFrame for testing.

    Each entry in ``columns`` maps a column name to the date from which it is
    available (1 from that date onward, 0 before).
    """
    idx = pd.date_range(start, end, freq="D")
    if columns is None:
        columns = {"A": "2020-01-01", "B": "2021-01-01", "C": "2022-01-01"}
    data = {
        col: (idx >= pd.Timestamp(from_date)).astype(int)
        for col, from_date in columns.items()
    }
    return pd.DataFrame(data, index=idx)


class TestPlotAvailability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")
        cls.mock_show = patch("matplotlib.pyplot.show").start()

        cls.df = _make_binary_df()

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
        # ColA available throughout; ColB available only in 2022; ColC never.
        idx = pd.date_range("2020-01-01", "2022-12-31", freq="MS")
        df = pd.DataFrame(
            {
                "ColA": np.ones(len(idx), dtype=int),
                "ColB": (idx >= pd.Timestamp("2022-01-01")).astype(int),
                "ColC": np.zeros(len(idx), dtype=int),
            },
            index=idx,
        )
        args = self.valid_args.copy()
        args["df"] = df
        args["return_fig"] = True
        fig = view_availability(**args)
        ax = fig.axes[0]
        # y-axis labels reflect column order after sorting (heatmap transposes df)
        labels = [t.get_text() for t in ax.get_yticklabels()]
        cola_pos = labels.index("ColA")
        colb_pos = labels.index("ColB")
        colc_pos = labels.index("ColC")
        # ColA has more observations than ColB; both precede ColC (zero count)
        self.assertLess(cola_pos, colc_pos)
        self.assertLess(colb_pos, colc_pos)

    def test_single_column(self):
        df = _make_binary_df(columns={"OnlyCol": "2020-06-01"})
        try:
            view_availability(df)
        except Exception as e:
            self.fail(f"view_availability raised {e} for single-column DataFrame")

    def test_all_zeros_column(self):
        idx = pd.date_range("2020-01-01", "2021-12-31", freq="MS")
        df = pd.DataFrame(
            {
                "Active": np.ones(len(idx), dtype=int),
                "Inactive": np.zeros(len(idx), dtype=int),
            },
            index=idx,
        )
        try:
            view_availability(df)
        except Exception as e:
            self.fail(f"view_availability raised {e} for all-zeros column")

    # ------------------------------------------------------------------
    # Validation — df
    # ------------------------------------------------------------------

    def test_invalid_df_not_dataframe(self):
        for bad in ["string", 42, [1, 2, 3], None]:
            with self.assertRaises(TypeError, msg=f"Expected TypeError for df={bad!r}"):
                view_availability(bad)

    def test_invalid_df_empty(self):
        with self.assertRaises(ValueError):
            view_availability(pd.DataFrame())

    def test_invalid_df_non_datetime_index(self):
        df = self.df.copy()
        df.index = range(len(df))
        with self.assertRaises(TypeError):
            view_availability(df)

    def test_invalid_df_non_binary_values(self):
        df = self.df.copy().astype(float)
        df.iloc[0, 0] = 0.5
        with self.assertRaises(ValueError):
            view_availability(df)

    def test_invalid_df_continuous_values(self):
        idx = pd.date_range("2020-01-01", periods=50, freq="D")
        df = pd.DataFrame({"X": np.random.randn(50)}, index=idx)
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
