import unittest
from unittest.mock import patch
from typing import List

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from macrosynergy.management.utils import standardise_dataframe
from macrosynergy.visuals import view_weights


class TestViewWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        plt.close("all")
        cls.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")
        cls.mock_show = patch("matplotlib.pyplot.show").start()

    @classmethod
    def tearDownClass(cls) -> None:
        patch.stopall()
        plt.close("all")
        matplotlib.use(cls.mpl_backend)

    def setUp(self) -> None:
        self.cids: List[str] = ["USEQ", "DXEQ", "EMEQ", "CASH"]
        dates = pd.bdate_range("2015-01-01", "2020-12-31")
        rng = np.random.default_rng(42)
        raw = pd.DataFrame(
            rng.normal(0, 1, (len(dates), len(self.cids))),
            index=dates,
            columns=self.cids,
        )
        ew = np.exp(raw)
        # Softmax weights: non-negative and summing to one on every date.
        self.weights: pd.DataFrame = ew.div(ew.sum(axis=1), axis=0)

        dfa = self.weights.stack().rename("value").reset_index()
        dfa.columns = ["real_date", "cid", "value"]
        dfa["xcat"] = "WGT"
        self.qdf = standardise_dataframe(dfa)

    def tearDown(self) -> None:
        plt.close("all")

    def test_wide_input(self):
        try:
            view_weights(self.weights)
        except Exception as e:
            self.fail(f"view_weights raised {e} unexpectedly")

        fig = view_weights(self.weights, return_fig=True)
        self.assertIsInstance(fig, plt.Figure)

        # Weights that sum to one stack to one.
        top = max(
            path.vertices[:, 1].max()
            for path in fig.axes[0].collections[-1].get_paths()
        )
        self.assertAlmostEqual(float(top), 1.0, places=6)

    def test_qdf_input_matches_wide(self):
        fig_w = view_weights(self.weights, cids=self.cids, return_fig=True)
        fig_q = view_weights(
            self.qdf, xcat="WGT", cids=self.cids, return_fig=True
        )
        self.assertEqual(
            [t.get_text() for t in fig_w.axes[0].get_legend().get_texts()],
            [t.get_text() for t in fig_q.axes[0].get_legend().get_texts()],
        )

    def test_cids_order_and_labels(self):
        # A non-alphabetical order must be honoured, since it is the stacking order.
        order = ["CASH", "EMEQ", "USEQ"]
        fig = view_weights(self.weights, cids=order, return_fig=True)
        self.assertEqual(
            [t.get_text() for t in fig.axes[0].get_legend().get_texts()], order
        )

        fig = view_weights(
            self.weights,
            cids=order,
            cid_labels={"CASH": "Cash", "USEQ": "US equity"},
            return_fig=True,
        )
        self.assertEqual(
            [t.get_text() for t in fig.axes[0].get_legend().get_texts()],
            ["Cash", "EMEQ", "US equity"],
        )

    def test_titles_and_labels(self):
        fig = view_weights(
            self.weights,
            title="Allocation weights",
            ylabel="WGTMACRO",
            xlabel="date",
            return_fig=True,
        )
        ax = fig.axes[0]
        self.assertEqual(ax.get_title(), "Allocation weights")
        self.assertEqual(ax.get_ylabel(), "WGTMACRO")
        self.assertEqual(ax.get_xlabel(), "date")

    def test_legend_can_be_suppressed(self):
        fig = view_weights(self.weights, legend=False, return_fig=True)
        self.assertIsNone(fig.axes[0].get_legend())

    def test_freq_downsamples_to_period_end(self):
        fig = view_weights(self.weights, freq="A", return_fig=True)
        self.assertIsInstance(fig, plt.Figure)

        # Annual sampling keeps the last observation of each calendar year.
        expected = self.weights.loc[
            self.weights.index.to_series().groupby(self.weights.index.year).max().values
        ]
        self.assertEqual(len(expected), self.weights.index.year.nunique())

        for freq in ["D", "W", "M", "Q", "A"]:
            try:
                view_weights(self.weights, freq=freq)
            except Exception as e:
                self.fail(f"view_weights(freq={freq!r}) raised {e} unexpectedly")

    def test_start_end_truncation(self):
        fig = view_weights(
            self.weights, start="2018-01-01", end="2018-12-31", return_fig=True
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_negative_weights_rejected(self):
        # A stack of mixed-sign bands does not add up, so it is refused outright.
        mixed = self.weights.copy()
        mixed.iloc[0, 0] = -0.5
        with self.assertRaises(ValueError) as ctx:
            view_weights(mixed)
        self.assertIn("change sign", str(ctx.exception))

    def test_input_validation(self):
        with self.assertRaises(TypeError):
            view_weights([1, 2, 3])

        with self.assertRaises(ValueError):
            view_weights(pd.DataFrame())

        # A quantamental frame needs the category naming the weights.
        with self.assertRaises(ValueError):
            view_weights(self.qdf)

        # ... and a wide frame must not be given one.
        with self.assertRaises(ValueError):
            view_weights(self.weights, xcat="WGT")

        with self.assertRaises(ValueError):
            view_weights(self.weights, cids=["NOT_A_CID"])

        with self.assertRaises(ValueError):
            view_weights(self.qdf, xcat="NOT_AN_XCAT")

        with self.assertRaises(ValueError):
            view_weights(self.weights, freq="H")

        with self.assertRaises(TypeError):
            view_weights(self.weights, cid_labels=["USEQ"])

        # No data left after truncation.
        with self.assertRaises(ValueError):
            view_weights(self.weights, start="2100-01-01")


if __name__ == "__main__":
    unittest.main()
