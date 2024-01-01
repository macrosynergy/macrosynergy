import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from tests.simulate import make_qdf
from macrosynergy.visuals.correlation import view_correlation, _cluster_correlations
from macrosynergy.panel.correlation import correl_matrix, _transform_df_for_cross_sectional_corr
from macrosynergy.management.utils import reduce_df

from matplotlib import pyplot as plt
import matplotlib
from unittest.mock import patch

class TestAll(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Prevents plots from being displayed during tests.
        plt.close("all")
        self.mpl_backend: str = matplotlib.get_backend()
        self.mock_show = patch("matplotlib.pyplot.show").start()
    
    @classmethod
    def tearDownClass(self) -> None:
        patch.stopall()
        plt.close("all")
        patch.stopall()
        matplotlib.use(self.mpl_backend)

    def setUp(self) -> None:
        self.cids = ["AUD", "CAD", "GBP"]
        self.xcats = ["XR", "CRY", "GROWTH", "INFL"]

        df_cids = pd.DataFrame(
            index=self.cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )
        df_cids.loc["AUD", :] = ["2010-01-01", "2020-12-31", 0.5, 2]
        df_cids.loc["CAD", :] = ["2010-01-01", "2020-11-30", 0, 1]
        df_cids.loc["GBP", :] = ["2012-01-01", "2020-11-30", -0.2, 0.5]

        df_xcats = pd.DataFrame(
            index=self.xcats,
            columns=[
                "earliest",
                "latest",
                "mean_add",
                "sd_mult",
                "ar_coef",
                "back_coef",
            ],
        )
        df_xcats.loc["CRY", :] = ["2010-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
        df_xcats.loc["XR", :] = ["2011-01-01", "2020-12-31", 0, 1, 0, 0.3]
        df_xcats.loc["GROWTH", :] = ["2010-01-01", "2020-10-30", 1, 2, 0.9, 0.5]
        df_xcats.loc["INFL", :] = ["2010-01-01", "2020-10-30", 0, 2, 0.9, 0.5]

        # Standard df for tests.
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.dfd = dfd
        self.corr = correl_matrix(
                df=self.dfd,
                xcats=["XR"],
                xcats_secondary=["CRY"],
                cids=["AUD"],
                cids_secondary=["GBP"],
                max_color=0.1,
                plot=False,
            )

    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_view_correlation(self):
        try:
            view_correlation(corr=self.corr)
        except Exception as e:
            self.fail(f"view_correlation raised {e} unexpectedly")

    def test_view_correlation_types(self):
        with self.assertRaises(TypeError):
            view_correlation(corr="invalid_type")
        with self.assertRaises(TypeError):
            view_correlation(corr=self.corr, max_color="invalid_type")
        with self.assertRaises(TypeError):
            view_correlation(corr=self.corr, mask="invalid_type")
        with self.assertRaises(TypeError):
            view_correlation(corr=self.corr, cluster="invalid_type")
        with self.assertRaises(TypeError):
            view_correlation(corr=self.corr, title=0)
        with self.assertRaises(TypeError):
            view_correlation(corr=self.corr, size="invalid_type")
        with self.assertRaises(TypeError):
            view_correlation(corr=self.corr, xlabel=0)
        with self.assertRaises(TypeError):
            view_correlation(corr=self.corr, ylabel=0)

    def test_cluster_correlations(self):
        df = reduce_df(self.dfd, xcats=["XR"], cids=self.cids)
        df_w = _transform_df_for_cross_sectional_corr(df, val="value")

        corr1 = df_w.corr(method="pearson")
        corr2 = corr1.copy()

        # Clustering rows and columns separately should provide the same outcome
        # for a symmetric dataframe.
        corr1 = _cluster_correlations(corr1, is_symmetric=True)
        corr2 = _cluster_correlations(corr2, is_symmetric=False)
        corr2 = _cluster_correlations(corr2.T, is_symmetric=False).T

        assert_frame_equal(corr1, corr2)


if __name__ == "__main__":
    unittest.main()
