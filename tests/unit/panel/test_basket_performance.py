import unittest
import random
import pandas as pd
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.panel.basket_performance import basket_performance


class MyTestCase(unittest.TestCase):
    def setUp(self):
        cids = ["AUD", "GBP", "NZD", "USD", "TRY"]
        xcats = ["FX_XR", "FX_CRY", "EQ_XR", "EQ_CRY"]

        df_cids = pd.DataFrame(
            index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )

        df_cids.loc["AUD"] = ["2010-12-01", "2020-12-31", 0, 1]
        df_cids.loc["GBP"] = ["2011-01-01", "2020-11-30", 0, 2]
        df_cids.loc["NZD"] = ["2012-01-01", "2020-11-30", 0, 3]
        df_cids.loc["USD"] = ["2013-01-01", "2020-09-30", 0, 4]
        df_cids.loc["TRY"] = ["2002-01-01", "2020-09-30", 0, 5]

        df_xcats = pd.DataFrame(
            index=xcats,
            columns=[
                "earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"
            ],
        )
        df_xcats.loc["FX_XR"] = ["2010-01-01", "2020-12-31", 0, 1, 0, 0.2]
        df_xcats.loc["FX_CRY"] = ["2011-01-01", "2020-10-30", 1, 1, 0.9, 0.5]
        df_xcats.loc["EQ_XR"] = ["2011-01-01", "2020-10-30", 0.5, 2, 0, 0.2]
        df_xcats.loc["EQ_CRY"] = ["2013-01-01", "2020-10-30", 1, 1, 0.9, 0.5]

        random.seed(2)
        self.dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.black = {
            "AUD": ["2000-01-01", "2003-12-31"],
            "GBP": ["2018-01-01", "2100-01-01"]
        }

        self.contracts = ["AUD_FX", "NZD_FX", "GBP_EQ", "USD_EQ"]

    def test_inverse_standard_deviations(self):
        dfd = basket_performance(
            self.dfd,
            self.contracts,
            ret="XR",
            cry="CRY",
            weight_meth="invsd",
            lback_meth="ma",
            lback_periods=21,
            weights=None,
            weight_xcat=None,
            max_weight=1.0,
            return_weights=True,
        )
        self.assertIsInstance(dfd, pd.DataFrame)

        self.assertEqual(
            ["ticker", "real_date", "value"], dfd.columns.values.tolist()
        )

    def test_inverse_standard_deviations_xma(self):
        dfd = basket_performance(
            self.dfd,
            self.contracts,
            ret="XR",
            cry="CRY",
            weight_meth="invsd",
            lback_meth="xma",
            lback_periods=21,
            weights=None,
            weight_xcat=None,
            max_weight=1.0,
            return_weights=True,
        )
        self.assertIsInstance(dfd, pd.DataFrame)

        self.assertEqual(
            ["ticker", "real_date", "value"], dfd.columns.values.tolist()
        )

    def test_equal(self):
        dfd = basket_performance(
            self.dfd,
            self.contracts,
            ret="XR",
            cry="CRY",
            weight_meth="equal",
            weights=None,
            weight_xcat=None,
            max_weight=1.0,
            return_weights=False,
        )
        self.assertIsInstance(dfd, pd.DataFrame)

        self.assertEqual(
            ["ticker", "real_date", "value"], dfd.columns.values.tolist()
        )

    def test_max_weight(self):
        with self.assertRaises(NotImplementedError):
            basket_performance(
                self.dfd,
                self.contracts,
                ret="XR",
                cry="CRY",
                weight_meth="invsd",
                lback_meth="ma",
                lback_periods=21,
                weights=None,
                weight_xcat=None,
                max_weight=0.3,
                return_weights=True,
            )


if __name__ == '__main__':
    unittest.main()
