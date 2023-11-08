import unittest
from macrosynergy.signal.signal_return import SignalsReturns

from macrosynergy.management.simulate import make_qdf
from sklearn.metrics import accuracy_score, precision_score
from scipy import stats
import random
import pandas as pd
import numpy as np
from typing import List, Dict
import matplotlib


class TestAll(unittest.TestCase):
    def setUp(self):
        """
        Create a standardised DataFrame defined over the three categories.
        """

        self.cids: List[str] = ["AUD", "CAD", "GBP", "NZD", "USD"]
        self.xcats: List[str] = ["XR", "CRY", "GROWTH", "INFL"]

        df_cids = pd.DataFrame(
            index=self.cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )

        # Purposefully choose a different start date for all cross-sections. Used to test
        # communal sampling.
        df_cids.loc["AUD"] = ["2011-01-01", "2020-12-31", 0, 1]
        df_cids.loc["CAD"] = ["2009-01-01", "2020-10-30", 0, 2]
        df_cids.loc["GBP"] = ["2010-01-01", "2020-08-30", 0, 5]
        df_cids.loc["NZD"] = ["2008-01-01", "2020-06-30", 0, 3]
        df_cids.loc["USD"] = ["2012-01-01", "2020-12-31", 0, 4]

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

        df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
        df_xcats.loc["CRY"] = ["2000-01-01", "2020-10-30", 0, 2, 0.95, 1]
        df_xcats.loc["GROWTH"] = ["2001-01-01", "2020-10-30", 0, 2, 0.9, 1]
        df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 0, 2, 0.8, 0.5]

        random.seed(2)
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

        self.dfd: pd.DataFrame = dfd

        black = {
            "AUD": ["2000-01-01", "2003-12-31"],
            "GBP": ["2018-01-01", "2100-01-01"],
        }

        self.blacklist: Dict[str, List[str]] = black

        assert "dfd" in vars(self).keys(), (
            "Instantiation of DataFrame missing from " "field dictionary."
        )

    def test_constructor(self):
        # Test the Class's constructor.

        # First, test the assertions.
        # Testing that the dataframe is actually of type Pandas.DataFrame
        with self.assertRaises(TypeError):
            sr_df_type = SignalsReturns(
                df="STRING", rets="XR", sigs="CRY", freqs="D", blacklist=self.blacklist
            )
        # Testing that the dataframe has the correct columns
        with self.assertRaises(ValueError):
            sr_df = SignalsReturns(
                df=pd.DataFrame(index=["FAIL"], columns=["FAIL"]),
                rets="XR",
                sigs="CRY",
                freqs="D",
                blacklist=self.blacklist,
            )
        # Test to confirm the primary signal must be present in the passed Dataframe
        with self.assertRaises(AssertionError):
            sr_sigs = SignalsReturns(
                df=self.dfd,
                rets="XR",
                sigs="MISSING",
                freqs="D",
                blacklist=self.blacklist,
            )
        # Test to confirm the primary signal must be present in the passed Dataframe
        with self.assertRaises(AssertionError):
            sr_rets = SignalsReturns(
                df=self.dfd,
                rets="MISSING",
                sigs="CRY",
                freqs="D",
                blacklist=self.blacklist,
            )
        # Test to ensure that freqs is a specified string or list of strings
        with self.assertRaises(ValueError):
            sr_freq = SignalsReturns(
                df=self.dfd,
                rets="XR",
                sigs="CRY",
                freqs="X",
                blacklist=self.blacklist,
            )
        # Test that cosp is set to a boolean value
        with self.assertRaises(TypeError):
            sr_cosp = SignalsReturns(
                df=self.dfd,
                rets="XR",
                sigs="CRY",
                freqs="D",
                cosp="FAIL",
                blacklist=self.blacklist,
            )
        # Test that Signals Returns can take in lists as arguments
        self.assertTrue(
            SignalsReturns(
                df=self.dfd,
                rets=["XR", "GROWTH"],
                sigs=["CRY", "INFL"],
                freqs=["Q", "M"],
                signs=[1, 1],
                agg_sigs=["last", "mean"],
                blacklist=self.blacklist,
            )
        )

    def test_single_relation_table(self):
        sr = SignalsReturns(
            df=self.dfd,
            rets="XR",
            sigs="CRY",
            freqs="Q",
            blacklist=self.blacklist,
            slip=1,
        )

        sr_no_slip = SignalsReturns(
            df=self.dfd,
            rets="XR",
            sigs="CRY",
            freqs="Q",
            blacklist=self.blacklist,
            slip=0,
        )

        # Test that each argument must be of the correct type
        with self.assertRaises(TypeError):
            sr.single_relation_table(ret=2)

        with self.assertRaises(TypeError):
            sr.single_relation_table(xcat=2)

        with self.assertRaises(TypeError):
            sr.single_relation_table(freq=2)

        with self.assertRaises(TypeError):
            sr.single_relation_table(agg_sigs=2)

        sr.single_relation_table()
        sr_no_slip.single_relation_table()

        # Test that dataframe has been reduced to just the relevant columns and has
        # applied slippage

        self.assertTrue(set(sr.dfd["xcat"]) == set(["XR", "CRY"]))

        self.assertTrue(sr.dfd["value"][0] != sr.df["value"][0])

        self.assertTrue(sr_no_slip.dfd["value"][0] == sr_no_slip.df["value"][0])

        sr.single_relation_table(ret="XR", xcat="CRY", freq="Q", agg_sigs="last")

        self.assertTrue(set(sr.dfd["xcat"]) == set(["XR", "CRY"]))

        self.assertTrue(sr.dfd["value"][0] != sr.df["value"][0])

        # Test Negative signs are correctly handled

        with self.assertRaises(TypeError):
            sr_sign_fail = SignalsReturns(
                df=self.dfd,
                rets="XR",
                sigs="CRY",
                freqs="Q",
                signs=["FAIL"],
                blacklist=self.blacklist,
                slip=1,
            )

        # Ensure that the signs doesn't have a longer length than the number of signals
        with self.assertRaises(ValueError):
            sr_long_signs = SignalsReturns(
                df=self.dfd,
                rets="XR",
                sigs="CRY",
                freqs="Q",
                signs=[-1, -1],
                blacklist=self.blacklist,
                slip=1,
            )

        # Test table outputted is correct
        data = {
            "cid": [
                "AUD",
                "AUD",
                "AUD",
                "AUD",
                "AUD",
                "AUD",
                "AUD",
                "AUD",
                "AUD",
                "AUD",
            ],
            "xcat": ["XR", "XR", "XR", "XR", "XR", "CRY", "CRY", "CRY", "CRY", "CRY"],
            "real_date": [
                "1990-01-01",
                "1990-01-02",
                "1990-01-03",
                "1990-01-04",
                "1990-01-05",
                "1990-01-01",
                "1990-01-02",
                "1990-01-03",
                "1990-01-04",
                "1990-01-05",
            ],
            "value": [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        }

        test_df = pd.DataFrame(data)

        sr_correct = SignalsReturns(
            df=test_df,
            rets="XR",
            sigs="CRY",
            freqs="D",
            blacklist=None,
            slip=0,
        )

        srt = sr_correct.single_relation_table()

        correct_stats = [
            0.25,
            0.25,
            0.5,
            0.75,
            0.5,
            0.0,
            -0.57735,
            0.42265,
            -0.57735,
            0.31731,
        ]

        for val1, val2 in zip(srt.iloc[0].values.tolist(), correct_stats):
            self.assertTrue(np.isclose(val1, val2))

        # Check when signs are negative

        sr_correct_neg = SignalsReturns(
            df=test_df,
            rets="XR",
            sigs="CRY",
            freqs="D",
            signs=-1,
            blacklist=None,
            slip=0,
        )

        srt = sr_correct_neg.single_relation_table()

        correct_stats = [
            0.75,
            0.75,
            0.5,
            0.75,
            1.0,
            0.5,
            0.57735,
            0.42265,
            0.57735,
            0.31731,
        ]

        for val1, val2 in zip(srt.iloc[0].values.tolist(), correct_stats):
            self.assertTrue(np.isclose(val1, val2))

    def test_multiple_relation_table(self):
        num_of_acc_cols = 10

        sr_unsigned = SignalsReturns(
            df=self.dfd,
            rets="XR",
            sigs="CRY",
            freqs="Q",
            agg_sigs="last",
            blacklist=self.blacklist,
            slip=1,
        )

        self.assertTrue(
            sr_unsigned.multiple_relations_table(rets="XR", xcats="CRY").shape
            == (1, num_of_acc_cols)
        )

        sr_mrt = SignalsReturns(
            df=self.dfd,
            rets=["XR", "GROWTH"],
            sigs=["CRY", "INFL"],
            freqs=["Q", "M"],
            # signs=[1, 1],
            agg_sigs=["last", "mean"],
            blacklist=self.blacklist,
        )

        self.assertTrue(
            sr_mrt.multiple_relations_table().shape == (16, num_of_acc_cols)
        )

        with self.assertRaises(ValueError):
            sr_mrt.multiple_relations_table(rets="TEST")

        with self.assertRaises(ValueError):
            sr_mrt.multiple_relations_table(xcats="TEST")

        with self.assertRaises(ValueError):
            sr_mrt.multiple_relations_table(freqs="TEST")

        with self.assertRaises(ValueError):
            sr_mrt.multiple_relations_table(agg_sigs="TEST")

        # Test that the table is inputs can take in both a list of strings and a string
        # self.assertTrue(sr_mrt.multiple_relations_table(rets="XR", freqs='Q'))

        rets = ["XR", "GROWTH"]
        xcats = ["INFL"]
        freqs = ["Q", "M"]
        agg_sigs = ["mean"]
        mrt = sr_mrt.multiple_relations_table(
            rets=rets, xcats=xcats, freqs=freqs, agg_sigs=agg_sigs
        )
        self.assertTrue(mrt.shape == (4, num_of_acc_cols))

    def test_single_statistic_table(self):
        sr = SignalsReturns(
            df=self.dfd,
            rets="XR",
            sigs="CRY",
            freqs="Q",
            blacklist=self.blacklist,
            slip=1,
        )

        with self.assertRaises(ValueError):
            sr.single_statistic_table(stat="FAIL")

        with self.assertRaises(ValueError):
            sr.single_statistic_table(stat="accuracy", type="FAIL")

        with self.assertRaises(TypeError):
            sr.single_statistic_table(stat="accuracy", rows="FAIL")

        with self.assertRaises(TypeError):
            sr.single_statistic_table(stat="accuracy", columns="FAIL")

        with self.assertRaises(ValueError):
            sr.single_statistic_table(stat="accuracy", rows=["FAIL"])

        with self.assertRaises(ValueError):
            sr.single_statistic_table(stat="accuracy", columns=["FAIL"])

        # Test that table is correctly shaped

        self.assertTrue(sr.single_statistic_table(stat="accuracy").shape == (1, 1))

        sr = SignalsReturns(
            df=self.dfd,
            rets=["XR", "GROWTH"],
            sigs=["CRY", "INFL"],
            freqs=["Q", "M"],
            agg_sigs=["last", "mean"],
            blacklist=self.blacklist,
            slip=1,
        )

        self.assertTrue(sr.single_statistic_table(stat="accuracy").shape == (4, 4))

        # Test that the table is correctly shaped when rows and columns are specified

        self.assertTrue(
            sr.single_statistic_table(
                stat="accuracy", rows=["freq", "xcat", "ret"], columns=["agg_sigs"]
            ).shape
            == (8, 2)
        )

        sr = SignalsReturns(
            df=self.dfd,
            rets=["XR", "GROWTH"],
            sigs=["CRY", "INFL"],
            freqs=["Q", "M"],
            agg_sigs=["last", "mean"],
            blacklist=self.blacklist,
            signs=[1, -1],
        )

        self.assertTrue(
            sr.single_statistic_table(
                stat="accuracy", rows=["xcat"], columns=["freq", "agg_sigs", "ret"]
            ).index[1]
            == "INFL_NEG"
        )

    def test_set_df_labels(self):
        rets = ["XR", "GROWTH"]
        freqs = ["Q", "M"]
        sigs = ["CRY", "INFL"]
        agg_sigs = ["mean", "last"]

        rows_dict = {"xcat": sigs, "ret": rets, "freq": freqs, "agg_sigs": agg_sigs}
        rows = ["xcat", "ret", "freq"]
        columns = ["agg_sigs"]

        sr = SignalsReturns(
            df=self.dfd,
            rets=rets,
            sigs=sigs,
            freqs=freqs,
            agg_sigs=agg_sigs,
            blacklist=self.blacklist,
        )

        rows_names, columns_names = sr.set_df_labels(
            rows_dict=rows_dict, rows=rows, columns=columns
        )
        expected_col_names = ["mean", "last"]
        expected_row_names = [
            "CRY/XR/Q",
            "CRY/XR/M",
            "CRY/GROWTH/Q",
            "CRY/GROWTH/M",
            "INFL/XR/Q",
            "INFL/XR/M",
            "INFL/GROWTH/Q",
            "INFL/GROWTH/M",
        ]

        self.assertTrue(rows_names == expected_row_names)
        self.assertTrue(columns_names == expected_col_names)

        rows = ["xcat", "ret"]
        columns = ["agg_sigs", "freq"]

        rows_names, columns_names = sr.set_df_labels(
            rows_dict=rows_dict, rows=rows, columns=columns
        )

        expected_col_names = ["mean/Q", "mean/M", "last/Q", "last/M"]
        expected_row_names = ["CRY/XR", "CRY/GROWTH", "INFL/XR", "INFL/GROWTH"]

        self.assertTrue(rows_names == expected_row_names)
        self.assertTrue(columns_names == expected_col_names)

        rows = ["xcat"]
        columns = columns = ["agg_sigs", "ret", "freq"]

        rows_names, columns_names = sr.set_df_labels(
            rows_dict=rows_dict, rows=rows, columns=columns
        )

        expected_col_names = [
            "mean/XR/Q",
            "mean/XR/M",
            "mean/GROWTH/Q",
            "mean/GROWTH/M",
            "last/XR/Q",
            "last/XR/M",
            "last/GROWTH/Q",
            "last/GROWTH/M",
        ]
        expected_row_names = ["CRY", "INFL"]

        self.assertTrue(rows_names == expected_row_names)
        self.assertTrue(columns_names == expected_col_names)

        return 0

    def test_get_rowcol(self):
        rets = ["XR", "GROWTH"]
        freqs = ["Q", "M"]
        sigs = ["CRY", "INFL"]
        agg_sigs = ["mean", "last"]

        sr = SignalsReturns(
            df=self.dfd,
            rets=rets,
            sigs=sigs,
            freqs=freqs,
            agg_sigs=agg_sigs,
            blacklist=self.blacklist,
        )

        hash = "XR/CRY/Q/mean"
        rows = ["xcat", "ret", "freq"]
        columns = ["agg_sigs"]

        self.assertTrue(sr.get_rowcol(hash, rows) == "CRY/XR/Q")
        self.assertTrue(sr.get_rowcol(hash, columns) == "mean")

    def test_single_statistic_table_show_heatmap(self):
        self.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")

        sr = SignalsReturns(
            df=self.dfd,
            rets="XR",
            sigs="CRY",
            freqs="Q",
            blacklist=self.blacklist,
            slip=1,
        )

        try:
            sr.single_statistic_table(stat="accuracy", show_heatmap=True)
        except Exception as e:
            self.fail(f"single_statistic_table raised {e} unexpectedly")

        try:
            sr.single_statistic_table(
                stat="accuracy", show_heatmap=True, row_names=["X"], column_names=["Y"]
            )
        except Exception as e:
            self.fail(f"single_statistic_table raised {e} unexpectedly")

        try:
            sr.single_statistic_table(
                stat="accuracy",
                show_heatmap=True,
                title="Test",
                min_color=0,
                max_color=1,
                annotate=False,
                figsize=(10, 10),
            )
        except Exception as e:
            self.fail(f"single_statistic_table raised {e} unexpectedly")


if __name__ == "__main__":
    unittest.main()
