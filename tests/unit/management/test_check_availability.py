import io
import unittest
import unittest.mock
from typing import List, Optional
from unittest.mock import patch
from macrosynergy.management.simulate import make_test_df
import matplotlib
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

from macrosynergy.management.utils.check_availability import (
    business_day_dif,
    check_availability,
    check_enddates,
    check_startyears,
    missing_in_df,
)
from tests.simulate import make_qdf


class TestAll(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mpl_backend: str = matplotlib.get_backend()
        matplotlib.use("Agg")
        cls.mock_show = patch("matplotlib.pyplot.show").start()

    def setUp(self) -> None:
        self.cids: List[str] = ["AUD", "CAD", "GBP"]
        self.xcats: List[str] = ["CRY", "XR", "GROWTH", "INFL", "GDP"]

        df_cids = pd.DataFrame(
            index=self.cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
        )

        df_cids.loc["AUD", :] = ["2011-01-01", "2022-08-10", 0.5, 2]
        df_cids.loc["CAD", :] = ["2011-01-01", "2022-08-24", 0, 1]
        df_cids.loc["GBP", :] = ["2011-01-01", "2022-08-15", -0.2, 0.5]

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

        df_xcats.loc["CRY", :] = ["2011-01-01", "2022-08-25", 1, 2, 0.9, 0.5]
        df_xcats.loc["XR", :] = ["2011-01-01", "2022-08-25", 0, 1, 0, 0.3]
        df_xcats.loc["GROWTH", :] = ["2011-01-01", "2022-08-25", 0, 2, 0, 0.4]
        df_xcats.loc["INFL", :] = ["2011-01-01", "2022-08-25", 0, 3, 0, 0.6]
        df_xcats.loc["GDP", :] = ["2011-01-01", "2022-08-25", 0, 1, 0, 0.7]

        np.random.seed(0)

        self.dfd: pd.DataFrame = make_qdf(df_cids, df_xcats, back_ar=0.75)

    def tearDown(self) -> None:
        plt.close("all")

    @classmethod
    def tearDownClass(cls) -> None:
        patch.stopall()
        plt.close("all")
        matplotlib.use(cls.mpl_backend)

    def test_check_startyears(self):
        dfd = self.dfd

        df_sy = check_startyears(self.dfd)
        # Reorder the categories alphabetically.
        df_sy = df_sy.reindex(sorted(df_sy.columns), axis=1)

        df_exp = pd.DataFrame(
            data=np.zeros((3, 5)), index=self.cids, columns=self.xcats
        )

        for cid in self.cids:
            for xcat in self.xcats:
                # Validate on the DataFrame received by the method.
                filt_1 = (dfd["xcat"] == xcat) & (dfd["cid"] == cid)
                dfd_reduced = dfd[filt_1]["real_date"].dt.year
                year_date = min(dfd_reduced)
                df_exp.loc[cid, xcat] = year_date

        df_exp = df_exp.reindex(sorted(df_exp.columns), axis=1)
        self.assertTrue((df_sy.astype(int)).equals(df_exp.astype(int)))

    def test_check_enddates(self):
        dfd = self.dfd

        df_ed = check_enddates(dfd)

        # Reorder the categories alphabetically.
        df_ed = df_ed.reindex(sorted(df_ed.columns), axis=1)

        df_exp: pd.DataFrame = pd.DataFrame(
            # data=np.zeros((3, 5)),
            index=self.cids,
            columns=self.xcats,
        )

        for cid in self.cids:
            for xcat in self.xcats:
                # Validate on the DataFrame received by the method.
                filt_1 = (dfd["xcat"] == xcat) & (dfd["cid"] == cid)
                end_date = dfd[filt_1]["real_date"].max()
                df_exp.loc[cid, xcat] = pd.Timestamp(end_date).strftime("%Y-%m-%d")

        df_exp = df_exp.reindex(sorted(df_exp.columns), axis=1)
        self.assertTrue(df_ed.equals(df_exp))

    def test_business_day_dif(self):
        dfd: pd.DataFrame = self.dfd

        dfe: pd.DataFrame = check_enddates(dfd)
        dfe: pd.DataFrame = dfe.apply(pd.to_datetime)

        maxdate = dfe.max().max()
        bus_df = business_day_dif(df=dfe, maxdate=maxdate)

        # The last realised value in the series occurs on "2022-08-24". Australia series
        # will have a final value of "2022-08-15" and Great Britain will have
        # "2022-08-10".
        self.assertTrue(all(bus_df.loc["AUD", :].values == 10))
        self.assertTrue(all(bus_df.loc["GBP", :].values == 7))

    def test_missing_in_df(self):

        dfd: pd.DataFrame = self.dfd
        cids: List[str] = self.cids
        xcats: List[str] = self.xcats

        # Drop a few random tickers from the DataFrame.
        random.seed(1)
        tkdrop = random.sample([(_c, _x) for _c in cids for _x in xcats], 5)
        for _c, _x in tkdrop:
            dfd = dfd[~((dfd["cid"] == _c) & (dfd["xcat"] == _x))]
        dfd = dfd.reset_index(drop=True)

        # Mock the standard output.
        output: Optional[str] = None
        with unittest.mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            missing_in_df(dfd, xcats, cids)
            output = mock_stdout.getvalue()

        # Validate the output.
        lines = [l for l in output.splitlines() if l.strip()]  # Remove empty lines.
        # first line should be no missing xcats
        self.assertTrue(lines[0].startswith("No missing XCATs across DataFrame."))

        # each consecutive line should be a missing cid for a xcat
        for line in lines[1:]:
            self.assertTrue(line.startswith("Missing cids for"))
            xcat_str, cids_str = line.split(":")
            _xcat = xcat_str.strip().replace("Missing cids for ", "")
            _cids = eval(cids_str.strip())
            _tkdrop = [(_c, _xcat) for _c in _cids]
            # each cid, xcat pair should be in the list of dropped tickers
            self.assertTrue(all([_ in tkdrop for _ in _tkdrop]))

        # select one more random xcat
        random_xcat: str = random.choice(xcats)
        dfd = dfd[~(dfd["xcat"] == random_xcat)]

        with unittest.mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            missing_in_df(dfd, xcats, cids)
            output = mock_stdout.getvalue()

        # Validate the output.
        lines = [l for l in output.splitlines() if l.strip()]
        # first line should be a missing xcat, the rest has been tested above
        self.assertTrue(lines[0].startswith("Missing XCATs across DataFrame:"))
        extracted_xcat = eval(lines[0].split(":")[1].strip())
        self.assertTrue(len(extracted_xcat) == 1)
        self.assertTrue(extracted_xcat[0] == random_xcat)

        with self.assertRaises(TypeError):
            missing_in_df(df=1, xcats=xcats, cids=cids)

        with self.assertRaises(ValueError):
            missing_in_df(df=dfd[0:0], xcats=xcats, cids=cids)

        with self.assertRaises(TypeError):
            missing_in_df(df=dfd, xcats=[1], cids=cids)

        self.assertIsNone(missing_in_df(df=dfd, xcats=["apple", "banana"], cids=cids))

    def test_check_availability(self):

        cids: List[str] = self.cids
        xcats: List[str] = self.xcats

        df = make_test_df(cids=cids, xcats=xcats)

        with self.assertRaises(TypeError):
            check_availability(df=df, xcats=xcats, cids=cids, start_years="True")

        with self.assertRaises(TypeError):
            check_availability(df=df, xcats=xcats, cids=cids, missing_recent="True")

        with self.assertRaises(ValueError):
            bad_cids = ["apple", "banana"]
            check_availability(df=df, xcats=xcats, cids=bad_cids)

    def test_check_availability_start_years(self):

        check_availability(
            df=self.dfd, xcats=self.xcats, cids=self.cids, start_years=True
        )

    def test_check_availability_missing_recent(self):
        check_availability(
            df=self.dfd, xcats=self.xcats, cids=self.cids, missing_recent=True
        )


if __name__ == "__main__":
    unittest.main()
