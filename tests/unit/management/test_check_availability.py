
import unittest
import pandas as pd
from macrosynergy.management.check_availability import *
from tests.simulate import make_qdf

class TestAll(unittest.TestCase):

    def dataframe_constructor(self):

        self.__dict__["cids"] = ["AUD", "CAD", "GBP"]
        self.__dict__["xcats"] = ["CRY", "XR", "GROWTH", "INFL", "GDP"]

        df_cids = pd.DataFrame(index=self.cids,
                               columns=["earliest", "latest", "mean_add", "sd_mult"])

        df_cids.loc["AUD", :] = ["2011-01-01", "2022-08-10", 0.5, 2]
        df_cids.loc["CAD", :] = ["2011-01-01", "2022-08-24", 0, 1]
        df_cids.loc["GBP", :] = ["2011-01-01", "2022-08-15", -0.2, 0.5]

        df_xcats = pd.DataFrame(index=self.xcats,
                                columns=["earliest", "latest", "mean_add", "sd_mult",
                                         "ar_coef", "back_coef"])

        df_xcats.loc["CRY", :] = ["2011-01-01", "2022-08-25", 1, 2, 0.9, 0.5]
        df_xcats.loc["XR", :] = ["2011-01-01", "2022-08-25", 0, 1, 0, 0.3]
        df_xcats.loc["GROWTH", :] = ["2011-01-01", "2022-08-25", 0, 2, 0, 0.4]
        df_xcats.loc["INFL", :] = ["2011-01-01", "2022-08-25", 0, 3, 0, 0.6]
        df_xcats.loc["GDP", :] = ["2011-01-01", "2022-08-25", 0, 1, 0, 0.7]

        np.random.seed(0)

        self.__dict__["dfd"] = make_qdf(df_cids, df_xcats, back_ar=0.75)

    def test_check_startyears(self):

        self.dataframe_constructor()
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

        self.dataframe_constructor()
        dfd = self.dfd

        df_ed = check_enddates(dfd)
        # Reorder the categories alphabetically.
        df_ed = df_ed.reindex(sorted(df_ed.columns), axis=1)

        df_exp = pd.DataFrame(
            data=np.zeros((3, 5)), index=self.cids, columns=self.xcats
        )

        for cid in self.cids:
            for xcat in self.xcats:

                # Validate on the DataFrame received by the method.
                filt_1 = (dfd["xcat"] == xcat) & (dfd["cid"] == cid)
                dfd_reduced = dfd[filt_1]["real_date"]
                end_date = max(dfd_reduced)
                df_exp.loc[cid, xcat] = end_date.strftime("%Y-%m-%d")

        df_exp = df_exp.reindex(sorted(df_exp.columns), axis=1)
        self.assertTrue(df_ed.equals(df_exp))

    def test_business_day_dif(self):

        self.dataframe_constructor()
        dfd = self.dfd

        dfe = check_enddates(dfd)
        dfe = dfe.apply(pd.to_datetime)

        maxdate = dfe.max().max()
        bus_df = business_day_dif(df=dfe, maxdate=maxdate)

        # The last realised value in the series occurs on "2022-08-24". Australia series
        # will have a final value of "2022-08-15" and Great Britain will have
        # "2022-08-10".
        self.assertTrue(all(bus_df.loc["AUD", :].values == 10))
        self.assertTrue(all(bus_df.loc["GBP", :].values == 7))


if __name__ == '__main__':

    unittest.main()
