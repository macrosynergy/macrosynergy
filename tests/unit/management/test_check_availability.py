import unittest
import numpy as np
import pandas as pd

from macrosynergy.management.simulate_quantamental_data import make_qdf_
from macrosynergy.management.check_availability import check_startyears, check_enddates

cids = ['AUD', 'CAD', 'GBP']
xcats = ['CRY', 'XR']
df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
df_cids.loc['AUD', :] = ['2010-01-01', '2020-12-31', 0.5, 2]
df_cids.loc['CAD', :] = ['2011-01-01', '2020-11-30', 0, 1]
df_cids.loc['GBP', :] = ['2012-01-01', '2020-11-30', -0.2, 0.5]

df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
df_xcats.loc['CRY', :] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
df_xcats.loc['XR', :] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]

dfd, fields_cats, fields_cids, df_year, df_end, df_missing, cids_cats = make_qdf_(df_cids, df_xcats, back_ar = 0.75)

df_sy = check_startyears(df_year)
df_ed = check_enddates(df_end)


class TestAll(unittest.TestCase):
    def test_check_startyears(self):

        df_sy = check_startyears(df_year)
        df_exp = pd.DataFrame(data = np.zeros((3, 2)), index = cids, columns = xcats)
        for cid in cids:
            for xcat in xcats:
                cid_max = pd.Series(pd.to_datetime(df_cids.loc[cid, 'earliest'])).dt.year
                xcat_max = pd.Series(pd.to_datetime(df_xcats.loc[xcat, 'earliest'])).dt.year
                df_exp.loc[cid, xcat] = np.max([cid_max, xcat_max])
        self.assertTrue((df_sy.astype(int)).equals(df_exp.astype(int)))

    def test_check_enddates(self):
        
        df_ed = check_enddates(df_end)
        df_exp = pd.DataFrame(data = np.zeros((3, 2)), index = cids, columns = xcats)
        for cid in cids:
            for xcat in xcats:
                cid_min = pd.to_datetime(df_cids.loc[cid, 'latest'])
                xcat_min = pd.to_datetime(df_xcats.loc[xcat, 'latest'])
                df_exp.loc[cid, xcat] = pd.to_datetime(np.min([cid_min, xcat_min]).strftime("%Y-%m-%d"))
                df_exp.index.name = 'cid'
                df_exp.columns.name = 'xcat'

        df_ed = df_ed.astype(dtype = object)
        self.assertTrue(df_ed.equals(df_exp))


if __name__ == '__main__':
    unittest.main()
