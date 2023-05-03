import unittest
import pandas as pd
from tests.simulate import make_qdf
from macrosynergy.panel.view_timelines import view_timelines
import matplotlib

class TestAll(unittest.TestCase):

    def dataframe_construction(self):
        cids = ['AUD', 'CAD', 'GBP', 'NZD']
        xcats = ['XR', 'CRY', 'INFL']
        df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                    'sd_mult'])
        df_cids.loc['AUD', ] = ['2010-01-01', '2020-12-31', 0.2, 0.2]
        df_cids.loc['CAD', ] = ['2011-01-01', '2020-11-30', 0, 1]
        df_cids.loc['GBP', ] = ['2012-01-01', '2020-11-30', 0, 2]
        df_cids.loc['NZD', ] = ['2012-01-01', '2020-09-30', -0.1, 3]

        df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                    'sd_mult', 'ar_coef', 'back_coef'])

        df_xcats.loc['XR', ] = ['2010-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
        df_xcats.loc['INFL', ] = ['2015-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
        df_xcats.loc['CRY', ] = ['2013-01-01', '2020-10-30', 1, 2, 0.95, 0.5]
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.cids = cids
        self.xcats = xcats
        self.dfd = dfd

        return cids, xcats, dfd
    
    
    def test_view_timelines(self):

        cids, xcats, dfd = self.dataframe_construction()
        
        # test that all the sample cases run
        
        # set backend to Agg to avoid spawning a window
        matplotlib.use('Agg')

        try:
            view_timelines(dfd, xcats=xcats[:2], cids=cids[0],
                        size=(10, 5), title='AUD Return and Carry')

            view_timelines(dfd, xcats=xcats, cids=cids[0],
                        xcat_grid=True, title_adj=0.8,
                        xcat_labels=['Return', 'Carry', 'Inflation'],
                        title='AUD Return, Carry & Inflation')

            view_timelines(dfd, xcats=[xcats[1]], cids=cids, ncol=2, title='Carry',
                        cs_mean=True)

            view_timelines(dfd, xcats=[xcats[0]], cids=cids, ncol=2,
                        cumsum=True, same_y=False, aspect=2)
            
            view_timelines(dfd, xcats=[xcats[0]], cids=cids, ncol=2,
                        cumsum=True, same_y=False, aspect=2, single_chart=True)
            
            # test that any indexing of the  works
            for c in dfd.columns:
                dfdr = dfd.copy().set_index(c)
                view_timelines(dfdr, xcats=[xcats[0]], cids=cids, ncol=2,
                            cumsum=True, same_y=False, aspect=2, single_chart=True)
            
            dfdr = dfd.copy().set_index('real_date')
            # rename column 'value' to 'qwerty'
            dfdr = dfdr.rename(columns={'value': 'qwerty'})
            view_timelines(dfdr, xcats=[xcats[0]], cids=cids, ncol=2,
                        cumsum=True, same_y=False, aspect=2, single_chart=True,
                        val='qwerty')
                        
        except Exception as e:
            self.fail(e)
            
            
        # test that the right errors are raised
        with self.assertRaises(TypeError):
            view_timelines(dfd, xcats=[1], cids=cids, ncol=2, title='Carry',
                cs_mean=92.1) # cs_mean must be boolean
            
        with self.assertRaises(ValueError):
            view_timelines(dfd, xcats=xcats, cids=cids[0],
                xcat_grid=True, title_adj=0.8,
                xcat_labels=['Return', 'Carry', 'Inflation'],
                title='AUD Return, Carry & Inflation', 
                cs_mean=True) # cs_mean must be False if len(xcats) > 1
            
        with self.assertRaises(ValueError):
            view_timelines(dfd, xcats=xcats, cids=cids,
                xcat_grid=True, title_adj=0.8,
                xcat_labels=['Return', 'Carry', 'Inflation'],
                title='AUD Return, Carry & Inflation', 
                ) # if xcat_grid == True, len(xcats) must be == 1
            
        with self.assertRaises(TypeError):
            view_timelines(dfd, xcats=xcats, cids=cids[0],
                title_adj=0.8, same_y=True,
                xcat_labels=['Return', 'Carry', 'Inflation'],
                title='AUD Return, Carry & Inflation', 
                xcat_grid='test',) # xcat_grid must be boolean
            
        with self.assertRaises(TypeError):
            view_timelines(dfd, xcats=xcats, cids=cids[0],
                title_adj=0.8, same_y=True,
                xcat_labels=['Return', 'Carry', 'Inflation'],
                title='AUD Return, Carry & Inflation', 
                single_chart='test',) # single_chart must be boolean

        with self.assertRaises(ValueError):
            view_timelines(dfd, xcats=xcats, cids=cids[0],
                title_adj=0.8, same_y=True,
                xcat_labels=['Return', 'Carry', 'Inflation'],
                title='AUD Return, Carry & Inflation', 
                single_chart=True, xcat_grid=True) # (xcat_grid && single_chart) must be False
            
        with self.assertRaises(ValueError):
            dfdr = dfd.copy().set_index('real_date').drop('cid', axis=1)
            view_timelines(dfdr, xcats=xcats, cids=cids[0],
                title_adj=0.8, same_y=True,
                xcat_labels=['Return', 'Carry', 'Inflation'],
                title='AUD Return, Carry & Inflation',
                xcat_grid=True) # df must have a column named 'cid'


        
if __name__ == '__main__':
    unittest.main()