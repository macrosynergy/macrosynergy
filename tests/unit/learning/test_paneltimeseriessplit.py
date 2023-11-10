import unittest 
from macrosynergy.backend import get_current_backend

if get_current_backend() == "pandas":
    import pandas as pd
elif get_current_backend() == "modin.pandas":
    import modin.pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from macrosynergy.learning import PanelTimeSeriesSplit

class TestAll(unittest.TestCase):

    def setUp(self):
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(
            index=cids, columns=["earliest", "latest"]
        )
        df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2000-01-01", "2020-12-31"]

        tuples = []

        for cid in cids:
            # get list of all elidgible dates
            sdate = df_cids.loc[cid]["earliest"]
            edate = df_cids.loc[cid]["latest"]
            all_days = pd.date_range(sdate, edate)
            work_days = all_days[all_days.weekday < 5]
            for work_day in work_days:
                tuples.append((cid, work_day))

        n_samples = len(tuples)
        ftrs = np.random.normal(loc=0,scale=1,size=(n_samples,3))
        labels = np.matmul(ftrs,[1,2,-1]) + np.random.normal(0,0.5,len(ftrs))
        df = pd.DataFrame(data=np.concatenate((np.reshape(labels,(-1,1)),ftrs),axis=1), index=pd.MultiIndex.from_tuples(tuples,names=["cid","real_date"]),columns=xcats,dtype=np.float32)

        self.X = df.drop(columns="XR")
        self.y = df["XR"]

    def test_crossval_application(self):
        # Given a generated panel with a true linear relationship between features and target,
        # test that the cross validation procedure correctly identifies that a linear regression
        # is more suitable than a 1-nearest neighbor model.
        self.setUp()

        # models
        lr = LinearRegression()
        knn = KNeighborsRegressor(n_neighbors=1)
        splitter = PanelTimeSeriesSplit(train_intervals=1,min_cids=2,min_periods=21*12,test_size=1)
        lrsplits = cross_val_score(lr, self.X, self.y, scoring="neg_root_mean_squared_error", cv = splitter, n_jobs=-1)
        knnsplits = cross_val_score(knn, self.X, self.y, scoring="neg_root_mean_squared_error", cv = splitter, n_jobs=-1)

        self.assertLess(np.mean(-lrsplits),np.mean(-knnsplits))

if __name__ == '__main__':

    unittest.main()