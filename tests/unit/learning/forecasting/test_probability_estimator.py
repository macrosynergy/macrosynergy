import unittest

import numpy as np
import pandas as pd

from macrosynergy.learning import ProbabilityEstimator

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class TestProbabilityEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Generate data with true linear relationship
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2019-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2020-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2020-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2019-06-01", "2020-12-31"]

        tuples = []

        for cid in cids:
            # get list of all eligible dates
            sdate = df_cids.loc[cid]["earliest"]
            edate = df_cids.loc[cid]["latest"]
            all_days = pd.date_range(sdate, edate)
            work_days = all_days[all_days.weekday < 5]
            for work_day in work_days:
                tuples.append((cid, work_day))

        n_samples = len(tuples)
        ftrs = np.random.normal(loc=0, scale=1, size=(n_samples, 3))
        labels = np.matmul(ftrs, [1, 2, -1]) + np.random.normal(0, 0.5, len(ftrs))
        df = pd.DataFrame(
            data=np.concatenate((np.reshape(labels, (-1, 1)), ftrs), axis=1),
            index=pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"]),
            columns=xcats,
            dtype=np.float32,
        )

        self.X = df.drop(columns="XR")
        self.X_numpy = self.X.values
        self.X_nan = self.X.copy()
        self.X_nan["nan_col"] = np.nan
        self.y = np.sign( df["XR"] )
        self.y_numpy = self.y.values
        self.y_nan = self.y.copy()
        self.y_nan.iloc[0] = np.nan

        # Valid model fitting
        self.pe_lr = ProbabilityEstimator(LogisticRegression()).fit(self.X, self.y)
        self.lr = LogisticRegression().fit(self.X, self.y)
        self.pe_rf = ProbabilityEstimator(RandomForestClassifier(random_state=1)).fit(self.X, self.y)
        self.rf = RandomForestClassifier(random_state=1).fit(self.X, self.y)

    def test_types_init(self):
        # classifier
        self.assertRaises(TypeError, ProbabilityEstimator, classifier=1)
        self.assertRaises(TypeError, ProbabilityEstimator, classifier=LinearRegression)

    def test_valid_init(self):
        # Check that the attributes are set correctly
        pe = ProbabilityEstimator(LogisticRegression())
        self.assertIsInstance(pe.classifier, LogisticRegression)
        self.assertEqual(pe.classes_, [-1,1])
        self.assertEqual(pe.feature_importances_, None)

    def test_types_fit(self):
        # X - when a dataframe
        pe = ProbabilityEstimator(classifier=LogisticRegression())
        self.assertRaises(TypeError, pe.fit, X=1, y=self.y)
        self.assertRaises(TypeError, pe.fit, X="X", y=self.y)
        self.assertRaises(ValueError, pe.fit, X=self.X.reset_index(), y=self.y)
        self.assertRaises(ValueError, pe.fit, X=self.X_nan, y=self.y)
        self.assertRaises(ValueError, pe.fit, X=self.X_nan.values, y=self.y)
        # X - when a numpy array
        self.assertRaises(ValueError, pe.fit, X=self.X.reset_index().values, y=self.y)
        self.assertRaises(
            ValueError, pe.fit, X=self.X_nan.reset_index(drop=True).values, y=self.y
        )
        # y - when a series
        self.assertRaises(TypeError, pe.fit, X=self.X, y=1)
        self.assertRaises(TypeError, pe.fit, X=self.X, y="y")
        self.assertRaises(ValueError, pe.fit, X=self.X, y=self.y_nan)
        self.assertRaises(ValueError, pe.fit, X=self.X, y=self.y_nan.values)
        # y - when a dataframe
        self.assertRaises(ValueError, pe.fit, X=self.X, y=self.y.reset_index())
        self.assertRaises(
            ValueError,
            pe.fit,
            X=self.X,
            y=pd.DataFrame(self.y_nan.reset_index(drop=True)),
        )
        # y - when a numpy array
        self.assertRaises(
            ValueError, pe.fit, X=self.X.values, y=np.zeros((len(self.X), 2))
        )
        self.assertRaises(
            ValueError, pe.fit, X=self.X.values, y=np.array([np.nan] * len(self.X))
        )

        self.assertRaises(ValueError, pe.fit, X=self.X, y=self.y[:-1])

    def test_valid_fit(self):
        """ Check that the underlying logistic regression is the same as the one fitted """
        # Coefficients
        self.assertTrue(np.allclose(self.pe_lr.classifier.coef_, self.lr.coef_))
        self.assertTrue(np.allclose(self.pe_lr.coef_, self.lr.coef_))
        # Intercept
        self.assertTrue(np.allclose(self.pe_lr.classifier.intercept_, self.lr.intercept_))
        self.assertTrue(np.allclose(self.pe_lr.intercept_, self.lr.intercept_))
        # Check feature importances are correct
        self.assertTrue(np.allclose(self.pe_lr.feature_importances_, np.abs(self.lr.coef_) / np.sum(np.abs(self.lr.coef_))))

        """ Check that the underlying random forest is the same as the one fitted """
        # Check feature importances are correct
        self.assertTrue(np.allclose(self.pe_rf.feature_importances_, self.rf.feature_importances_))

    def test_types_predict(self):
        pe = ProbabilityEstimator(LogisticRegression()).fit(self.X, self.y)
        # X - when a dataframe
        self.assertRaises(TypeError, pe.predict, X=1)
        self.assertRaises(TypeError, pe.predict, X="X")
        self.assertRaises(ValueError, pe.predict, X=self.X.iloc[:, :-1])
        self.assertRaises(ValueError, pe.predict, X=self.X_nan.values)
        self.assertRaises(
            ValueError,
            pe.predict,
            X=pd.DataFrame(
                data = np.array([["hello"] * self.X.shape[1]] * self.X.shape[0]),
                columns = self.X.columns,
                index = self.X.index,
            ),
        )
        self.assertRaises(
            ValueError,
            pe.predict,
            X=pd.DataFrame(
                data = np.array([[np.nan] * self.X.shape[1]] * self.X.shape[0]),
                columns = self.X.columns,
                index = self.X.index,
            ),
        )
        # X - when a numpy array
        self.assertRaises(ValueError, pe.predict, X=np.expand_dims(self.X_numpy, 0))
        self.assertRaises(
            ValueError,
            pe.predict,
            X=np.array([["hello"] * self.X.shape[1]] * self.X.shape[0])
        )
        self.assertRaises(
            ValueError,
            pe.predict,
            X=np.array([[np.nan] * self.X.shape[1]] * self.X.shape[0])
        ) 

    def test_valid_predict(self):
        """ Check that the predictions of the probability estimator are the same as the ones from the underlying classifier """
        self.assertTrue(np.allclose(self.pe_lr.predict(self.X), self.lr.predict(self.X)))
        self.assertTrue(np.allclose(self.pe_rf.predict(self.X), self.rf.predict(self.X)))

    def test_types_createsignal(self):
        pe = ProbabilityEstimator(LogisticRegression()).fit(self.X, self.y)
        # X - when a dataframe
        self.assertRaises(TypeError, pe.create_signal, X=1)
        self.assertRaises(TypeError, pe.create_signal, X="X")
        self.assertRaises(ValueError, pe.create_signal, X=self.X.iloc[:, :-1])
        self.assertRaises(ValueError, pe.create_signal, X=self.X_nan.values)
        self.assertRaises(
            ValueError,
            pe.create_signal,
            X=pd.DataFrame(
                data = np.array([["hello"] * self.X.shape[1]] * self.X.shape[0]),
                columns = self.X.columns,
                index = self.X.index,
            ),
        )
        self.assertRaises(
            ValueError,
            pe.create_signal,
            X=pd.DataFrame(
                data = np.array([[np.nan] * self.X.shape[1]] * self.X.shape[0]),
                columns = self.X.columns,
                index = self.X.index,
            ),
        )
        # X - when a numpy array
        self.assertRaises(ValueError, pe.create_signal, X=np.expand_dims(self.X_numpy, 0))
        self.assertRaises(
            ValueError,
            pe.create_signal,
            X=np.array([["hello"] * self.X.shape[1]] * self.X.shape[0])
        )
        self.assertRaises(
            ValueError,
            pe.create_signal,
            X=np.array([[np.nan] * self.X.shape[1]] * self.X.shape[0])
        )

    def test_valid_createsignal(self):
        """ Check that the signals of the probability estimator are the same as the ones from the underlying classifier """
        self.assertTrue(np.allclose(self.pe_lr.create_signal(self.X), self.lr.predict_proba(self.X)[:,1] - 0.5))
        self.assertTrue(np.allclose(self.pe_rf.create_signal(self.X), self.rf.predict_proba(self.X)[:,1] - 0.5))