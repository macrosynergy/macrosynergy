import numpy as np
import pandas as pd
import unittest


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)
from macrosynergy.learning import (
    panel_significance_probability,
    regression_accuracy,
    regression_balanced_accuracy,
    sharpe_ratio,
    sortino_ratio,
    neg_mean_abs_corr,
    LinearRegressionSystem,
    RidgeRegressionSystem,
    LADRegressionSystem,
    CorrelationVolatilitySystem,
)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import VotingRegressor

class TestAll(unittest.TestCase):
    def setUp(self):
        cids = ["AUD", "CAD", "GBP", "USD"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2020-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2020-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2020-06-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2020-10-01", "2020-12-31"]

        tuples = []

        for cid in cids:
            # get list of all elidgible dates
            sdate = df_cids.loc[cid]["earliest"]
            edate = df_cids.loc[cid]["latest"]
            all_days = pd.date_range(sdate, edate)
            work_days = all_days[all_days.weekday < 5]
            for work_day in work_days:
                tuples.append((cid, work_day))

        n_predictions = len(tuples)

        # generate random predictions
        classifier_predictions = np.random.choice([0, 1], size=n_predictions)
        regressor_predictions = np.random.normal(size=n_predictions)

        # generate random true values
        classifier_true = np.random.choice([0, 1], size=n_predictions)
        regressor_true = np.random.normal(size=n_predictions)

        self.classifier_predictions = pd.Series(
            data=classifier_predictions, index=pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"])
        )
        self.regressor_predictions = pd.Series(
            data=regressor_predictions, index=pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"])
        )
        self.classifier_true = pd.Series(
            data=classifier_true, index=pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"])
        )
        self.regressor_true = pd.Series(
            data=regressor_true, index=pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"])
        )

        # Create a dataframe to test the scorers, as opposed to the metrics 
        xcats = ["XR", "CPI", "GROWTH", "RIR"]
        ftrs = np.random.normal(loc=0, scale=1, size=(n_predictions, 3))
        labels = np.matmul(ftrs, [1, 2, -1]) + np.random.normal(0, 0.5, len(ftrs))
        df = pd.DataFrame(
            data=np.concatenate((np.reshape(labels, (-1, 1)), ftrs), axis=1),
            index=pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"]),
            columns=xcats,
            dtype=np.float32,
        )

        self.X_train = df.drop(columns="XR")
        self.y_train = df["XR"]

        self.regression_systems = [LinearRegressionSystem(), RidgeRegressionSystem(), LADRegressionSystem()]

    def test_valid_panel_significance_probability(self):
        map_result = panel_significance_probability(self.regressor_true, self.regressor_predictions)
        self.assertIsInstance(map_result, float, "panel_significance_probability should return a float")
        self.assertGreaterEqual(map_result, 0, "panel_significance_probability should return a value greater or equal to zero")
        self.assertLessEqual(map_result, 1, "panel_significance_probability should return a value less or equal to one")
        # check that if all true values are zero, then the result is zero
        all_zeros = pd.Series(data=np.zeros(len(self.regressor_true)), index=self.regressor_true.index)
        map_result = panel_significance_probability(all_zeros, self.regressor_predictions)
        self.assertEqual(map_result, 0, "panel_significance_probability should return zero if all true values are zero")

    def test_types_panel_significance_probability(self):
        with self.assertRaises(TypeError):
            panel_significance_probability("self.regressor_true", self.regressor_predictions)
        with self.assertRaises(TypeError):
            panel_significance_probability(self.regressor_true.reset_index(), self.regressor_predictions)
        with self.assertRaises(ValueError):
            panel_significance_probability(self.regressor_true, self.regressor_predictions[:-1])


    def test_valid_regression_accuracy(self):
        acc_result = regression_accuracy(self.regressor_true, self.regressor_predictions)
        true_signs = np.sign(self.regressor_true)
        pred_signs = np.sign(self.regressor_predictions)
        expected_result = accuracy_score(true_signs, pred_signs)
        self.assertEqual(acc_result, expected_result, "regression_accuracy should return the same value as sklearn.metrics.accuracy_score")

    def test_types_regression_accuracy(self):
        with self.assertRaises(TypeError):
            regression_accuracy("self.regressor_true", self.regressor_predictions)
        with self.assertRaises(TypeError):
            regression_accuracy(self.regressor_true.reset_index(), self.regressor_predictions)
        with self.assertRaises(ValueError):
            regression_accuracy(self.regressor_true, self.regressor_predictions[:-1])

    def test_valid_regression_balanced_accuracy(self):
        true_signs = np.sign(self.regressor_true)
        pred_signs = np.sign(self.regressor_predictions)
        bac_result = regression_balanced_accuracy(self.regressor_true, self.regressor_predictions)
        expected_result = balanced_accuracy_score(true_signs, pred_signs)
        self.assertEqual(bac_result, expected_result, "regression_balanced_accuracy should return the same value as sklearn.metrics.balanced_accuracy_score")

    def test_types_regression_balanced_accuracy(self):
        with self.assertRaises(TypeError):
            regression_balanced_accuracy("self.regressor_true", self.regressor_predictions)
        with self.assertRaises(TypeError):
            regression_balanced_accuracy(self.regressor_true.reset_index(), self.regressor_predictions)
        with self.assertRaises(ValueError):
            regression_balanced_accuracy(self.regressor_true, self.regressor_predictions[:-1])

    def test_valid_sharpe_ratio(self):
        sharpe_result = sharpe_ratio(self.regressor_true, self.regressor_predictions)
        self.assertIsInstance(sharpe_result, float, "sharpe_ratio should return a float")

    def test_types_sharpe_ratio(self):
        with self.assertRaises(TypeError):
            sharpe_ratio("self.regressor_true", self.regressor_predictions)
        with self.assertRaises(TypeError):
            sharpe_ratio(self.regressor_true.reset_index(), self.regressor_predictions)
        with self.assertRaises(ValueError):
            sharpe_ratio(self.regressor_true, self.regressor_predictions[:-1])

    def test_valid_sortino_ratio(self):
        sortino_result = sortino_ratio(self.regressor_true, self.regressor_predictions)
        self.assertIsInstance(sortino_result, float, "sortino_ratio should return a float")

    def test_types_sortino_ratio(self):
        with self.assertRaises(TypeError):
            sortino_ratio("self.regressor_true", self.regressor_predictions)
        with self.assertRaises(TypeError):
            sortino_ratio(self.regressor_true.reset_index(), self.regressor_predictions)
        with self.assertRaises(ValueError):
            sortino_ratio(self.regressor_true, self.regressor_predictions[:-1])

    def test_types_neg_mean_abs_corr(self):
        """ estimator """
        # Should fail if the estimator isn't a sklearn estimator
        with self.assertRaises(TypeError):
            neg_mean_abs_corr(estimator="self.regressor_true", X_test = self.X_train, y_test = self.y_train)
        # Should fail if the estimator isn't a sklearn regressor
        with self.assertRaises(TypeError):
            neg_mean_abs_corr(estimator=LogisticRegression(), X_test = self.X_train, y_test = self.y_train)
        # Should fail if the estimator isn't a system of linear models or a voting regressor 
        with self.assertRaises(ValueError):
            estimator = VotingRegressor(
                [
                    ("OLS_system", LinearRegressionSystem()),
                    ("OLS", LinearRegression()),
                ]
            )
            neg_mean_abs_corr(
                estimator=estimator,
                X_test = self.X_train,
                y_test = self.y_train,
            )
        with self.assertRaises(ValueError):
            estimator = VotingRegressor(
                [
                    ("OLS", LinearRegression()),
                    ("OLS_system", LinearRegressionSystem()),
                ]
            )
            neg_mean_abs_corr(
                estimator=estimator,
                X_test = self.X_train,
                y_test = self.y_train,
            )
        with self.assertRaises(ValueError):
            estimator = VotingRegressor(
                [
                    ("OLS1", LinearRegression()),
                    ("OLS2", LinearRegression()),
                ]
            )
            neg_mean_abs_corr(
                estimator=estimator,
                X_test = self.X_train,
                y_test = self.y_train,
            )
        
        for system in self.regression_systems:
            """ X_train """
            # Should fail if the X_test isn't a pandas dataframe
            with self.assertRaises(TypeError):
                neg_mean_abs_corr(estimator=system, X_test = "self.X_train", y_test = self.y_train)
            # Should fail if X_test is not multi-indexed
            with self.assertRaises(ValueError):
                neg_mean_abs_corr(estimator=system, X_test = self.X_train.reset_index(), y_test = self.y_train)
        
            """ y_train """
            # Should fail if y_test isn't a pandas series
            with self.assertRaises(TypeError):
                neg_mean_abs_corr(estimator=system, X_test = self.X_train, y_test = "self.y_train")
            # Should fail if y_test is not multi-indexed
            with self.assertRaises(ValueError):
                neg_mean_abs_corr(estimator=system, X_test = self.X_train, y_test = self.y_train.reset_index())

            """ correlation """
            # Should fail if correlation is not a string
            with self.assertRaises(TypeError):
                neg_mean_abs_corr(estimator=system, X_test = self.X_train, y_test = self.y_train, correlation = 1)
            # Should fail if correlation is not a valid string
            with self.assertRaises(ValueError):
                neg_mean_abs_corr(estimator=system, X_test = self.X_train, y_test = self.y_train, correlation = "invalid")

    def test_valid_neg_mean_abs_corr(self):
        pass
if __name__ == "__main__":
    unittest.main()