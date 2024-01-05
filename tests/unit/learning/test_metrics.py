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
)

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

    def test_functionalities(self):
        """
        Test, where possible, that each of the metrics work as expected.
        """
        # panel_significance_probability
        map_result = panel_significance_probability(self.regressor_true, self.regressor_predictions)
        self.assertIsInstance(map_result, float, "panel_significance_probability should return a float")
        self.assertGreaterEqual(map_result, 0, "panel_significance_probability should return a value greater or equal to zero")
        self.assertLessEqual(map_result, 1, "panel_significance_probability should return a value less or equal to one")
        # regression accuracy
        acc_result = regression_accuracy(self.regressor_true, self.regressor_predictions)
        true_signs = np.sign(self.regressor_true)
        pred_signs = np.sign(self.regressor_predictions)
        expected_result = accuracy_score(true_signs, pred_signs)
        self.assertEqual(acc_result, expected_result, "regression_accuracy should return the same value as sklearn.metrics.accuracy_score")
        # regression balanced accuracy
        bac_result = regression_balanced_accuracy(self.regressor_true, self.regressor_predictions)
        expected_result = balanced_accuracy_score(true_signs, pred_signs)
        self.assertEqual(bac_result, expected_result, "regression_balanced_accuracy should return the same value as sklearn.metrics.balanced_accuracy_score")
        # sharpe_ratio
        sharpe_result = sharpe_ratio(self.regressor_true, self.regressor_predictions)
        self.assertIsInstance(sharpe_result, float, "sharpe_ratio should return a float")
        # sortino_ratio
        sortino_result = sortino_ratio(self.regressor_true, self.regressor_predictions)
        self.assertIsInstance(sortino_result, float, "sortino_ratio should return a float")

if __name__ == "__main__":
    unittest.main()