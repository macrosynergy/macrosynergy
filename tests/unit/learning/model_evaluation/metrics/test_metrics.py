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
    correlation_coefficient,
)

class TestMetrics(unittest.TestCase):
    def setUp(self):
        cids = ["AUD", "CAD", "GBP", "USD"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2020-06-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2020-06-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2020-09-01", "2020-12-31"]
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

    def test_types_accuracy(self):
        pass