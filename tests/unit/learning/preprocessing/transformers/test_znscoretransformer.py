import numpy as np
import pandas as pd
import unittest
import itertools
from parameterized import parameterized

from macrosynergy.learning import ZnScoreAverager

from sklearn.base import TransformerMixin

class TestZnScoreAverager(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Generate data with true linear relationship
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2019-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2019-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2019-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2019-01-01", "2020-12-31"]

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
        self.y = df["XR"]

    def test_constructor(self):
        # Testing the constructor for correct attribute initialization
        averager = ZnScoreAverager(neutral="mean", use_signs=True)
        self.assertEqual(averager.neutral, "mean")
        self.assertTrue(averager.use_signs)

        # Testing the constructor for handling invalid inputs
        with self.assertRaises(TypeError):
            ZnScoreAverager(neutral=123)
        with self.assertRaises(ValueError):
            ZnScoreAverager(neutral="invalid_value")
        with self.assertRaises(TypeError):
            ZnScoreAverager(use_signs="not_a_boolean")

    def test_fit_neutral_zero(self):
        averager = ZnScoreAverager(neutral="zero")
        averager.fit(self.X)
        self.assertIsNotNone(averager.training_mads)
        self.assertEqual(averager.training_n, len(self.X))

        with self.assertRaises(TypeError):
            averager.fit("not_a_dataframe")

        with self.assertRaises(ValueError):
            averager.fit(pd.DataFrame())

    def test_fit_neutral_mean(self):
        averager = ZnScoreAverager(neutral="mean")
        averager.fit(self.X)
        self.assertIsNotNone(averager.training_means)
        self.assertIsNotNone(averager.training_sum_squares)
        self.assertEqual(averager.training_n, len(self.X))

        with self.assertRaises(TypeError):
            averager.fit("not_a_dataframe")

        with self.assertRaises(ValueError):
            averager.fit(pd.DataFrame())

    def test_transform_types_neutral_zero(self):
        averager = ZnScoreAverager(neutral="zero")
        averager.fit(self.X)
        transformed = averager.transform(self.X)
        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertEqual(transformed.shape, (self.X.shape[0], 1))

        with self.assertRaises(TypeError):
            averager.transform("not_a_dataframe")

    def test_transform_types_neutral_mean(self):
        averager = ZnScoreAverager(neutral="mean")
        averager.fit(self.X)
        transformed = averager.transform(self.X)
        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertEqual(transformed.shape, (self.X.shape[0], 1))

        with self.assertRaises(TypeError):
            averager.transform("not_a_dataframe")

    @parameterized.expand([["zero"], ["mean"]])
    def test_transform_values_use_signs(self, neutral):
        averager = ZnScoreAverager(neutral=neutral, use_signs=True)
        averager.fit(self.X)
        transformed = averager.transform(self.X).abs()
        transformed_abs = transformed.abs()
        self.assertTrue(np.all(transformed_abs == 1))

    def test_get_expanding_count(self):
        averager = ZnScoreAverager(neutral="zero")
        self.assertTrue(
            np.all(
                averager._get_expanding_count(self.X)[-1]
                == np.array([len(self.X)] * self.X.columns.size)
            )
        )