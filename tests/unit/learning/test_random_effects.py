import unittest

import numpy as np
import pandas as pd
from linearmodels.panel import RandomEffects as lm_RandomEffects
from scipy.stats import expon
from sklearn.base import BaseEstimator
from statsmodels.tools.tools import add_constant

from macrosynergy.learning import RandomEffects


class TestRandomEffects(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Generate data with true linear relationship
        cids = ["AUD", "CAD", "GBP", "USD"]
        xcats = ["XR", "CPI", "GROWTH", "RIR"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["AUD"] = ["2020-01-01", "2020-12-31"]
        df_cids.loc["CAD"] = ["2020-01-01", "2020-12-31"]
        df_cids.loc["GBP"] = ["2020-01-01", "2020-12-31"]
        df_cids.loc["USD"] = ["2020-06-01", "2020-12-31"]

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

        X = df.drop(columns="XR")
        y = df["XR"]

        unique_xss = sorted(X.index.get_level_values(0).unique())
        xs_codes = dict(zip(unique_xss, range(1, len(unique_xss) + 1)))

        self.X = X
        self.y = y

        self.lm_X = X.rename(xs_codes, level=0, inplace=False).copy()
        self.lm_y = y.rename(xs_codes, level=0, inplace=False).copy()

    def test_init(self):
        re = RandomEffects()
        self.assertIsInstance(re, BaseEstimator)
        self.assertIsInstance(re, RandomEffects)
        self.assertEqual(re.fit_intercept, True)
        self.assertEqual(re.group_col, "real_date")

    def test_init_with_params(self):
        re = RandomEffects(fit_intercept=False, group_col="cid")
        self.assertIsInstance(re, BaseEstimator)
        self.assertIsInstance(re, RandomEffects)
        self.assertEqual(re.fit_intercept, False)
        self.assertEqual(re.group_col, "cid")

    def test_invalid_init(self):
        with self.assertRaises(TypeError):
            RandomEffects(fit_intercept=False, group_col=True)

        with self.assertRaises(TypeError):
            RandomEffects(fit_intercept="invalid", group_col="cid")

    def test_invalid_fit(self):
        re = RandomEffects()
        with self.assertRaises(TypeError):
            re.fit("invalid", self.y)

        invalid_idx_X = pd.DataFrame()

        with self.assertRaises(ValueError):
            re.fit(invalid_idx_X, self.y)

        re.group_col = "invalid"
        with self.assertRaises(ValueError):
            re.fit(self.X, self.y)

        # Drop half of x
        X = self.X.iloc[: len(self.X) // 2].copy()
        with self.assertRaises(ValueError):
            re.fit(X, self.y)


    def test_fit_no_intercept(self):
        """Test fit method with no intercept against linearmodels implementation"""
        re = RandomEffects(fit_intercept=False, group_col="real_date")
        re.fit(self.X, self.y)
        lm_re = lm_RandomEffects(self.lm_y.swaplevel(), self.lm_X.swaplevel()).fit()

        self.assertTrue(np.allclose(re.params.values, lm_re.params.values, atol=1e-3))
        self.assertTrue(np.allclose(re.pvals.values, lm_re.pvalues.values, atol=1e-2))
        self.assertTrue(np.allclose(re.cov, lm_re.cov.values, atol=1e-3))
        self.assertTrue(np.allclose(re.residual_ss, lm_re.resid_ss, atol=1e-3))
        self.assertTrue(np.allclose(re.residuals, lm_re._resids, atol=1e-3))
        self.assertTrue(np.allclose(re.std_errors.values, lm_re.std_errors.values, atol=1e-3))

        
    def test_fit_with_intercept(self):
        """Test fit method with intercept against linearmodels implementation"""
        re = RandomEffects(fit_intercept=True, group_col="real_date")
        re.fit(self.X, self.y)

        lm_X = add_constant(self.lm_X)
        lm_re = lm_RandomEffects(self.lm_y.swaplevel(), lm_X.swaplevel()).fit()

        self.assertTrue(np.allclose(re.params.values, lm_re.params.values, atol=1e-3))
        self.assertTrue(np.allclose(re.pvals.values, lm_re.pvalues.values, atol=1e-2))
        self.assertTrue(np.allclose(re.cov, lm_re.cov.values, atol=1e-3))
        self.assertTrue(np.allclose(re.residual_ss, lm_re.resid_ss, atol=1e-3))
        self.assertTrue(np.allclose(re.residuals, lm_re._resids, atol=1e-3))
        self.assertTrue(np.allclose(re.std_errors.values, lm_re.std_errors.values, atol=1e-3))

    def test_fit_by_ftr(self):
        """Test fit method with intercept against linearmodels implementation"""

        for ftr in self.X.columns:
            re = RandomEffects(fit_intercept=True, group_col="real_date")
            re.fit(self.X[[ftr]], self.y)

            lm_X = add_constant(self.lm_X[[ftr]])
            lm_re = lm_RandomEffects(self.lm_y.swaplevel(), lm_X.swaplevel()).fit()

            self.assertTrue(np.allclose(re.params.values, lm_re.params.values, atol=1e-3))
            self.assertTrue(np.allclose(re.pvals.values, lm_re.pvalues.values, atol=1e-2))
            self.assertTrue(np.allclose(re.cov, lm_re.cov.values, atol=1e-3))
            self.assertTrue(np.allclose(re.residual_ss, lm_re.resid_ss, atol=1e-3))
            self.assertTrue(np.allclose(re.residuals, lm_re._resids, atol=1e-3))
            self.assertTrue(np.allclose(re.std_errors.values, lm_re.std_errors.values, atol=1e-3))



if __name__ == "__main__":
    tests = TestRandomEffects()
    tests.setUpClass()
    tests.test_fit_no_intercept
    tests.test_fit_with_intercept()
    tests.test_fit_by_ftr()
