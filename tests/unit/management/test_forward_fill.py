import unittest
import pandas as pd
import numpy as np

from macrosynergy.management.utils import forward_fill_wide_df

class TestForwardFillWideDF(unittest.TestCase):

    def setUp(self):
        self.dates = pd.date_range('2022-01-01', periods=4)

    def test_simple_forward_fill(self):
        df = pd.DataFrame({'A': [1, np.nan, np.nan, 4], 'B': [np.nan, 2, np.nan, np.nan]}, index=self.dates)
        expected = pd.DataFrame({'A': [1, np.nan, np.nan, 4], 'B': [np.nan, 2, 2, np.nan]}, index=self.dates)
        result = forward_fill_wide_df(df.copy(), n=1)
        pd.testing.assert_frame_equal(result, expected)

    def test_n_greater_than_1(self):
        df = pd.DataFrame({'A': [np.nan, 1, np.nan, np.nan]}, index=self.dates)
        expected = pd.DataFrame({'A': [np.nan, 1, 1, 1]}, index=self.dates)
        result = forward_fill_wide_df(df.copy(), n=2)
        pd.testing.assert_frame_equal(result, expected)

    def test_blacklist(self):
        df = pd.DataFrame({'A': [1, np.nan, np.nan, np.nan]}, index=self.dates)
        blacklist = {'A': [self.dates[1], self.dates[1]]}
        expected = pd.DataFrame({'A': [1, np.nan, 1, np.nan]}, index=self.dates)
        result = forward_fill_wide_df(df.copy(), blacklist=blacklist, n=2)
        pd.testing.assert_frame_equal(result, expected)

    def test_column_without_valid(self):
        df = pd.DataFrame({'A': [np.nan, np.nan]}, index=pd.date_range('2022-01-01', periods=2))
        result = forward_fill_wide_df(df.copy())
        pd.testing.assert_frame_equal(result, df)

    def test_invalid_types(self):
        with self.assertRaises(TypeError):
            forward_fill_wide_df([1, 2, 3])
        with self.assertRaises(TypeError):
            forward_fill_wide_df(pd.DataFrame({'A': [1]}), blacklist=[])
        with self.assertRaises(ValueError):
            forward_fill_wide_df(pd.DataFrame({'A': [1]}), n=1.5)

    def test_blacklist_partial_overlap(self):
        df = pd.DataFrame({'A': [1, np.nan, np.nan, np.nan], 'B': [np.nan, 2, np.nan, np.nan]}, index=self.dates)
        blacklist = {'A': [self.dates[2], self.dates[3]]}
        expected = pd.DataFrame({'A': [1, 1, np.nan, np.nan], 'B': [np.nan, 2, 2, 2]}, index=self.dates)
        result = forward_fill_wide_df(df.copy(), blacklist=blacklist, n=3)
        pd.testing.assert_frame_equal(result, expected)

if __name__ == "__main__":
    unittest.main()
