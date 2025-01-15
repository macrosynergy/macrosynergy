import unittest

import numpy as np
import pandas as pd
from parameterized import parameterized

from macrosynergy.management.utils import calculate_cumulative_weights, ewm_sum


class TestMathUtils(unittest.TestCase):
    
    def setUpClass():
        np.random.seed(0)
    
    def test_calculate_cumulative_weights(self):
        df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [4, 5, 6, 7]
        })
        halflife = 1
        expected_weights = pd.Series([1., 1.5, 1.75, 1.875], index=df.index) 
        weights = calculate_cumulative_weights(df, halflife)
        pd.testing.assert_series_equal(weights, expected_weights, check_exact=False, atol=1e-04)
        
        halflife = 2
        expected_weights = pd.Series([1., 1.7071, 2.2071, 2.5607], index=df.index) 
        weights = calculate_cumulative_weights(df, halflife)
        pd.testing.assert_series_equal(weights, expected_weights, check_exact=False, atol=1e-04)

    def test_ewm_sum(self):
        # Test with a simple DataFrame
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        halflife = 1
        expected_result = pd.DataFrame({
            'A': [1.0, 2.5, 4.25],
            'B': [4.0, 7, 9.5]
        })
        result = ewm_sum(df, halflife)

        pd.testing.assert_frame_equal(result, expected_result)
    
    @parameterized.expand([[1], [2], [5.5]])
    def test_ewm_sum_random(self, halflife):

        df = pd.DataFrame(np.random.randn(10, 3), columns=['A', 'B', 'C'])
        
        halflife = 2

        expected_result = pd.DataFrame(np.zeros(df.shape), columns=df.columns)

        for r in range(len(df)):
            moving_sum = np.zeros(df.shape[1])
            for l in range(r + 1):
                w = (1/2) ** ((r - l) / halflife)
                moving_sum += df.iloc[l] * w
            
            expected_result.iloc[r] = moving_sum

        result = ewm_sum(df, halflife)

        pd.testing.assert_frame_equal(result, expected_result)

    def test_ewm_sum_inputs(self):
        # Test for invalid DataFrame input to ewm_sum
        with self.assertRaises(TypeError):
            ewm_sum("not_a_dataframe", 2)
            
        # Test for invalid halflife input to ewm_sum
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        with self.assertRaises(TypeError):
            ewm_sum(df, "not_a_number")

    def test_empty_df(self):
        # Test with an empty DataFrame
        df = pd.DataFrame()
        halflife = 2
        
        result = ewm_sum(df, halflife)
        expected_result = pd.DataFrame()
        
        pd.testing.assert_frame_equal(result, expected_result)
        

if __name__ == "__main__":
    unittest.main()
