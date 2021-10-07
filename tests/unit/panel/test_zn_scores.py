
import unittest
import numpy as np
import pandas as pd
from itertools import groupby
from random import randint, choice, shuffle, seed
from collections import defaultdict

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.panel.make_zn_scores import pn_neutral, cross_neutral, make_zn_scores, nan_insert


cids = ['AUD', 'CAD', 'GBP']
xcats = ['CRY', 'XR']
df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
df_cids.loc['AUD', :] = ['2010-01-01', '2020-12-31', 0.5, 2]
df_cids.loc['CAD', :] = ['2011-01-01', '2020-11-30', 0, 1]
df_cids.loc['GBP', :] = ['2012-01-01', '2020-11-30', -0.2, 0.5]

df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
df_xcats.loc['CRY', :] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
df_xcats.loc['XR', :] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]

dfd = make_qdf(df_cids, df_xcats, back_ar = 0.75)

seed()
class TestAll(unittest.TestCase):

    def test_pen_neutral(self):

        ## Assess the dimensions of the returned Array. Should return a one-dimensional Array whose length matches the number of rows in the DataFrame.
        xcat = 'XR'
        df = reduce_df(dfd, xcats = [xcat], cids = cids)
        df_pivot = df.pivot(index='real_date', columns='cid', values='value')
        ar_neutral = pn_neutral(df_pivot, 'mean', True)
        self.assertTrue(df_pivot.shape[0] == len(ar_neutral))
        
        arr = np.linspace(1, 11, 11)
        arr = arr.astype(dtype = np.float32)
        arr_rev = arr[::-1]
        arr_data = np.column_stack((arr, arr_rev))
        
        columns = ['Series_1', 'Series_2']
        df = pd.DataFrame(data = arr_data, columns = columns)

        ## Validate the np.repeat() operation is working correctly, using the above DataFrame, if the sequential parameter equals False.
        ## Test that only a single unique value is returned, sample median, and test the value equals the expected median, 6.0.
        neutral = "median"
        sequential = False
        ar_neutral = pn_neutral(df, neutral, sequential)

        self.assertIsInstance(ar_neutral, np.ndarray)
        item_list = list(set(ar_neutral))

        self.assertTrue(len(item_list) == 1)
        med_val = arr[len(arr) // 2]
        item = item_list.pop()
        
        self.assertEqual(item, 6.0)

        ## Assess the median algorithm, and the stacking mechanism. 
        size = 45
        value = (size // 2) + 1
        arr = np.linspace(1, size, size)
        arr = arr.astype(dtype = np.float32)
        arr_rev = arr[::-1]
        arr_data = np.column_stack((arr, arr_rev))

        df = pd.DataFrame(data = arr_data, columns = columns)
        
        ar_neutral = pn_neutral(df, neutral, sequential = True)
        ar_val = np.repeat(value, repeats = size)
        
        self.assertTrue(np.all(ar_neutral == ar_val))

        ## Assess the mean algorithm.
        size = randint(1, 115)
        arr = np.linspace(1, size, size, dtype = np.float32)
        
        arr_rev = arr[::-1]
        arr_data = np.column_stack((arr, arr_rev))
        
        np.random.shuffle(arr_data)
        df = pd.DataFrame(data = arr_data, columns = columns)
        min_ = np.min(arr_data)
        max_ = np.max(arr_data)
        val = float((min_ + max_) / 2)

        ar_neutral = pn_neutral(df, 'mean', sequential = False)
        data_unq = list(groupby(ar_neutral))
        data_unq = data_unq.pop()[0]

        self.assertEqual(data_unq, val)

    def test_cross_neutral(self):

        ## Assess the dimensions of the Array. Irrespective of the two parameters, "neutral" & "sequential", the dimensions of the output Array should match the received DataFrame. 
        arr = np.linspace(1, 100, 100, dtype = np.float32)
        arr = arr.reshape((20, 5))
        columns = arr.shape[1]
        columns = ['Series_' + str(i + 1) for i in range(columns)]
        df = pd.DataFrame(data = arr, columns = columns)

        neutral = choice(['mean', 'median', 'zero'])
        sequential = choice([True, False])

        arr_neutral = cross_neutral(df, neutral, sequential)
        self.assertIsInstance(arr_neutral, np.ndarray)

        df_shape = df.shape
        self.assertEqual(df_shape, arr_neutral.shape)

        ## Test the Cross_Sectional median algorithm's functionality with a contrived data set.
        ## Generate a two dimensional Array consisting of an iterative sequence, F(x) = (x + 1), where the input is the first column's index, and the adjacent column will host the sequence in reverse.
        ## If the Cross-Sectional Rolling Median algorithm is correct, the difference between the two columns will be the input into the aforementioned function in reserve.
        size = randint(1, 115)
        
        input_ = list(range(0, size, 1))
        col_1 = list(map(lambda x: x + 1, input_))
        col_2 = list(reversed(col_1))
        
        col_1 = np.array(col_1)
        col_2 = np.array(col_2)
        data = np.column_stack((col_1, col_2))
        data = data.astype(dtype = np.float16)

        no_columns = data.shape[1]
        col_names = ['Series_' + str(i + 1) for i in range(no_columns)]
        
        df = pd.DataFrame(data = data, columns = col_names)
        arr_neut = cross_neutral(df, 'median', sequential = True)
        col_dif = np.subtract(arr_neut[:, 1], arr_neut[:, 0])

        input_ = np.array(input_, dtype = np.float16)
        input_rev = input_[::-1] ## Reverse the input using slicing.

        self.assertTrue(np.all(col_dif == input_rev))

        ## Test the column-wise Mean (using the entire realised return series).
        ## Again, construct a two-dimensional Array where the second column is computed by multiplying the elements in the first column by a factor of ten.
        col1 = np.linspace(1, 21, 21, dtype = np.float16)
        shuffle(col_1)
        col2 = col1 * 10
        stack_col = np.column_stack((col1, col2))
        df = pd.DataFrame(data = stack_col, columns = ['Series_1', 'Series_2'])
        
        arr_neutral = cross_neutral(df, neutral = 'mean', sequential = False)
        self.assertTrue(np.all(arr_neutral[:, 0] == 11.0))
        self.assertTrue(np.all(arr_neutral[:, 0] == arr_neutral[:, 1] / 10))

    def test_nan_insert(self):
        ## Testing the Minimum Observations.
        arr_d = np.zeros((40, 4), dtype = object)
        
        data = np.linspace(1, 40, 40, dtype = np.float32)
        shuffle(data)
        data = data.reshape((8, 5))
        data[0:3, 2] = np.nan
        data[0, 0] = np.nan
        data[0:4, 4] = np.nan
        
        arr_d[:, 3] = np.ravel(data, order = 'F')
        extend_cids = ['AUD', 'CAD', 'FRA', 'GBP', 'USD']
        arr_d[:, 0] = np.array(sorted(extend_cids * 8))

        arr_d[:, 1] = np.repeat('XR', 40)
        dates = pd.date_range(start = "2020-01-01", periods = 8, freq = 'd')
        arr_d[:, 2] = np.array(list(dates) * 5)
        contrived_df = pd.DataFrame(data = arr_d, columns = ['cid', 'xcat', 'real_date', 'value'])
        dfw = contrived_df.pivot(index = 'real_date', columns = 'cid', values = 'value')
        min_obs = 3
        dfw_zns = nan_insert(dfw, min_obs)
        
        ## Numpy comparison.
        df_copy = dfw.copy()
        nan_arr = np.isnan(data)
        indices = np.where(nan_arr == False)
        indices_d = tuple(zip(indices[1], indices[0]))
        indices_dict = defaultdict(list)
        for tup in indices_d:
            indices_dict[tup[0]].append(tup[1])

        for k, v in indices_dict.items():
            df_copy.iloc[:, k][v[0]:(v[0] + min_obs)] = np.nan

        test = (df_copy.fillna(0) == dfw_zns.fillna(0)).to_numpy()
        self.assertTrue(np.all(test))
        

    ## Unable to test the dimensions of the DataFrame input being equal to the output due to the use of pd.unstack(): retains previously added NaNs during the pivot.
    def test_zn_scores(self):

        ## Using the globally defined DataFrame.
        with self.assertRaises(AssertionError):
            df = make_zn_scores(dfd, 'XR', cids, start = None, end = None, sequential = False, neutral = 'std', thresh = 1.5,
                                postfix = 'ZN')
        with self.assertRaises(AssertionError):
            df = make_zn_scores(dfd, 'XR', cids, start = None, end = None, sequential = False, neutral = 'std', thresh = 0.5,
                                pan_weight = 1.0, postfix = 'ZN')

        with self.assertRaises(AssertionError):
            df = make_zn_scores(dfd, 'XR', cids, start = None, end = None, sequential = False, neutral = 'std', thresh = 1.5,
                                pan_weight = 1.2, postfix = 'ZN')


        
            
        ## Test the Zn_Score, with a Panel Weighting of one, using the Mean for the neutral parameter.
        val = randint(1, 39)
        data = np.linspace(-val, val, (val * 2) + 1, dtype = np.float16)
        mean = sum(data) / len(data)
        col1 = data[-val:]
        col2 = data[:val]
        col2 = col2[::-1] ## Reverse the data series to reflect the linear, negative correlation.
        data = np.concatenate((col1, col2))
        ## The two series are uniformally distributed around the panel mean.
        ## Therefore, the evolving standard deviation will grow at a constant rate, 0.5 increment, to reflect the negative linear correlation between the two return series.

        data_col = np.column_stack((col1, col2))

        arr_d = np.zeros((len(data), 4), dtype = object)
        arr_d[:, 3] = data

        aud = np.repeat('AUD', len(col1))
        cad = np.repeat('CAD', len(col1))
        arr_d[:, 0] = np.concatenate((aud, cad))
        dates = pd.date_range(start = "2020-01-01", periods = len(data) / 2, freq = 'd')
        arr_d[:, 2] = np.array(list(dates) * 2)
        arr_d[:, 1] = np.repeat('XR', len(data))

        ## The panel mean will equal zero.
        ## The Standard Deviation Array will be a one-dimensional Array given the statistic is computed across all cross-sections.
        end = (len(col1) / 2) + 0.5
        std = np.linspace(1, end, len(col1)) ## f(x) = y.
        std = std[:, np.newaxis]
        
        rational = np.divide((data_col - mean), std)

        cids_ = ['AUD', 'CAD']
        contrived_df = pd.DataFrame(data = arr_d, columns = ['cid', 'xcat', 'real_date', 'value'])

        df = make_zn_scores(contrived_df, 'XR', cids_, sequential = False, min_obs = 0, neutral = 'mean',
                            pan_weight = 1.0)

        check_val = np.concatenate((rational[:, 0], rational[:, 1]))
        zn_score_algo = df['value'].to_numpy()
        self.assertTrue(np.all(check_val == zn_score_algo))
        

if __name__ == '__main__':

    unittest.main()
