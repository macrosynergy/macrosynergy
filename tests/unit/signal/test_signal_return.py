

import unittest
from macrosynergy.signal.signal_return import SignalReturnRelations

from tests.simulate import make_qdf
from sklearn.metrics import accuracy_score, precision_score
from scipy import stats
import random
import pandas as pd
import numpy as np


class TestAll(unittest.TestCase):

    def dataframe_generator(self):
        """
        Create a standardised DataFrame defined over the three categories.
        """

        self.__dict__['cids'] = ['AUD', 'CAD', 'GBP', 'NZD', 'USD']
        self.__dict__['xcats'] = ['XR', 'CRY', 'GROWTH', 'INFL']

        df_cids = pd.DataFrame(index=self.cids, columns=['earliest', 'latest',
                                                         'mean_add', 'sd_mult'])

        # Purposefully choose a different start date for all cross-sections. Used to test
        # communal sampling.
        df_cids.loc['AUD'] = ['2011-01-01', '2020-12-31', 0, 1]
        df_cids.loc['CAD'] = ['2009-01-01', '2020-10-30', 0, 2]
        df_cids.loc['GBP'] = ['2010-01-01', '2020-08-30', 0, 5]
        df_cids.loc['NZD'] = ['2008-01-01', '2020-06-30', 0, 3]
        df_cids.loc['USD'] = ['2012-01-01', '2020-12-31', 0, 4]

        df_xcats = pd.DataFrame(index=self.xcats, columns=['earliest', 'latest',
                                                           'mean_add', 'sd_mult',
                                                           'ar_coef', 'back_coef'])

        df_xcats.loc['XR'] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
        df_xcats.loc['CRY'] = ['2000-01-01', '2020-10-30', 0, 2, 0.95, 1]
        df_xcats.loc['GROWTH'] = ['2001-01-01', '2020-10-30', 0, 2, 0.9, 1]
        df_xcats.loc['INFL'] = ['2001-01-01', '2020-10-30', 0, 2, 0.8, 0.5]

        random.seed(2)
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

        self.__dict__['dfd'] = dfd

        black = {'AUD': ['2000-01-01', '2003-12-31'],
                 'GBP': ['2018-01-01', '2100-01-01']}

        self.__dict__['blacklist'] = black

        assert 'dfd' in vars(self).keys(), "Instantiation of DataFrame missing from " \
                                           "field dictionary."

    def test_constructor(self):

        self.dataframe_generator()
        # Test the Class's constructor.

        # First, test the assertions.
        # Trivial test to confirm the primary signal must be present in the passed
        # DataFrame.
        with self.assertRaises(AssertionError):

            srr = SignalReturnRelations(self.dfd, ret='XR', sig="Missing",
                                        freq='D', blacklist=self.blacklist)

        signal = 'CRY'
        srr = SignalReturnRelations(self.dfd, ret='XR', sig=signal,
                                    freq='D', blacklist=self.blacklist)

        # The signal will invariably be used as the explanatory variable and the return
        # as the dependent variable.
        # Confirm that the signal is lagged after applying categories_df().

        # Choose an arbitrary date and confirm that the signal in the original DataFrame
        # has been lagged by a day. Confirm on multiple cross-sections: AUD & USD.
        df_signal = self.dfd[self.dfd['xcat'] == signal]
        arbitrary_date_one = '2011-01-10'
        arbitrary_date_two = '2020-10-27'

        test_aud = df_signal[df_signal['real_date'] == arbitrary_date_one]
        test_aud = test_aud[test_aud['cid'] == 'AUD']['value']

        test_usd = df_signal[df_signal['real_date'] == arbitrary_date_two]
        test_usd = test_usd[test_usd['cid'] == 'USD']['value']

        lagged_df = srr.df
        aud_lagged = lagged_df.loc['AUD', signal]['2011-01-11']
        condition = round(float(test_aud), 5) - round(aud_lagged, 5)
        self.assertTrue(abs(condition) < 0.0001)

        usd_lagged = lagged_df.loc['USD', signal]['2020-10-28']
        condition = round(float(test_usd), 5) - round(usd_lagged, 5)
        self.assertTrue(condition < 0.0001)

        # In addition to the DataFrame returned by categories_df(), an instance of the
        # Class will hold two "tables" for each segmentation type.
        # Confirm the indices are the expected: cross-sections or years.
        test_index = list(srr.df_cs.index)[3:]
        self.assertTrue(sorted(self.cids) == sorted(test_index))

    def test_constructor_multiple_sigs(self):

        self.dataframe_generator()
        # The signal return Class allows for additional signals to be passed, upon
        # instantiation, to understand the primary signal's performance relative to other
        # possible signals. The analysis will be completed on the panel level.

        # First, test the assertions.
        # Any additional signals passed must either be a string or list.
        with self.assertRaises(AssertionError):
            srr = SignalReturnRelations(self.dfd, ret='XR', sig='CRY',
                                        rival_sigs=set(['GROWTH', 'INFL']), freq='D',
                                        blacklist=self.blacklist)
        # Signals passed must be a subset of the categories defined in the DataFrame. If
        # not, will raise an assertion.
        with self.assertRaises(AssertionError):
            # GDP is not a defined category.
            srr = SignalReturnRelations(self.dfd, ret='XR', sig='CRY',
                                        rival_sigs=set(['GROWTH', 'INFL', 'GDP']),
                                        freq='D', blacklist=self.blacklist)

        primary_signal = 'CRY'
        rival_signals = ['GROWTH', 'INFL']
        srr = SignalReturnRelations(self.dfd, ret='XR', sig=primary_signal,
                                    rival_sigs=rival_signals, sig_neg=False, freq='D',
                                    blacklist=self.blacklist)
        # First, confirm the signal list stored on the instance comprises both the
        # primary signal and the rival signals.
        self.assertTrue([primary_signal] + rival_signals == srr.signals)

        # Secondly, confirm the DataFrame is defined over the expected columns.
        self.assertTrue(list(srr.df.columns) ==
                        [primary_signal] + rival_signals + ['XR'])

        # Test the negative conversion if the parameter 'sig_neg' is set to True. If set
        # to True, the signal fields will have a postfix, '_NEG', appended.
        srr_neg = SignalReturnRelations(self.dfd, ret='XR', sig=primary_signal,
                                        rival_sigs=rival_signals, sig_neg=True,
                                        blacklist=self.blacklist, freq='D')

        # Firstly, confirm the signal names have been updated correctly.
        srr_signals = list(map(lambda s: s.split('_'), srr_neg.signals))
        signals = [primary_signal] + rival_signals
        for i, s in enumerate(srr_signals):
            self.assertTrue(s[1] == 'NEG')
            self.assertTrue(s[0] == signals[i])

        self.assertEqual(srr_neg.sig[-4:], '_NEG')
        self.assertEqual(srr_neg.sig[:-4], primary_signal)

        # Secondly, confirm the actual DataFrame's columns have been updated.
        test_columns = list(srr_neg.df.columns)
        # Confirms the update has been made on the DataFrame level.
        self.assertTrue(test_columns == srr_neg.signals + ['XR'])

        # Lastly, check the original values have been multiplied by minus one. Therefore,
        # add the two DataFrames which should equate to zero. The multiplication by minus
        # one only occurs on the signals.

        srr_neg.df.rename(columns=dict(zip(srr_neg.signals, srr.signals)), inplace=True)
        zero_df = srr.df + srr_neg.df
        zero_df_sigs = zero_df.loc[:, srr.signals]
        sum_columns = zero_df_sigs.sum(axis=0)
        self.assertTrue(np.all(sum_columns.to_numpy() == 0.0))

    def test_constructor_communal(self):

        self.dataframe_generator()

        # Used to test the communal sample period by setting the parameter equal to True.
        # The DataFrame instantiated on the instance is a multi-index DataFrame where the
        # outer index will be qualified by the available cross-sections and the interior
        # index will be timestamps. The columns will be the respective signals plus
        # return.
        # If the parameter is set to True, the individual cross-section's start dates
        # will be aligned across the panels: the proposed start date will have a
        # realised value for all categories for that specific cross-section. The logic is
        # applied to each cross-section.

        primary_signal = "CRY"
        rival_signals = ["GROWTH", "INFL"]
        # Set "cosp" equal to True.
        srr_cosp = SignalReturnRelations(self.dfd, ret="XR", sig=primary_signal,
                                         rival_sigs=rival_signals, sig_neg=False,
                                         cosp=True, freq="D", blacklist=None)

        # The start date for the communal series should be:
        # start_dates = {'AUD': '2011-01-01', 'CAD': '2009-01-01', 'GBP': '2010-01-01',
        #                'NZD': '2008-01-01', 'USD': '2012-01-01'}

        df = srr_cosp.df
        # Test across all cross-sections - aligned on the cross-section's intersection.
        # Account for weekends.
        expected_date = {'AUD': '2011-01-03', 'CAD': '2009-01-01', 'GBP': '2010-01-01',
                         'NZD': '2008-01-01', 'USD': '2012-01-02'}
        for c, cid_df in df.groupby(level=0):
            # Isolate the interior index.
            series_s_date = str(cid_df.iloc[0, :].name[1]).split(' ')[0]
            self.assertEqual(expected_date[c], series_s_date)

        # Test the values.
        dfd = srr_cosp.dfd
        filt_1 = (dfd['real_date'] == "2011-01-04") & (dfd['xcat'] == "XR")
        dfd_filt = dfd[filt_1]
        benchmark_value = float(dfd_filt[dfd_filt["cid"] == "AUD"]["value"])
        benchmark_value = round(benchmark_value, 5)

        test_row = srr_cosp.df.loc['AUD'].loc["2011-01-04"]
        condition = abs(benchmark_value - round(test_row["XR"], 5)) < 0.00001
        self.assertTrue(condition)

        # Account for lagging the signals. Therefore, the signal values will reference
        # the previous day.
        filt_2 = (dfd['real_date'] == "2011-01-03") & (dfd["cid"] == "AUD")
        dfd_filt = dfd[filt_2]
        signals = ([primary_signal] + rival_signals)

        for s in signals:
            test_value = float(dfd_filt[dfd_filt["xcat"] == s]["value"])
            test_value = round(test_value, 5)
            condition = abs(test_value - round(test_row[s], 5)) < 0.00001
            self.assertTrue(condition)

        # Confirm the dimensions of the return column remains unchanged - alignment
        # occurs exclusively on the signals. To test, confirm the columns are the same
        # regardless of whether communal sampling is applied.
        srr = SignalReturnRelations(self.dfd, ret="XR", sig=primary_signal,
                                    rival_sigs=rival_signals, sig_neg=False, cosp=False,
                                    freq="D", blacklist=None)
        self.assertTrue(srr.df.loc[:, srr.ret].shape == srr_cosp.df.loc[:, srr.ret].shape)

    def test__slice_df__(self):

        self.dataframe_generator()
        # Method used to confirm that the segmentation of the original DataFrame is
        # being applied correctly: either cross-sectional or yearly basis. Therefore, if
        # a specific "cs" is passed, will the DataFrame be reduced correctly ?

        signal = 'CRY'
        srr = SignalReturnRelations(self.dfd, sig=signal, ret='XR',
                                    freq='D', blacklist=self.blacklist)
        df = srr.df.dropna(how='any')

        # First, test cross-sectional basis.
        # Choose a "random" cross-section.

        df_cs = srr.__slice_df__(df=df, cs='GBP', cs_type='cids')

        # Test the values on a fixed date.
        fixed_date = '2010-01-04'
        test_values = dict(df.loc['GBP', fixed_date])
        segment_values = dict(df_cs.loc[fixed_date, :])

        for c, v in test_values.items():
            self.assertTrue(v - segment_values[c] < 0.0001)

        # Test the yearly segmentation.
        df['year'] = np.array(df.reset_index(level=1)['real_date'].dt.year)
        df_cs = srr.__slice_df__(df=df, cs='2013', cs_type='years')

        # Confirm that the year column contains exclusively '2013'. If so, able to deduce
        # that the segmentation works correctly for yearly type.
        df_cs_year = df_cs['year'].to_numpy()
        df_cs_year = np.array(list(map(lambda y: str(y), df_cs_year)))
        self.assertTrue(np.all(df_cs_year == '2013'))

    def test__output_table__(self):

        self.dataframe_generator()
        # Test the method responsible for producing the table of metrics assessing the
        # signal-return relationship.

        # Firstly, for the six "Positive Ratio" statistics, confirm the computed value
        # for the accuracy score is correct. If so, able to conclude the other scores
        # are being assembled in the returned table correctly.
        signal = 'CRY'
        return_ = 'XR'
        srr = SignalReturnRelations(self.dfd, sig=signal, ret=return_,
                                    freq='D', blacklist=self.blacklist)
        df_cs = srr.__output_table__(cs_type='cids')

        # The lagged signal & returns have been reduced to[-1, 1] which are interpreted
        # as indicator random variables.
        # Test value.
        df_cs_aud_acc = df_cs.loc['AUD', 'accuracy']

        # Accounts for removal of dropna() from categories_df() function.
        srr_df = srr.df.dropna(axis=0, how='any')

        aud_df = srr_df.loc['AUD', :]
        # Remove zero values.
        aud_df = aud_df[~((aud_df.iloc[:, 0] == 0) | (aud_df.iloc[:, 1] == 0))]
        # In the context of the accuracy score, reducing the DataFrame to boolean values
        # will work equivalently to [1, -1].
        aud_df = aud_df > 0
        y_pred = aud_df[signal]
        y_true = aud_df[return_]
        accuracy = accuracy_score(y_true, y_pred)
        self.assertTrue(abs(df_cs_aud_acc - accuracy) < 0.00001)

        # Aim to test Kendall Tau correlation statistic via stats.rankdata() - Kendall
        # Tau is implicitly ranking the data but using the original values.
        # Kendall Tau is non-parametric, and both the return & signal series will be
        # used as quasi-ranking data.
        df_cs_usd_ken = df_cs.loc['USD', 'kendall']

        usd_df = srr_df.loc['USD', :]
        usd_df = usd_df[~((usd_df.iloc[:, 0] == 0) | (usd_df.iloc[:, 1] == 0))]
        x = stats.rankdata(usd_df[signal]).astype(int)
        y = stats.rankdata(usd_df[return_]).astype(int)

        kendall_tau, p_value = stats.kendalltau(x, y)
        # Kendall Tau offers value when used in conjunction with Pearson's
        # correlation coefficient which is a linear measure.
        # For instance, if the Pearson correlation coefficient is close to zero but
        # the Kendall Tau is close to one, it can be deduced that there is a
        # relationship between the two variables but a non-linear relationship.
        # Alternatively, if the Pearson coefficient is close to one but the Kendall
        # Tau is closer to zero, it suggests that the sample is exposed to a small
        # number of sharp outliers.
        self.assertTrue(abs(df_cs_usd_ken - kendall_tau) < 0.00001)

        # Test the linear correlation measure, Pearson correlation coefficient.
        df_cs_usd_pearson = df_cs.loc['USD', 'pearson']
        sig_ret_cov_matrix = np.cov(usd_df[signal], usd_df[return_])
        sig_ret_cov = sig_ret_cov_matrix[0, 1]

        # Covariance divided by the product of the variance.
        manual_calc = sig_ret_cov / (np.std(usd_df[signal]) * np.std(usd_df[return_]))

        self.assertTrue(abs(df_cs_usd_pearson - manual_calc) < 0.0001)

        # Test the precision score which will record the signal's positive bias. This is
        # important because the greater the number of false positives, the more exposed a
        # strategy is and subsequently any gains, through true positives, will be
        # reversed.
        positive_signals_index = usd_df[signal] > 0
        positive_signals = usd_df[signal][positive_signals_index]
        positive_signals = np.sign(positive_signals)

        negative_signals = usd_df[signal][~positive_signals_index]
        negative_signals = np.sign(negative_signals)

        return_series_pos = np.sign(usd_df[return_][positive_signals_index])
        return_series_neg = np.sign(usd_df[return_][~positive_signals_index])

        positive_accuracy = accuracy_score(
            positive_signals, return_series_pos
        )
        negative_accuracy = accuracy_score(
            negative_signals, return_series_neg
        )

        manual_precision = (positive_accuracy + (1 - negative_accuracy)) / 2
        df_cs_usd_posprec = df_cs.loc['USD', 'pos_prec']
        
        self.assertTrue(abs(manual_precision - df_cs_usd_posprec) < 0.1)

        # Lastly, confirm that 'Mean' row is computed using exclusively the respective
        # segmentation types. Test on yearly data and balanced accuracy.
        df_ys = srr.__output_table__(cs_type='years')
        df_ys_mean = df_ys.loc['Mean', 'bal_accuracy']

        dfx = df_ys[~df_ys.index.isin(['Panel', 'Mean', 'PosRatio'])]
        dfx_balance = dfx['bal_accuracy']
        condition = np.abs(np.mean(dfx_balance) - df_ys_mean)
        self.assertTrue(condition < 0.00001)

    def test__rival_sigs__(self):

        self.dataframe_generator()
        # Method is used to produce the metric table for the secondary signals. The
        # analysis will be completed on the panel level.

        # Test the construction of the table is correct and the values include all
        # cross-sections.
        primary_signal = "CRY"
        rival_signals = ["GROWTH", "INFL"]
        srr = SignalReturnRelations(self.dfd, ret="XR", sig=primary_signal,
                                    rival_sigs=rival_signals, sig_neg=False, freq="D",
                                    blacklist=self.blacklist)

        df_sigs = srr.__rival_sigs__()

        # Firstly, confirm that the index consists of only the primary and rival signals.
        self.assertEqual(list(df_sigs.index), [primary_signal] + rival_signals)

        # Secondly, test the actual calculation on a single signal. Test the accuracy
        # score. If correct, all metrics should be correct.
        growth_accuracy = df_sigs.loc["GROWTH", "accuracy"]

        test_df = srr.df.loc[:, ["GROWTH", "XR"]]
        test_df = test_df.dropna(axis=0, how='any')
        df_sgs = np.sign(test_df)
        manual_value = accuracy_score(df_sgs["GROWTH"], df_sgs["XR"])
        self.assertEqual(growth_accuracy, manual_value)

    def test_signals_table(self):

        self.dataframe_generator()
        # If defined, will return the panel-level table for the rival signals. The method
        # receives a single parameter, "sigs", whose default is set to None (all
        # available signals are returned).

        primary_signal = "CRY"
        rival_signals = ["GROWTH", "INFL"]
        srr = SignalReturnRelations(self.dfd, ret="XR", sig=primary_signal,
                                    rival_sigs=rival_signals, sig_neg=False, freq="M",
                                    blacklist=self.blacklist)
        # Firstly, confirm that if the parameter 'sigs' is left undefined, all signals
        # will be displayed in the table (default setting).
        df_sigs = srr.signals_table()
        self.assertEqual(list(df_sigs.index), [primary_signal] + rival_signals)

        # Secondly, confirm that a specific subset of the signals can be displayed.
        df_sigs = srr.signals_table(sigs=["CRY", "INFL"])
        self.assertEqual(list(df_sigs.index), [primary_signal] + ["INFL"])

        # Confirm that a list of signals must be passed. The method is for comparison
        # purposes. Therefore, at a minimum, it expects to receive the primary signal
        # plus an additional rival signal.
        with self.assertRaises(AssertionError):
            # Pass in a single signal as a string.
            df_sigs = srr.signals_table(sigs="INFL")

        # Lastly, confirm that an AttributeError will be thrown if the method is called
        # but additional signals have not been passed to the instance.
        srr = SignalReturnRelations(self.dfd, ret="XR", sig=primary_signal,
                                    rival_sigs=None, sig_neg=False, freq="M",
                                    blacklist=self.blacklist)
        with self.assertRaises(AttributeError):
            df_sigs = srr.signals_table()

    def test__yaxis_lim__(self):

        self.dataframe_generator()

        signal = 'CRY'
        return_ = 'XR'
        srr = SignalReturnRelations(self.dfd, sig=signal, ret=return_,
                                    freq='D', blacklist=self.blacklist)
        df_cs = srr.__output_table__(cs_type='cids')
        dfx = df_cs[~df_cs.index.isin(['PosRatio'])]
        dfx_acc = dfx.loc[:, ['accuracy', 'bal_accuracy']]
        arr_acc = dfx_acc.to_numpy()
        arr_acc = arr_acc.flatten()

        # Flatten the array - only concerned with the minimum across both dimensions. If
        # the minimum value is less than 0.45, use the minimum value to initiate the
        # range. Test the above logic.
        ylim = srr.__yaxis_lim__(accuracy_df=dfx_acc)

        min_value = min(arr_acc)
        if min_value < 0.45:
            self.assertTrue(ylim == min_value)
        else:
            self.assertTrue(ylim == 0.45)


if __name__ == "__main__":

    unittest.main()