
import unittest
import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
from tests.simulate import make_qdf
from macrosynergy.panel.return_beta import *
from macrosynergy.management.shape_dfs import reduce_df
import math

class TestAll(unittest.TestCase):

    def dataframe_construction(self):

        # Emerging Market Asian countries.
        cids = ['IDR', 'INR', 'KRW', 'MYR', 'PHP']
        # Add the US - used as the hedging asset.
        cids += ['USD']

        self.__dict__['cids'] = cids
        xcats = ['FXXR_NSA', 'GROWTHXR_NSA', 'INFLXR_NSA', 'EQXR_NSA']
        self.__dict__['xcats'] = xcats

        df_cids = pd.DataFrame(index=self.cids, columns=['earliest', 'latest', 'mean_add',
                                                         'sd_mult'])

        df_cids.loc['IDR'] = ['2010-01-01', '2020-12-31', 0.5, 2]
        df_cids.loc['INR'] = ['2011-01-01', '2020-11-30', 0, 1]
        df_cids.loc['KRW'] = ['2012-01-01', '2020-11-30', -0.2, 0.5]
        df_cids.loc['MYR'] = ['2013-01-01', '2020-09-30', -0.2, 0.5]
        df_cids.loc['PHP'] = ['2002-01-01', '2020-09-30', -0.1, 2]
        df_cids.loc['USD'] = ['2000-01-01', '2020-03-20', 0, 1.25]

        df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                      'sd_mult', 'ar_coef', 'back_coef'])

        df_xcats.loc['FXXR_NSA'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
        df_xcats.loc['GROWTHXR_NSA'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
        df_xcats.loc['INFLXR_NSA'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]
        df_xcats.loc['EQXR_NSA'] = ['2000-01-01', '2022-03-14', 0.5, 2, 0, 0.2]

        # If the asset being used as the hedge experiences a blackout period, then it is
        # probably not an appropriate asset to use in the hedging strategy.
        black = {'IDR': ['2010-01-01', '2012-01-04'],
                 'INR': ['2010-01-01', '2013-12-31'],
                 }
        self.__dict__['blacklist'] = black

        # Standard df for tests.
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.__dict__['dfd'] = dfd

        # The Unit Test will be based on the hedging strategy: hedge FX returns
        # (FXXR_NSA) against US Equity, S&P 500, (USD_EQXR_NSA).
        cid_hedge = 'USD'
        xcat_hedge = 'EQXR_NSA'
        self.__dict__['benchmark_df'] = reduce_df(dfd, xcats=[xcat_hedge],
                                                  cids=cid_hedge)

        self.__dict__["unhedged_df"] = reduce_df(dfd, xcats=['FXXR_NSA'],
                                                 cids=cids)
        self.__dict__["dfp_w"] = self.unhedged_df.pivot(index='real_date', columns='cid',
                                                        values='value')

    def test_date_alignment(self):
        """
        Firstly, return_beta.py will potentially use a single asset to hedge a panel
        which can consist of multiple cross-sections, and each cross-section could be
        defined over differing time-series. Therefore, the .date_alignment() method is
        used to ensure the asset being used as the hedge and the asset being hedged are
        defined over the same timestamps. The method will return the proposed start &
        end date.
        """

        self.dataframe_construction()

        # Verify that two series passed will be aligned after applying the respective
        # method.
        # Test on MYR_FXXR_NSA against the hedging asset, USD_EQXR_NSA (both are defined
        # over different time horizons).
        c = 'MYR'
        xr = self.dfp_w[c]
        # Adjusts for the effect of pivoting.
        xr = xr.dropna(axis=0, how="all")

        br = pd.Series(data=self.benchmark_df['value'],
                       index=self.benchmark_df['real_date'])

        start_date, end_date = date_alignment(unhedged_return=xr,
                                              benchmark_return=br)
        # The latest start date of the two pd.Series.
        target_start = '2013-01-01'
        start_date = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        self.assertTrue(start_date == target_start)

        end_date = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        target_end = '2020-03-20'
        self.assertTrue(end_date == target_end)

    def test_hedge_calculator(self):
        """
        Method designed to calculate the hedge ratios used across the panel: each cross-
        section in the panel will have a different sensitivity parameter relative to the
        benchmark.
        Further, the frequency in which the hedge ratios are calculated is delimited by
        the 'refreq' parameter. The sample size of data, number of dates used in the
        re-estimation, will increase at a rate controlled by the parameter: the hedge
        ratio will be continuously re-estimated as days pass but will always include all
        realised timestamps (inclusive of the start_date).
        """

        self.dataframe_construction()

        # The method returns a standardised DataFrame. Confirm the first date in the
        # DataFrame is after the minimum observation date. The parameter 'min_obs'
        # ensures a certain number of days have passed until a hedge ratio is calculated.

        # Analysis completed using a single cross-section from the panel.
        c = 'KRW'
        xr = self.dfp_w[c]
        xr = xr.astype(dtype=np.float16)
        # Adjusts for the effect of pivoting.
        xr = xr.dropna(axis=0, how="all")

        br = pd.Series(data=self.benchmark_df['value'].to_numpy(),
                       index=self.benchmark_df['real_date'])
        br = br.astype(dtype=np.float16)

        # Apply the .date_alignment() method to establish the start & end date of the
        # re-estimation date series. Confirms the re-estimation frequency has been
        # correctly applied.
        # The frequency tested on will be monthly: business month end frequency.
        start_date, end_date = date_alignment(unhedged_return=xr, benchmark_return=br)
        dates_re = pd.date_range(start=start_date, end=end_date, freq='BM')
        str_time = lambda date: pd.Timestamp(str(date))
        dates_re = list(map(str_time, dates_re))

        min_observation = 50
        # Produce daily business day date series to determine the date that corresponds
        # to the specified minimum observation.
        daily_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        test_min_obs = daily_dates.to_numpy()[60]

        df_hr = hedge_calculator(unhedged_return=xr, benchmark_return=br,
                                 rdates=dates_re, cross_section=c, meth='ols',
                                 min_obs=min_observation)
        # Confirm the first computed hedge ratio value falls after the minimum
        # observation date.
        test_date = df_hr['real_date'].iloc[0]
        self.assertTrue(test_date > test_min_obs)

        # In the example, the hedge ratio is computed monthly - last business day of the
        # respective month. Therefore, assuming the minimum observation date does not
        # fall on the final day of the month, the first date recorded in the returned
        # DataFrame should be the final business date of the same month as the minimum
        # observation date. For instance, 23/03/2010 -> 30/03/2010.
        # Test both dates are defined during the same month.
        test_min_obs_month = pd.Timestamp(test_min_obs).month
        self.assertTrue(test_min_obs_month == test_date.month)

        # The re-estimated hedge ratio will be computed using realised data up until the
        # respective date (in the above example, the final business day of the month).
        # However, the hedge ratio is applied from the following date (the date the
        # position will change) and to all the intermediary dates up until the next
        # re-estimation date. Therefore, confirm that the first date in the DataFrame is
        # a np.nan value representing the shift mechanism.
        first_value = df_hr['value'].iloc[0]
        self.assertTrue(math.isnan(first_value))

        # The examination of the hedging mechanism will come through graphical
        # interpretation.
        # However, test the computed hedge ratio on a "random" re-estimation date to
        # confirm both return series up until the respective date have been used.
        # Test date is '2013-03-29'.

        s_date, e_date = date_alignment(unhedged_return=xr, benchmark_return=br)
        xr = xr.truncate(before=s_date, after=e_date)
        br = br.truncate(before=s_date, after=e_date)

        X = sm.add_constant(xr.loc[:'2013-03-29'])
        y = br.loc[:'2013-03-29']
        mod = sm.OLS(y, X)
        result = mod.fit().params[1]

        # Test on the next business day given the shift. The hedge ratio computed on the
        # re-estimation date is applied to the return series on the next business day
        # after re-estimation.
        test_value = float(df_hr[df_hr['real_date'] == '2013-04-01']['value'])
        self.assertTrue(result == test_value)

    def test_adjusted_returns(self):
        """
        Method used to compute the hedge ratio returns. The hedge ratio will determine
        the position taken in the benchmark asset. Therefore, adjust the returns across
        the panel to account for the short position taken the hedging asset (proportional
        to the computed sensitivity parameter between the cross-section and the
        benchmark). A simple example of the formula is:
        IDR_FXXR_NSA_H = IDR_FXXR_NSA - HR_IDR * USD_EQXR_NSA.
        """

        self.dataframe_construction()

        br = pd.Series(data=self.benchmark_df['value'].to_numpy(),
                       index=self.benchmark_df['real_date'])
        br = br.astype(dtype=np.float16)

        # The method, adjusted_returns(), will compute the hedged return across the
        # entire panel. Call hedge_ratio method and pass the returned DataFrame
        # separately into adjusted_returns() method.
        # The adjusted_returns() method will be called inside the main hedge_ratio()
        # subroutine if a parameter is set to set to True.

        br_cat = "USD_EQXR_NSA"
        # Standardised dataframe consisting of exclusively the hedge-ratios.
        df_hedge = return_beta(df=self.dfd, xcat='FXXR_NSA', cids=self.cids,
                               benchmark_return=br_cat, start='2010-01-01',
                               end='2020-10-30', blacklist=self.blacklist, meth='ols',
                               oos=True, refreq='m', min_obs=60,
                               hedged_returns=False)

        dfw = self.unhedged_df.pivot(index='real_date', columns='cid', values='value')

        # Standardised dataframe of the adjusted returns.
        df_stack = adjusted_returns(benchmark_return=br, df_hedge=df_hedge, dfw=dfw)

        # Choose a "random" date and confirm the values of two cross-sections through
        # manuel calculation.
        # "Random" date is "2016-06-01"
        dates = list(dfw.index)
        date = dfw.index[len(dates) // 2]

        test_date = df_stack[df_stack['real_date'] == date]
        # Test on the two cross-sections: 'IDR' & 'INR'.
        # Hedge Return.
        INR_HR = float(test_date[test_date['cid'] == 'INR']['value'])
        IDR_HR = float(test_date[test_date['cid'] == 'IDR']['value'])

        hedge_row = df_hedge[df_hedge['real_date'] == date]
        INR_H = float(hedge_row[hedge_row['cid'] == 'INR']['value'])
        IDR_H = float(hedge_row[hedge_row['cid'] == 'IDR']['value'])

        return_row = dfw.loc[date]
        INR_R = return_row['INR']
        IDR_R = return_row['IDR']

        br_date = br.loc[date]

        # Manual calculation.
        INR_return = INR_R - (INR_H * br_date)
        IDR_return = IDR_R - (IDR_H * br_date)

        self.assertTrue(INR_return == INR_HR)
        self.assertTrue(IDR_return == IDR_HR)

    def test_hedge_ratio(self):
        """
        Estimates hedge ratios with respect to a hedge benchmark. The subroutine also
        allows for returning hedged returns if the respective parameter is set to True.
        The method will primarily test the workflow of the function: the logic & source
        code have been covered by previous Unit Tests.
        As stated, the main function of the method is to test the efficacy of the
        assert statements included and the workflow of the main driver function.
        """

        self.dataframe_construction()
        br_cat = "USD_EQXR_NSA"

        with self.assertRaises(AssertionError):
            # The categories the respective DataFrame is defined over are
            # ['FXXR_NSA', 'GROWTHXR_NSA', 'INFLXR_NSA', 'EQXR_NSA']. Therefore, choosing
            # a benchmark of USD intuitive GDP growth will throw an error given the
            # ticker is not in the database.
            test_br = "USD_INTRGDP_NSA"
            df_hedge = return_beta(df=self.dfd, xcat='FXXR_NSA', cids=self.cids,
                                   benchmark_return=test_br, start='2010-01-01',
                                   end='2020-10-30', blacklist=self.blacklist,
                                   meth='ols',
                                   oos=True, refreq='m', min_obs=60,
                                   hedged_returns=False)

        # Test the re-estimation frequency parameter.
        with self.assertRaises(AssertionError):
            # The re-estimation frequency can either be weekly, monthly or quarterly:
            # ['w', 'm', 'q']. Set the 'refreq' parameter to an incorrect value.
            df_hedge = return_beta(df=self.dfd, xcat='FXXR_NSA', cids=self.cids,
                                   benchmark_return=br_cat, start='2010-01-01',
                                   end='2020-10-30', blacklist=self.blacklist,
                                   meth='ols',
                                   oos=True, refreq='b', min_obs=60,
                                   hedged_returns=False)

        # The default number of minimum observations required to compute a hedge ratio is
        # 24. However, if the parameter is defined, the specified number must be greater
        # than 10 business days, two weeks.
        with self.assertRaises(AssertionError):
            df_hedge = return_beta(df=self.dfd, xcat='FXXR_NSA', cids=self.cids,
                                   benchmark_return=br_cat, start='2010-01-01',
                                   end='2020-10-30', blacklist=self.blacklist,
                                   meth='ols',
                                   oos=True, refreq='w', min_obs=8,
                                   hedged_returns=False)

        # Confirm the re-estimation frequency parameter is working correctly. Test on
        # weekly data where the final day of the week will invariably be the Friday. The
        # new re-estimation value should be applied from the next business day, the
        # following Monday.
        df_hedge = return_beta(df=self.dfd, xcat='FXXR_NSA', cids=self.cids,
                               benchmark_return=br_cat, start='2010-01-01',
                               end='2020-10-30', blacklist=self.blacklist,
                               meth='ols',
                               oos=True, refreq='w', min_obs=24,
                               hedged_returns=False)
        # Confirm on a single cross-section.
        df_hedge_INR = df_hedge[df_hedge['cid'] == 'INR']

        # Test on a random date, 2014-02-14. The date should be a Friday.
        date = pd.Timestamp('2014-02-14')
        self.assertTrue(pd.Timestamp(date).dayofweek == 4)
        df_hedge_INR_val = (df_hedge_INR[df_hedge_INR['real_date'] == date])['value']
        df_hedge_INR_val = float(df_hedge_INR_val)

        # Confirm the date in the DataFrame is a Monday and the hedge ratio is
        # re-estimated on the respective date.

        index = np.where(df_hedge_INR['real_date'] == date)
        index = next(iter(index))[0]
        next_index = index + 1
        test_row = df_hedge_INR.iloc[next_index]
        test_date = test_row['real_date']

        self.assertTrue(pd.Timestamp(test_date).dayofweek == 0)
        test_value = test_row['value']
        self.assertTrue(test_value != df_hedge_INR_val)


if __name__ == '__main__':

    unittest.main()