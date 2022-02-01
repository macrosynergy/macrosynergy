
import unittest
import numpy as np
import pandas as pd
from tests.simulate import make_qdf
from macrosynergy.panel.panel_calculator import panel_calculator
from random import randint, choice
from typing import List

class TestAll(unittest.TestCase):

    def dataframe_generator(self, date = '2002-01-01'):
        self.__dict__['cids'] = ['AUD', 'CAD', 'GBP', 'NZD', 'USD']
        self.__dict__['xcats'] = ['XR', 'CRY', 'GROWTH', 'INFL']
        df_cids = pd.DataFrame(index=self.cids,
                               columns=['earliest', 'latest', 'mean_add', 'sd_mult'])

        df_cids.loc['AUD'] = ['2000-01-01', '2020-12-31', 0.1, 1]
        df_cids.loc['CAD'] = ['2001-01-01', '2020-11-30', 0, 1]
        df_cids.loc['GBP'] = ['2002-01-01', '2020-11-30', 0, 2]
        df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]
        df_cids.loc['USD'] = [date, '2020-10-30', 0.2, 2]

        df_xcats = pd.DataFrame(index=self.xcats,
                                columns=['earliest', 'latest', 'mean_add', 'sd_mult',
                                         'ar_coef', 'back_coef'])

        df_xcats.loc['XR'] = ['2010-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
        df_xcats.loc['CRY'] = ['2011-01-01', '2020-12-31', 1, 2, 0.95, 1]
        df_xcats.loc['GROWTH'] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 1]
        df_xcats.loc['INFL'] = ['2011-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.__dict__['dfd'] = dfd

        black = {'AUD': ['2021-01-01', '2022-12-31'],
                 'GBP': ['2021-01-01', '2100-01-01']}

        self.__dict__['blacklist'] = black
        self.__dict__['start'] = '2010-01-01'
        self.__dict__['end'] = '2020-12-31'

    @staticmethod
    def dataframe_pivot(df_calc: pd.DataFrame, xcat: str):

        filt = (df_calc['xcat'] == xcat)
        df_new = df_calc[filt].pivot(index='real_date', columns='cid', values='value')
        df_new_trunc = df_new.dropna(axis=0, how="any")

        dates = list(df_new_trunc.index)
        # Further restrictions on possible values.
        date = choice(dates)
        while date > pd.Timestamp("2019-01-01") or date < pd.Timestamp("2014-01-01"):
            date = choice(dates)

        return df_new, date

    def row_value(self, filt_df: pd.DataFrame, date: pd.Timestamp, cid: str,
                  xcats: List[str]):

        values = []
        no_xcats = len(xcats)

        input_values = filt_df[filt_df['real_date'] == date]
        input_values = input_values[input_values['cid'] == cid]
        if no_xcats > 1:
            for cat in xcats:

                val = input_values[input_values['xcat'] == cat]['value']
                values.append(float(val))
            return values
        else:
            val = input_values[input_values['xcat'] == xcats[0]]['value']
            return float(val)

    @staticmethod
    def date_computer(start_date: pd.Timestamp, no_days: int):

        # Adjust for weekend. The original dataframe works exclusively with weekdays.
        condition = start_date.isoweekday() - no_days
        if condition <= 0:
            start_date -= pd.DateOffset(no_days + 2)
        else:
            start_date -= pd.DateOffset(no_days)

        return start_date

    def test_panel_calculator_dimension(self):
        # Function used test the alignment of dataframes if categories are defined over
        # different time-periods. If there are a differing date ranges, NaN values will
        # populate the respective dates that are not shared across each involved
        # category.
        self.dataframe_generator()

        formulas = ["NEW1 = np.abs( XR ) + 0.52 + 2 * CRY",
                    "NEW2 = NEW1 / XR"]
        df_calc = panel_calculator(df=self.dfd, calcs=formulas,
                                   cids=self.cids, start=self.start, end=self.end,
                                   blacklist=self.blacklist)
        # First preliminary check is the presence of both categories.
        cats_df = list(df_calc['xcat'].unique())
        self.assertTrue(cats_df == ["NEW1", "NEW2"])

        # The dimensions of the returned dataframe should match the dimensions of the
        # input dataframe once reduced to the categories involved in the expression.
        filt_1 = (self.dfd['xcat'] == 'XR') | (self.dfd['xcat'] == 'CRY')
        filt_df = self.dfd[filt_1]
        date_column = filt_df['real_date']

        # The produced category, "NEW1", should be defined over the below date-series
        # given the alignment of pandas. It is worth noting that the two series are not
        # defined over the same time-period but pd.DataFrames will match on the index.
        first_date = np.min(date_column)
        end_date = np.max(date_column)
        tuple_ = self.dataframe_pivot(df_calc, "NEW1")
        df_new1 = tuple_[0]
        date_range = list(df_new1.index)

        self.assertTrue(first_date == date_range[0])
        self.assertTrue(end_date == date_range[-1])

        # Test the dimensions on a testcase involving a single cross-section where the
        # cross-section is defined over a reduced time-period. Therefore, the majority of
        # dates in the returned dataframe will be NaN values.
        formula = "NEW1 = GROWTH - iUSD_INFL"
        formulas = [formula]

        # Both GROWTH & INFL have start dates set to "2011-01-01". However, USD's first
        # active trading day is 2015-01-01 which should be reflected in the returned
        # dataframe.
        usd_date = "2015-01-01"
        self.dataframe_generator(date=usd_date)
        df_calc = panel_calculator(df=self.dfd, calcs=formulas,
                                   cids=self.cids, start=self.start, end=self.end,
                                   blacklist=self.blacklist)

        df_calc_new1, date = self.dataframe_pivot(df_calc, xcat="NEW1")
        df_calc_new1 = df_calc_new1.dropna(axis=0, how='all')
        date_index = list(df_calc_new1.index)

        first_date = date_index[0]
        usd_date = pd.Timestamp(usd_date)
        self.assertTrue(usd_date == first_date)

    def test_panel_calculator(self):

        self.dataframe_generator()

        # Test the panel calculator on various testcases and validate that the computed
        # values are correct.

        # i)
        formulas = ["NEW1 = np.abs( XR ) + 0.52 + 2 * CRY",
                    "NEW2 = NEW1 / XR"]
        df_calc = panel_calculator(df=self.dfd, calcs=formulas,
                                   cids=self.cids, start=self.start, end=self.end,
                                   blacklist=self.blacklist)

        filt_1 = (self.dfd['xcat'] == 'XR') | (self.dfd['xcat'] == 'CRY')
        filt_df = self.dfd[filt_1]

        tuple_ = self.dataframe_pivot(df_calc, "NEW1")
        df_new1 = tuple_[0]

        date_series = filt_df['real_date']

        dates = list(date_series[date_series > pd.Timestamp("2013-01-01")])
        date = choice(dates)
        # Test on Australia.
        cross_section = 'AUD'
        computed_value = df_new1.loc[date][cross_section]
        xr, cry = self.row_value(filt_df, date, cross_section, ['XR', 'CRY'])

        # Manually produce: "NEW1 = np.abs( XR ) + 0.52 + 2 * CRY".
        self.assertTrue(computed_value == (np.abs(float(xr)) + 0.52 + 2 * float(cry)))

        # Check NEW2: "NEW2 = NEW1 / XR".
        cross_section = 'USD'
        df_new2, date = self.dataframe_pivot(df_calc, "NEW2")

        computed_value = df_new2.loc[date][cross_section]
        xr = self.row_value(filt_df, date, cross_section, ['XR'])
        new1_val = df_new1.loc[date][cross_section]

        self.assertTrue(computed_value == float(new1_val) / float(xr))

        # ii)
        # Test on the application of multiple numpy functions applied to a single cross-
        # section. Aim is to understand the breadth and durability of the eval() method.
        formula = "NEW1 = GROWTH - INFL"
        formula_2 = "NEW2 = np.square(np.abs( XR ))"
        formulas = [formula, formula_2]
        df_calc = panel_calculator(df=self.dfd, calcs=formulas,
                                   cids=self.cids, start=self.start, end=self.end,
                                   blacklist=self.blacklist)
        # Exclude the rudimentary check the rudimentary formula.
        filt_2 = (self.dfd['xcat'] != 'CRY')
        filt_df = self.dfd[filt_2]

        df_new2, date = self.dataframe_pivot(df_calc, xcat="NEW2")

        cross_section = 'USD'
        computed_value = df_new2.loc[date][cross_section]
        xr = self.row_value(filt_df, date, cross_section, ['XR'])
        xr = float(xr)

        # The formula is contrived but tests the strength of the incorporation of Numpy.
        self.assertTrue(computed_value == np.square(np.abs(xr)))

        # iii)
        # Test on the application of a single cross-section. Applying a binary operation
        # relative to a single cross-section: adjusting returns to US inflationary
        # pressure for example.

        # Adjust macroeconomic growth, across various countries, to US inflationary
        # pressure.
        formula = "NEW1 = ( GROWTH - iUSD_INFL ) / iUSD_XR"
        formulas = [formula]
        df_calc_cross = panel_calculator(df=self.dfd, calcs=formulas,
                                         cids=self.cids, start=self.start, end=self.end,
                                         blacklist=self.blacklist)

        cross_section = "GBP"
        df_new1, date = self.dataframe_pivot(df_calc_cross, xcat="NEW1")
        row_value_gbp = df_new1.loc[date][cross_section]

        xcats = ['GROWTH', 'INFL', 'XR']
        growth = self.row_value(filt_df, date, cross_section, ['GROWTH'])
        cross_section = 'USD'
        infl, xr = self.row_value(filt_df, date, cross_section, ['INFL', 'XR'])

        manual_calculator = (growth - infl) / xr
        self.assertTrue(row_value_gbp == manual_calculator)

    def test_panel_calculator_time_series(self):

        self.dataframe_generator()

        # Test the panel calculator on time-series operations applied to pandas
        # dataframes. For example, percentage change, shifts and other associated
        # methods.
        # Will implicitly test the two accompanying methods which are used when
        # time-series operations are applied.

        formula = "NEW1 = GROWTH.pct_change(periods=1, fill_method='pad') - " \
                  "INFL.pct_change(periods=1, fill_method='pad')"
        formula_2 = "NEW2 = INFL / XR"
        formulas = [formula, formula_2]
        df_calc = panel_calculator(df=self.dfd, calcs=formulas, cids=self.cids,
                                   start=self.start, end=self.end)
        df_new1, date = self.dataframe_pivot(df_calc, "NEW1")
        date_2 = date - pd.DateOffset(1)

        while date_2.isoweekday() > 5:
            date_2 -= pd.DateOffset(1)

        filt_1 = (self.dfd['xcat'] != 'CRY')
        filt_df = self.dfd[filt_1]
        cross_section = 'CAD'
        # To test the percentage change, will require values across consecutive days.
        growth, infl = self.row_value(filt_df, date, cross_section, ['GROWTH', 'INFL'])
        growth_2, infl_2 = self.row_value(filt_df, date_2, cross_section,
                                          ['GROWTH', 'INFL'])

        growth_pct = ((growth - growth_2) / growth_2)
        infl_pct = ((infl - infl_2) / infl_2)
        manual_compute = growth_pct - infl_pct
        row_value_cad = df_new1.loc[date][cross_section]

        self.assertTrue(round(manual_compute, 5) == round(row_value_cad, 5))

        # Validate the second field.
        tuple_ = self.dataframe_pivot(df_calc, "NEW2")
        df_new2 = tuple_[0]

        xr = self.row_value(filt_df, date, cross_section, ['XR'])
        manual_compute = infl / xr

        row_value_cad = df_new2.loc[date][cross_section]
        self.assertTrue(manual_compute == row_value_cad)

        # Secondary test on a different pandas operation. Testing the functionality
        # covers the entire breadth of the sample space.
        formula = "NEW1 = GROWTH.shift(periods=4, freq=None, axis=0)"
        formulas = [formula]
        df_calc = panel_calculator(df=self.dfd, calcs=formulas, cids=self.cids,
                                   start=self.start, end=self.end)
        filt_2 = (self.dfd['xcat'] == 'GROWTH')
        filt_df = self.dfd[filt_2]
        df_new1, date = self.dataframe_pivot(df_calc, "NEW1")

        date_2 = self.date_computer(date, no_days=4)

        growth = self.row_value(filt_df, date_2, cross_section, ['GROWTH'])
        row_value_cad = df_new1.loc[date][cross_section]

        self.assertTrue(round(growth, 5) == round(row_value_cad, 5))


if __name__ == '__main__':

    unittest.main()