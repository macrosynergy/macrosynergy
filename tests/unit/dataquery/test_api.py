

from macrosynergy.dataquery import api
from macrosynergy.dataquery.auth import OAuth, CertAuth
from datetime import datetime
from pandas import Timestamp
from pandas.tseries.offsets import BDay
from typing import List
from unittest import mock
from random import random
import unittest
import numpy as np

class TestDataQueryInterface(unittest.TestCase):

    @staticmethod
    def s_date_calc():
        """
        Dynamic calculation of the start date.
        """
        day = datetime.today().strftime('%Y-%m-%d')
        day = str(day)
        day = Timestamp(day)
        # Subtract a business week from today's date.
        new_date = day - BDay(5)

        return str(new_date.strftime('%Y-%m-%d'))

    @staticmethod
    def jpmaqs_indicators(metrics, tickers):
        """
        Functionality taken from api.Interface(). Will be implicitly tested by methods
        below.
        """

        dq_tix = []
        for metric in metrics:
            dq_tix += ["DB(JPMAQS," + tick + f",{metric})" for tick in tickers]

        return dq_tix

    @staticmethod
    def jpmaqs_value(elem: str):
        """
        Used to produce a value or grade for the associated ticker. If the metric is
        grade, the function will return 1.0 and if value, the function returns a random
        number between (0, 1).

        :param <str> elem: ticker.
        """
        ticker_split = elem.split(',')
        if ticker_split[-1][:-1] == 'grading':
            value = 1.0
        else:
            value = random()
        return value

    def dq_request(self, dq_expressions: List[str]):
        """
        Contrived request method to replicate output from DataQuery. Will replicate the
        form of a JPMaQS expression from DataQuery which will subsequently be used to
        test methods held in the api.Interface() Class.
        """
        aggregator = []
        for i, elem in enumerate(dq_expressions):
            elem_dict = {'item': (i + 1), 'group': None,
                         'attributes': [{'expression': elem, 'label': None,
                                         'attribute-id': None, 'attribute-name': None,
                                         'time-series': [['20220607',
                                                          self.jpmaqs_value(elem)]]}],
                         'instrument-id': None, 'instrument-name': None}
            aggregator.append(elem_dict)

        return aggregator

    def constructor(self, metrics: List[str] = ['value']):

        # Auth Connection.
        self.client_id = "client1"
        self.client_secret = "123"

        # Certificate & key connection.
        self.username = "user1"
        self.password = "123"

        self.crt = "/api_macrosynergy_com.crt"
        self.key = "/api_macrosynergy_com.key"

        self.endpoint = "/expressions/time-series"
        self.params = {"format": "JSON", "start-date": self.s_date_calc(),
                       "end-date": None, "calendar": "CAL_ALLDAYS",
                       "frequency": "FREQ_DAY",
                       "conversion": "CONV_LASTBUS_ABS",
                       "nan_treatment": "NA_NOTHING",
                       "data": "NO_REFERENCE_DATA"}

        cids_dmca = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY']
        xcats = ['EQXR_NSA', 'FXXR_NSA']
        self.metrics = metrics
        self.tickers = [cid + '_' + xcat for xcat in xcats for cid in cids_dmca]

        self.expression = self.jpmaqs_indicators(metrics=metrics,
                                                 tickers=self.tickers)

    @mock.patch("macrosynergy.dataquery.auth.OAuth",
                return_value=OAuth)
    def test_oauth_condition(self, mock_check_oauth):

        self.constructor()
        # Accessing DataQuery can be achieved via two methods: OAuth or Certificates /
        # Keys. To handle for the idiosyncrasies of the two access methods, split the
        # methods across individual Classes. The usage of each Class is controlled by the
        # parameter "oauth".
        # First check is that the DataQuery instance is using an OAuth Object if the
        # parameter "oauth" is set to to True.
        dq_access = api.Interface(oauth=True,
                                  client_id=self.client_id,
                                  client_secret=self.client_secret)
        self.assertTrue(dq_access.check_access())

    @mock.patch("macrosynergy.dataquery.api.Interface.check_access",
                return_value=CertAuth)
    def test_certauth_condition(self, mock_check_certauth):

        self.constructor()

        # Second check is that the DataQuery instance is using an CertAuth Object if the
        # parameter "oauth" is set to to False. The DataQuery Class's default is to use
        # certificate / keys.
        dq_access = api.Interface(username=self.username,
                                  password=self.password,
                                  crt=self.crt,
                                  key=self.key)
        self.assertTrue(dq_access.check_access())

    @mock.patch("macrosynergy.dataquery.auth.OAuth.get_dq_api_result",
               return_value={"info": {"code": 200}})
    def test_check_connection(self, mock_p_request):
        # If the connection to DataQuery is working, the response code will invariably be
        # 200. Therefore, use the Interface Object's method to check DataQuery
        # connections.

        self.constructor()

        with api.Interface(client_id=self.client_id,
                          client_secret=self.client_secret,
                          oauth=True) as dq:
            self.assertTrue(dq.check_connection())
            mock_p_request.assert_called_with(url=dq.access.base_url +
                                                  "/services/heartbeat")

        mock_p_request.assert_called_once()

    @mock.patch("macrosynergy.dataquery.auth.OAuth.get_dq_api_result",
               return_value={"info": {"code": 400}})
    def test_check_connection_fail(self, mock_p_fail):

        # Opposite of above method: if the connection to DataQuery fails, the error code
        # will be 400.

        self.constructor()

        with api.Interface(client_id=self.client_id,
                           client_secret=self.client_secret,
                           oauth=True) as dq:
            # Method returns a Boolean. In this instance, the method should return False
            # (unable to connect).
            self.assertTrue(not dq.check_connection())
            mock_p_fail.assert_called_with(url=dq.access.base_url +
                                               "/services/heartbeat")

        mock_p_fail.assert_called_once()

    def test_isolate_timeseries(self):

        self.constructor(metrics=['value', 'grading'])

        # First replicate the api.Interface()._request() method using the associated
        # JPMaQS expression.
        final_output = self.dq_request(dq_expressions=self.expression)

        # The method, .isolate_timeseries(), will receive the returned dictionary from
        # the ._request() method and return a dictionary where the keys are the tickers
        # and the values are stacked DataFrames where each column represents the metrics
        # that have been requested.

        # Therefore, assert that the dictionary contains the expected tickers and that
        # each value is a three-dimensional DataFrame: real_date, value, grade.
        results_dict, output_dict, s_list = api.Interface.isolate_timeseries(
                                                                            final_output,
                                                                            ['value', 'grading'],
                                                                            False, False
                                                                            )
        self.__dict__['results_dict'] = results_dict

        self.assertTrue(len(results_dict.keys()) == len(self.tickers))

        ticker_trunc = lambda t: t.split(',')[1]
        test_keys = list(map(ticker_trunc, results_dict.keys()))

        test_keys = sorted(test_keys)
        self.assertTrue(test_keys == sorted(self.tickers))

        first_ticker = next(iter(results_dict.keys()))
        self.assertTrue(results_dict[first_ticker].shape[1] == 3)

    def test_valid_ticker(self):

        # Call test_isolate_timeseries() to obtain the dictionary produced from the
        # associated method, isolate_timeseries().
        self.test_isolate_timeseries()

        # The method, self.valid_ticker(), is used to delimit if each ticker has a valid
        # time-series. To determine if a time-series is valid, pass through each date and
        # confirm that the associated value is not a NoneType. If all dates contain NaN
        # values, exclude the ticker from the DataFrame. For instance, USD_FXXR_NSA would
        # be removed.

        # All tickers held in the dictionary are valid tickers. Therefore, confirm the
        # keys for the two dictionary, received & returned, match.
        results_dict = api.Interface.valid_ticker(_dict=self.results_dict,
                                                  suppress_warning=True,
                                                  debug=False)
        self.assertTrue(len(results_dict.keys()) == len(self.results_dict.keys()))

        test = sorted(list(results_dict.keys()))
        benchmark = sorted(list(self.results_dict.keys()))
        self.assertTrue(test == benchmark)

        # Add a ticker that does not have a "valid" time-series and will subsequently be
        # removed. Confirm the series has been removed from the dictionary.
        f_ticker = next(iter(results_dict.keys()))
        shape = results_dict[f_ticker].shape
        # Again, as described above, a series is not valid if all values are NoneType.
        data = np.array([None] * (shape[0] * shape[1]))

        results_dict['DB(JPMAQS,USD_FXXR_NSA'] = data.reshape(shape)
        results_dict_USD = api.Interface.valid_ticker(self.results_dict,
                                                      suppress_warning=True,
                                                      debug=False)
        # Ticker should be removed from the dictionary.
        self.assertTrue('DB(JPMAQS,USD_FXXR_NSA' not in results_dict_USD.keys())

    def test_dataframe_wrapper(self):

        # After the time-series have been isolated for each ticker and all the tickers
        # have been validated, aggregate the results, still held in a dictionary, into
        # a single DataFrame where the columns will be the passed metrics, 'value' &
        # 'grading' plus the standardised JPMaQS columns: 'cid', 'xcat', 'real_date'.

        # The method api.Interface.valid_ticker() is not required given each ticker will
        # be valid by design.
        self.test_isolate_timeseries()

        results_dict = self.results_dict
        trial_df = api.Interface.dataframe_wrapper(_dict=results_dict, no_metrics=2,
                                                   original_metrics=['value', 'grading'])

        # Confirm the dictionary is a standardised DataFrame plus the respective metrics
        # passed.
        expected_columns = ['cid', 'xcat', 'real_date', 'value', 'grading']
        self.assertTrue(sorted(list(trial_df.columns)) == sorted(expected_columns))

        # Next confirm that tickers held in the DataFrame are the complete set: i.e all
        # the tickers defined in the constructor. All tickers are valid. Therefore,
        # confirm there is not any inadvertent leakage from constructing the DataFrame
        # and all tickers are included.

        tickers_df = list(trial_df['cid'] + '_' + trial_df['xcat'])
        self.assertTrue(sorted(tickers_df) == sorted(self.tickers))

        # Given the constructed nature of the DataFrame, confirm all values in the
        # 'grading' column are equal to 1.0.
        # Confirms the columns have the expected values.
        grades_test = np.all(trial_df['grading'] == 1.0)
        self.assertTrue(grades_test)

if __name__ == '__main__':
    unittest.main()
