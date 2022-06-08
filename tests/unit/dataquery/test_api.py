

from macrosynergy.dataquery import api
from macrosynergy.dataquery.auth import OAuth, CertAuth
from datetime import datetime
from pandas import Timestamp
from pandas.tseries.offsets import BDay
from typing import List
import unittest
import yaml
import os

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
    def base_directory():
        cwd = os.getcwd()
        cwd_list = str(cwd).split('/')
        # Distinguish between running locally or on GitHub.
        base_dir = '/'.join(cwd_list[:cwd_list.index('tests')])
        return base_dir

    def path_finder(self, file):

        path_bool = os.path.exists(self.base_dir + '/' + self.path + file)
        if not path_bool:
            cert_path = self.path + file
        else:
            cert_path = self.base_dir + '/' + self.path + file

        return cert_path

    @staticmethod
    def jpmaqs_indicators(metrics, tickers):

        dq_tix = []
        for metric in metrics:
            dq_tix += ["DB(JPMAQS," + tick + f",{metric})" for tick in tickers]

        return dq_tix

    def constructor(self, metrics: List[str] = ['value']):

        # Auth Connection.
        self.client_id = "e5qY4ZXMUQOZ0tVK"
        s = "sug9OVlfem54ep7blgbknzc3Xj5aqko4el71pyf09t6mcbdmax8lzZlsKZ5oUvY2j4qQjjhrtxevgMn"
        self.client_secret = s

        self.base_dir = self.base_directory()
        self.path = "tests/unit/dataquery/cert_files"

        # Certificate & key connection.
        conf_path = self.path_finder("/config.yml")
        with open(conf_path, 'r') as f:
            cf = yaml.load(f, Loader=yaml.FullLoader)
            self.cf = cf

        self.crt = self.path_finder("/api_macrosynergy_com.crt")
        self.key = self.path_finder("/api_macrosynergy_com.key")

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

    def test_oauth_condition(self):

        self.constructor()

        # Accessing DataQuery can be achieved via two methods: OAuth or Certificates /
        # Keys. To handle for the idiosyncrasies of the two access methods, split the
        # methods across individual Classes. The usage of each Class is controlled by the
        # parameter "oauth".
        dq_access = api.Interface(oauth=True,
                                  client_id=self.client_id,
                                  client_secret=self.client_secret)
        self.assertTrue(isinstance(dq_access.access, OAuth))

        # Default is to use the Certificates / Keys: oauth = False.
        dq_access = api.Interface(username=self.cf["dq"]["username"],
                                  password=self.cf["dq"]["password"],
                                  crt=self.crt,
                                  key=self.key)

        self.assertTrue(isinstance(dq_access.access, CertAuth))

    def test_fetch_threading(self):

        self.constructor()

        # Instantiate a local variable.
        params = self.params
        params["expressions"] = self.expression

        dq = api.Interface(username=self.cf["dq"]["username"],
                           password=self.cf["dq"]["password"],
                           crt=self.crt,
                           key=self.key)

        results = dq._fetch_threading(endpoint=self.endpoint,
                                      params=params)

        # Confirm the included tickers.
        test_ticker = []
        date_checker = []
        for elem in results:
            jpm_expression = elem["attributes"][0]['expression']
            f_date = elem["attributes"][0]['time-series'][0][0]
            f_date = str(Timestamp(f_date).strftime('%Y-%m-%d'))
            date_checker.append(f_date)
            test_ticker.append(jpm_expression.split(',')[1])

        self.assertTrue(len(self.tickers) == len(test_ticker))

        test_ticker = sorted(test_ticker)
        condition = test_ticker == sorted(self.tickers)
        self.assertTrue(condition)

        # Confirm the application of the start_date parameter.
        first_date = self.s_date_calc()
        self.assertTrue(first_date == next(iter(set(date_checker))))

    def test_request(self):

        # Test the threading functionality on a request larger than 20 tickers. Each
        # thread can only be split across multiple threads.
        self.constructor(metrics=['value', 'grading'])

        dq = api.Interface(username=self.cf["dq"]["username"],
                           password=self.cf["dq"]["password"],
                           crt=self.crt,
                           key=self.key)

        final_output = dq._request(endpoint=self.endpoint, tickers=self.expression,
                                   params={}, delay=0.3, start_date=self.s_date_calc())

        # Confirm the usage of threading, splitting the requests over multiple threads,
        # does not lead to any data leakages: all JPMaQS indicators are present in the
        # return dictionary.

        test_ticker = []
        for elem in final_output:
            jpm_expression = elem["attributes"][0]['expression']
            test_ticker.append(jpm_expression)

        self.assertTrue(len(self.expression) == len(test_ticker))

        test_ticker = sorted(test_ticker)
        condition = test_ticker == sorted(self.expression)
        self.assertTrue(condition)

        self.__dict__['final_output'] = final_output

    def test_isolate_timeseries(self):

        self.test_request()

        dq = api.Interface(username=self.cf["dq"]["username"],
                           password=self.cf["dq"]["password"],
                           crt=self.crt,
                           key=self.key)

        final_output = self.final_output

        # The method, .isolate_timeseries(), will receive the returned dictionary from
        # the ._request() method and return a dictionary where the keys are the tickers
        # and the values are stacked DataFrames where each column represents the metrics
        # that have been requested.
        # Therefore, assert that the dictionary contains the expected tickers and that
        # each value is a three-dimensional DataFrame: real_date, value, grade.
        results_dict, output_dict, s_list = dq.isolate_timeseries(final_output,
                                                                  ['value', 'grading'],
                                                                  False, False)

        self.assertTrue(len(results_dict.keys()) == len(self.tickers))

        ticker_trunc = lambda t: t.split(',')[1]
        test_keys = list(map(ticker_trunc, results_dict.keys()))

        test_keys = sorted(test_keys)
        self.assertTrue(test_keys == sorted(self.tickers))

        first_ticker = next(iter(results_dict.keys()))
        self.assertTrue(results_dict[first_ticker].shape[1] == 3)


if __name__ == '__main__':
    unittest.main()
