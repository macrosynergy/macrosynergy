

from macrosynergy.dataquery import api
from macrosynergy.dataquery.auth import OAuth, CertAuth
from datetime import datetime
from pandas import Timestamp
from pandas.tseries.offsets import BDay
import unittest
import yaml

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

    def constructor(self):

        # Auth Connection.
        self.__dict__['client_id'] = "e5qY4ZXMUQOZ0tVK"
        s = "sug9OVlfem54ep7blgbknzc3Xj5aqko4el71pyf09t6mcbdmax8lzZlsKZ5oUvY2j4qQjjhrtxevgMn"
        self.__dict__['client_secret'] = s

        # Certificate & key connection.
        with open("../../../config.yml", 'r') as f:
            cf = yaml.load(f, Loader=yaml.FullLoader)
            self.cf = cf

        self.__dict__['endpoint'] = "/expressions/time-series"
        self.__dict__['params'] = {"format": "JSON", "start-date": self.s_date_calc(),
                                   "end-date": None, "calendar": "CAL_ALLDAYS",
                                   "frequency": "FREQ_DAY",
                                   "conversion": "CONV_LASTBUS_ABS",
                                   "nan_treatment": "NA_NOTHING",
                                   "data": "NO_REFERENCE_DATA"}

        cids_dmca = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP']
        xcats = ['EQXR_NSA']
        metric = ['value']
        self.__dict__['tickers'] = [cid + '_' + xcat for xcat in xcats
                                    for cid in cids_dmca]
        expression = ["DB(JPMAQS," + tick + f",{metric})" for tick in self.tickers]
        self.__dict__["expression"] = expression

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
                                  crt="../../../api_macrosynergy_com.crt",
                                  key="../../../api_macrosynergy_com.key")

        self.assertTrue(isinstance(dq_access.access, CertAuth))

    def test_fetch_threading(self):

        self.constructor()

        # Instantiate a local variable.
        params = self.params
        params["expressions"] = self.expression

        dq = api.Interface(username=self.cf["dq"]["username"],
                           password=self.cf["dq"]["password"],
                           crt="../../../api_macrosynergy_com.crt",
                           key="../../../api_macrosynergy_com.key")

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


if __name__ == '__main__':
    unittest.main()
