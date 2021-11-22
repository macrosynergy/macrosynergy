import requests
import base64
import json
import pandas as pd
import numpy as np
from math import ceil
from collections import defaultdict

URL = "https://platform.jpmorgan.com/research/dataquery/api/v2"
class DataQueryInterface(object):

    def __init__(self, username: str, password: str,
                 crt: str = "api_macrosynergy_com.crt",
                 key: str = "api_macrosynergy_com.key"):

        self.auth = base64.b64encode(bytes(f'{username:s}:{password:s}',
                                           "utf-8")).decode('ascii')
        self.headers = {"Authorization": f"Basic {self.auth:s}"}
        self.base_url = URL
        self.cert = (crt, key)

    def _fetch_ts(self, params: dict, start_date: str = None, end_date: str = None,
                  calendar: str = "CAL_ALLDAYS", frequency: str = "FREQ_DAY"):

        params["format"] = "JSON"
        params["start-date"] = start_date
        params["end-date"] = end_date
        params["calendar"] = calendar
        params["frequency"] = frequency

        endpoint = "/expressions/time-series"
        url = self.base_url + endpoint
        results = []

        with requests.get(url=url, cert=self.cert, headers=self.headers,
                          params=params) as r:
            self.last_response = r.text
        response = json.loads(self.last_response)

        assert "instruments" in response.keys()
        results.extend(response["instruments"])

        return results

    def get_tickers(self, tickers, original_metrics, **kwargs):

        no_tickers = len(tickers)
        iterations = ceil(no_tickers / 20)
        remainder = no_tickers % 20

        results = []
        tickers_copy = tickers.copy()
        for i in range(iterations):
            if i < (iterations - 1):
                tickers = tickers_copy[i * 20: (i * 20) + 20]
            else:
                tickers = tickers_copy[-remainder:]

            params = {"expressions": tickers}
            output = self._fetch_ts(params=params, **kwargs)
            results.extend(output)

        no_metrics = len(set([tick.split(',')[-1][:-1] for tick in tickers_copy]))
        print(f"Number of metrics: {no_metrics}.")

        results_dict = self.isolate_timeseries(results, original_metrics)
        results_dict = self.valid_ticker(results_dict)

        results_copy = results_dict.copy()
        try:
            results_copy.popitem()
        except Exception as err:
            print(err)
            print("None of the tickers are available in the Database.")
            return
        else:
            return self.dataframe_wrapper(results_dict, no_metrics, original_metrics)

    @staticmethod
    def isolate_timeseries(list_, metrics):
        output_dict = defaultdict(dict)
        size = len(list_)

        for i in range(size):
            try:
                r = list_.pop()
            except IndexError:
                break
            else:
                dictionary = r['attributes'][0]
                ticker = dictionary['expression'].split(',')
                metric = ticker[-1][:-1]

                ticker_split = ','.join(ticker[:-1])
                ts_arr = np.array(dictionary['time-series'])

                if ticker_split not in output_dict:
                    output_dict[ticker_split]['real_date'] = ts_arr[:, 0]
                    output_dict[ticker_split][metric] = ts_arr[:, 1]
                else:
                    output_dict[ticker_split][metric] = ts_arr[:, 1]

        no_rows = ts_arr[:, 1].size
        modified_dict = {}
        d_frame_order = ['real_date'] + metrics

        for k, v in output_dict.items():
            arr = np.empty(shape=(no_rows, len(d_frame_order)), dtype=object)
            for i, metric in enumerate(d_frame_order):
                arr[:, i] = v[metric]

            modified_dict[k] = arr
        return modified_dict

    @staticmethod
    def column_check(v, col):
        returns = list(v[:, col])
        condition = all([isinstance(elem, type(None)) for elem in returns])

        return condition

    def valid_ticker(self, _dict):
        dict_copy = _dict.copy()
        for k, v in _dict.items():

            condition = self.column_check(v, 1)
            if condition:
                print(f"The ticker, {k}, does not exist in the Database.")
                dict_copy.pop(k)
            else:
                continue

        return dict_copy

    @staticmethod
    def dataframe_wrapper(_dict, no_metrics, original_metrics):
        tickers_no = len(_dict.keys())
        length = list(_dict.values())[0].shape[0]

        arr = np.empty(shape=(length * tickers_no, 3 + no_metrics), dtype=object)
        print(arr.shape)

        i = 0
        for k, v in _dict.items():
            ticker = k.split(',')
            ticker = ticker[1].split('_')

            cid = ticker[0]
            xcat = '_'.join(ticker[1:])

            cid_broad = np.repeat(cid, repeats=v.shape[0])
            xcat_broad = np.repeat(xcat, repeats=v.shape[0])
            data = np.column_stack((cid_broad, xcat_broad, v))

            row = i * v.shape[0]
            arr[row:row + v.shape[0], :] = data
            i += 1

        columns = ['cid', 'xcat', 'real_date']
        cols_output = columns + original_metrics

        df = pd.DataFrame(data=arr, columns=cols_output)

        df['real_date'] = pd.to_datetime(df['real_date'], yearfirst=True)
        df = df[df['real_date'].dt.dayofweek < 5]

        return df.fillna(value=np.nan).reset_index(drop=True)