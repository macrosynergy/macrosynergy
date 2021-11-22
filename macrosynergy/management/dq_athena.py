import requests
import base64
import json
import pandas as pd
import numpy as np
from math import ceil
from collections import defaultdict
import yaml
from datetime import datetime

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

        unique_tix = list(set(tickers))
        dq_tix = []
        for metric in original_metrics:
            dq_tix += ["DB(JPMAQS," + tick + f",{metric})" for tick in unique_tix]

        tickers = dq_tix

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

        output_dict = self.isolate_timeseries(results)
        output_dict = self.valid_ticker(output_dict, original_metrics)

        df_athena = self.df_column(output_dict, original_metrics)

        return self.standardise(df_athena)

    @staticmethod
    def isolate_timeseries(list_):

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

        return output_dict

    @staticmethod
    def column_check(v, metric):

        data = list(v[metric])
        condition = all([isinstance(elem, type(None)) for elem in data])

        return condition

    def valid_ticker(self, _dict, original_metrics):

        metric = next(iter(original_metrics))

        dict_copy = _dict.copy()
        for k, v in _dict.items():

            condition = self.column_check(v, metric)
            if condition:

                print(f"The ticker, {k + ')'}, does not exist in the Database.")
                dict_copy.pop(k)
            else:
                continue

        return dict_copy

    @staticmethod
    def df_column(output_dict, original_metrics):

        index = next(iter(output_dict.values()))['real_date']
        no_rows = index.size
        no_columns = len(output_dict.keys()) * len(original_metrics)
        arr = np.empty(shape=(no_rows, no_columns), dtype=np.float32)

        i = 0
        columns = []
        for metric in original_metrics:
            for k, v in output_dict.items():
                col_name = k + ',' + metric + ')'
                columns.append(col_name)
                arr[:, i] = v[metric]
                i += 1

        df = pd.DataFrame(data=arr, columns=columns, index=index)

        return df

    def standardise(self, df_athena):
        columns = list(df_athena.columns)
        index = df_athena.index
        dates = list(map(lambda d: datetime.strptime(d, "%Y%m%d"), index))

        metrics = set(map(lambda ticker: ticker.split(',')[-1][:-1], columns))
        metrics = list(metrics)

        output_columns = ['cid', 'xcat', 'real_date'] + metrics

        shape = df_athena.shape
        no_metrics = len(metrics)
        no_cids = int(shape[1] / no_metrics)

        no_rows = shape[0] * no_cids
        arr = np.empty(shape=(no_rows, len(output_columns)), dtype=object)

        dict_ = defaultdict(list)

        for col in range(df_athena.shape[1]):
            col_name = str(df_athena.iloc[:, col].name)
            ticker = col_name.split(',')[1]
            dict_[ticker].append(df_athena.iloc[:, col])

        no_dates = len(dates)
        i = 0
        for k, v in dict_.items():
            cid = k[:3]
            v.insert(0, np.repeat(cid, no_dates))
            v.insert(1, np.repeat(k[4:], no_dates))
            v.insert(2, dates)
            data = np.column_stack(tuple(v))

            arr[i * no_dates:no_dates * (i + 1), :] = data

            i += 1

        return pd.DataFrame(data=arr, columns=output_columns)

def dq_output(tickers=None, metrics=['value'], start_date='2000-01-01'):

    with open("../../config.yml", 'r') as f:
        cf = yaml.load(f, Loader=yaml.FullLoader)

    dq = DataQueryInterface(username=cf["dq"]["username"], password=cf["dq"]["password"],
                            crt="../../api_macrosynergy_com.crt",
                            key="../../api_macrosynergy_com.key")

    df_ts = dq.get_tickers(tickers=tickers, original_metrics=metrics,
                           start_date=start_date)

    return df_ts


if __name__ == "__main__":

    metrics = ['value', 'eop_lag', 'mop_lag']
    xcats = ['CPIXFE_SJA_P6M6ML6AR', 'DU05YXR_NSA']
    cids = ['AUD', 'CAD', 'ESP']
    tickers = [cid + '_' + xcat for xcat in xcats for cid in cids]

    df = dq_output(tickers = tickers, metrics=metrics, start_date="2021-11-01")
    print(df)