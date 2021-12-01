import requests
import base64
import json
from math import ceil
import yaml
from typing import List
import os
import concurrent.futures
import time
import threading

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

    def _fetch_ts(self, params: dict):

        endpoint = "/expressions/time-series"
        url = self.base_url + endpoint

        with threading.Lock():
            with requests.get(url=url, cert=self.cert, headers=self.headers,
                              params=params) as r:
                return r.text

    def _optimize(self, tickers: List[str], start_date: str = None, end_date: str = None,
                  calendar: str = "CAL_ALLDAYS", frequency: str = "FREQ_DAY"):

        params = {"format": "JSON", "start-date": start_date, "end-date": end_date,
                  "calendar": calendar, "frequency": frequency}

        output = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i, elem in enumerate(tickers):
                # The expression can either be a List of Tickers, or a singular ticker.
                time.sleep(0.3)
                params["expressions"] = elem
                results = executor.submit(self._fetch_ts, params)
                output.append(results)

            for f in concurrent.futures.as_completed(output):
                try:
                    response = json.loads(f.result())
                except ValueError:
                    print(f"Ticker unavailable: {elem}.")
                else:
                    output.extend(response["instruments"])

        return output

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

            temporary = self._optimize(tickers=tickers, **kwargs)
            results.extend(temporary)

        return results

def dq_output(cids, xcats, metrics=['value'], start_date='2000-01-01'):
    os.chdir("/Users/kurransteeds/repos/macrosynergy/macrosynergy")

    tickers = [cid + '_' + xcat for xcat in xcats for cid in cids]

    with open("../config.yml", 'r') as f:
        cf = yaml.load(f, Loader=yaml.FullLoader)

    dq = DataQueryInterface(username=cf["dq"]["username"], password=cf["dq"]["password"],
                            crt="../api_macrosynergy_com.crt",
                            key="../api_macrosynergy_com.key")

    results = dq.get_tickers(tickers=tickers, original_metrics=metrics,
                             start_date=start_date)

    return results


if __name__ == "__main__":
    cids_dmca = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK',
                 'USD']  # DM currency areas
    cids_dmec = ['DEM', 'ESP', 'FRF', 'ITL', 'NLG']  # DM euro area countries
    cids_latm = ['ARS', 'BRL', 'COP', 'CLP', 'MXN', 'PEN']  # Latam countries
    cids_emea = ['HUF', 'ILS', 'PLN', 'RON', 'RUB', 'TRY', 'ZAR']  # EMEA countries
    cids_emas = ['CNY', 'HKD', 'IDR', 'INR', 'KRW', 'MYR', 'PHP', 'SGD', 'THB',
                 'TWD']  # EM Asia countries

    cids_eufx = ['CHF', 'HUF', 'NOK', 'PLN', 'RON', 'SEK']  # EUR benchmark
    cids_g2fx = ['GBP', 'RUB', 'TRY']  # dual benchmark
    cids_usfx = ['AUD', 'BRL', 'CAD', 'CLP', 'CNY', 'COP', 'EUR', 'IDR', 'ILS', 'INR',
                 'JPY', 'KRW', 'MYR',
                 'MXN', 'NZD', 'PEN', 'PHP', 'SGD', 'THB', 'TWD', 'ZAR']  # USD benchmark

    cids = cids_dmca + cids_dmec

    metrics = ['value']
    start = time.time()
    results = dq_output(cids=cids, xcats=['FXXR_NSA'], metrics=metrics,
                        start_date="2000-01-01")
    end = time.time() - start
    print(f"Time taken: {end}.")