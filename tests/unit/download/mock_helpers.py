from random import random
from typing import List, Dict, Any
import pandas as pd
from macrosynergy.download.dataquery import (
    OAUTH_BASE_URL,
    TIMESERIES_ENDPOINT,
    DataQueryInterface,
)


def random_string() -> str:
    """
    Used to generate random string for testing.
    """
    return "".join([chr(int(random() * 26 + 97)) for i in range(10)])


def mock_jpmaqs_value(elem: str) -> float:
    """
    Used to produce a value or grade for the associated ticker. If the metric is
    grade, the function will return 1.0 and if value, the function returns a random
    number between (0, 1).

    :param <str> elem: ticker.
    """
    ticker_split = elem.split(",")
    if ticker_split[-1][:-1] == "grading":
        value = 1.0
    else:
        value = random()
    return value


def mock_request_wrapper(
    dq_expressions: List[str], start_date: str, end_date: str
) -> List[Dict[str, Any]]:
    """
    Contrived request method to replicate output from DataQuery. Will replicate the
    form of a JPMaQS expression from DataQuery which will subsequently be used to
    test methods held in the api.Interface() Class.
    """
    aggregator: List[dict] = []
    dates: pd.DatetimeIndex = pd.bdate_range(start_date, end_date)
    for i, elem in enumerate(dq_expressions):
        if elem is None:
            raise ValueError("Expression cannot be None")
        if elem in ["KEYBOARD_INTERRUPT"]:
            raise KeyboardInterrupt
        elem_dict = {
            "item": (i + 1),
            "group": None,
            "attributes": [
                {
                    "expression": elem,
                    "label": None,
                    "attribute-id": None,
                    "attribute-name": None,
                    "time-series": [
                        [d.strftime("%Y%m%d"), mock_jpmaqs_value(elem)] for d in dates
                    ],
                },
            ],
            "instrument-id": None,
            "instrument-name": None,
        }
        aggregator.append(elem_dict)

    return aggregator


class MockDataQueryInterface(DataQueryInterface):
    @staticmethod
    def jpmaqs_value(elem: str) -> float:
        """
        Use the mock jpmaqs_value to return a mock numerical jpmaqs value.
        """
        return mock_jpmaqs_value(elem=elem)

    def request_wrapper(
        self, dq_expressions: List[str], start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Use the mock request_wrapper to return a mock response.
        """
        return mock_request_wrapper(
            dq_expressions=dq_expressions, start_date=start_date, end_date=end_date
        )

    def __init__(self, *args, **kwargs):
        # if there is nothing in args or kwargs, use the default config
        config: dict = {}
        if not args and not kwargs:
            config: dict = dict(
                client_id="test_clid",
                client_secret="test_clsc",
                crt="test_crt",
                key="test_key",
                username="test_user",
                password="test_pass",
            )

        self.mask_expressions = []
        self.duplicate_entries = []
        super().__init__(*args, **kwargs, **config)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def check_connection(self, *args, **kwargs) -> bool:
        return True

    def download_data(
        self, expressions: List[str], start_date: str, end_date: str, **kwargs
    ) -> pd.DataFrame:
        ts: List[dict] = self.request_wrapper(expressions, start_date, end_date)
        if self.mask_expressions:
            for d in ts:
                if d["attributes"][0]["expression"] in self.mask_expressions:
                    d["attributes"][0]["time-series"] = None
                    d["attributes"][0][
                        "message"
                    ] = f"MASKED - {d['attributes'][0]['expression']}"

        if self.duplicate_entries:
            # copy half of the entries from d["attributes"][0]["time-series"] and append to itself
            for d in ts:
                if d["attributes"][0]["expression"] in self.duplicate_entries:
                    len_ts = len(d["attributes"][0]["time-series"])
                    dupls: List[List[str, float]] = d["attributes"][0]["time-series"][
                        : len_ts // 2
                    ]
                    d["attributes"][0]["time-series"] = (
                        d["attributes"][0]["time-series"] + dupls
                    )
        tsc = []
        if self.catalogue:
            for d in ts:
                if any([c in d["attributes"][0]["expression"] for c in self.catalogue]):
                    tsc.append(d)
        ts = tsc if self.catalogue else ts
        return ts

    def get_catalogue(self, group_id: str = None) -> List[str]:
        return self.catalogue

    def _gen_attributes(
        self,
        msg_errors: List[str] = None,
        mask_expressions: List[str] = None,
        msg_warnings: List[str] = None,
        catalogue: List[str] = None,
        unavailable_expressions: List[str] = None,
        duplicate_entries: List[str] = None,
    ):
        self.msg_errors: List[str] = [] if msg_errors is None else msg_errors
        self.msg_warnings: List[str] = [] if msg_warnings is None else msg_warnings
        self.unavailable_expressions: List[str] = (
            [] if unavailable_expressions is None else unavailable_expressions
        )
        self.catalogue: List[str] = [] if catalogue is None else catalogue

        self.mask_expressions: List[str] = (
            [] if mask_expressions is None else mask_expressions
        )

        self.duplicate_entries: List[str] = (
            [] if duplicate_entries is None else duplicate_entries
        )
