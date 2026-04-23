import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def no_nan_data() -> pd.DataFrame:
    cid_start_dict = {
        "CAD": "2023-01-01",
        "AUD": "2023-04-01",
        "GBP": "2023-06-01",
        "USD": "2023-01-01",
    }

    cid_index_vals, date_index_vals = [], []
    for cid, start in cid_start_dict.items():
        date_range = pd.date_range(start, "2025-12-31", freq="ME")

        date_index_vals.extend(date_range.tolist())
        cid_index_vals.extend([cid] * len(date_range))

    data = pd.DataFrame(
        index=pd.MultiIndex.from_arrays(
            [cid_index_vals, date_index_vals], names=["cid", "real_date"]
        ),
        data=np.random.randn(len(cid_index_vals), 4).round(3),
        columns=["CPI", "GROWTH", "RIR", "XR"],
    )

    return data


@pytest.fixture
def nan_data(no_nan_data) -> pd.DataFrame:
    num_nans = {
        ("CAD", "CPI"): 0,
        ("CAD", "GROWTH"): 2,
        ("CAD", "RIR"): 0,
        ("CAD", "XR"): 1,
        ("AUD", "CPI"): 3,
        ("AUD", "GROWTH"): 3,
        ("AUD", "RIR"): 0,
        ("AUD", "XR"): 2,
        ("GBP", "CPI"): 1,
        ("GBP", "GROWTH"): 0,
        ("GBP", "RIR"): 1,
        ("GBP", "XR"): 1,
        ("USD", "CPI"): 0,
        ("USD", "GROWTH"): 0,
        ("USD", "RIR"): 0,
        ("USD", "XR"): 0,
    }

    nan_data = no_nan_data.copy()
    for (entity, column), n in num_nans.items():
        if n > 0:
            entity_index = nan_data.loc[entity].sort_index().index[:n]
            nan_data.loc[(entity, entity_index), column] = np.nan

    return nan_data
