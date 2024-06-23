"""
Constants used throughout the package.

- FREQUENCY_MAP:  used to map the frequencies from D, W, M, Q, A to Business Day Frequencies.
"""

from typing import List

FREQUENCY_MAP = {
    "D": "B",
    "W": "W-FRI",
    "M": "BM",
    "Q": "BQ",
    "A": "BA",
}
JPMAQS_METRICS: List[str] = ["value", "grading", "eop_lag", "mop_lag"]
