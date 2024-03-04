"""
Constants used throughout the package.

- FREQUENCY_MAP:  used to map the frequencies from D, W, M, Q, A to Business Day Frequencies.
"""

FREQUENCY_MAP = {
    "D": "B",
    "W": "W-FRI",
    "M": "BM",
    "Q": "BQ",
    "A": "BA",
}
