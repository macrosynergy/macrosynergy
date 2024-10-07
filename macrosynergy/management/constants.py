"""

Constants used throughout the package.

Frequency Mappings
------------------

* FREQUENCY_MAP:  used to map the frequencies from D, W, M, Q, A to Business Day Frequencies.

* FFILL_LIMITS: used to map the frequencies from D, W, M, Q, A to the maximum number of ffill periods.


Cross-sections of interest
--------------------------

* cids_dmca: DM currency areas

* cids_dmec: DM euro area countries

* cids_latm: Latam countries

* cids_emea: EMEA countries

* cids_emas: EM Asia countries

* cids_dm: DM currency areas and DM euro area countries

* cids_em: Latam, EMEA and EM Asia countries

* all_cids: All countries

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

ANNUALIZATION_FACTORS = {
    "A": 1,
    "BA": 1,
    "Q": 4,
    "BQ": 4,
    "BQE": 4,
    "M": 12,
    "BM": 12,
    "BME": 12,
    "W": 52,
    "W-FRI": 52,
    "B": 252,
    "D": 252,
}


FFILL_LIMITS = {
    "D": 1,
    "W": 5,
    "M": 24,
    "Q": 64,
    "A": 252,
    "B": 1,
    "W-FRI": 5,
    "BM": 24,
    "BQ": 64,
    "BA": 252,
}
DAYS_PER_FREQ = FFILL_LIMITS.copy()

# Cross-sections of interest
## DM currency areas
cids_dmca = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NOK", "NZD", "SEK", "USD"]
## DM euro area countries
cids_dmec = ["DEM", "ESP", "FRF", "ITL"]
## Latam countries
cids_latm = ["BRL", "COP", "CLP", "MXN", "PEN"]
## EMEA countries
cids_emea = ["CZK", "HUF", "ILS", "PLN", "RON", "RUB", "TRY", "ZAR"]
## EM Asia countries
cids_emas = ["CNY", "IDR", "INR", "KRW", "MYR", "PHP", "SGD", "THB", "TWD"]

cids_dm = cids_dmca + cids_dmec
cids_em = cids_latm + cids_emea + cids_emas

all_cids = sorted(cids_dm + cids_em)
