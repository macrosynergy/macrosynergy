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

# Cross-sections of interest
cids_dmca = [
    "AUD", "CAD", "CHF", "EUR", "GBP", "JPY","NOK", "NZD", "SEK", "USD",
]  # DM currency areas
cids_dmec = ["DEM", "ESP", "FRF", "ITL"]  # DM euro area countries
cids_latm = ["BRL", "COP", "CLP", "MXN", "PEN"]  # Latam countries
cids_emea = ["CZK", "HUF", "ILS", "PLN", "RON", "RUB", "TRY", "ZAR"]  # EMEA countries
cids_emas = [
    "CNY", "IDR", "INR", "KRW", "MYR", "PHP", "SGD", "THB", "TWD",
]  # EM Asia countries

cids_dm = cids_dmca + cids_dmec
cids_em = cids_latm + cids_emea + cids_emas

cids = sorted(cids_dm + cids_em)
