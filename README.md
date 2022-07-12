![Macrosynergy](docs/source/_static/MACROSYNERGY_Logo_Primary.png)

# Macrosynergy Quant Research
[![PyPI Latest Release](https://img.shields.io/pypi/v/macrosynergy.svg)](https://pypi.org/project/macrosynergy/)
[![Package Status](https://img.shields.io/pypi/status/macrosynergy.svg)](https://pypi.org/project/macrosynergy/)
[![License](https://img.shields.io/pypi/l/macrosynergy.svg)](https://github.com/macrosynergy/macrosynergy/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/macrosynergy?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/macrosynergy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

The Macrosynergy package supports financial market research and the development of trading strategies based on formats and conventions of the J.P. Morgan Macrosynergy  Quantamental System (JPMaQS). JPMaQS provides quantitative-fundamental (quantamental) and market data in simple daily formats in accordance with the information state of markets. The Macrosynergy package consists of five sub-packages:

1. [management](./macrosynergy/management): simulates, analyses and reshapes standard quantamental dataframes.
2. [panel](./macrosynergy/panel): analyses and visualizes panels of quantamental data.
3. [signal](./macrosynergy/signal): transforms quantamental indicators into trading signals and does naive analysis.
4. [pnl](./macrosynergy/pnl): constructs portfolios based on signals, applies risk management and analyses realistic PnLs.
5. [dataquery](./macrosynergy/dataquery): interface for donwloading data from JP Morgan DataQuery, with main module [api.py](./macrosynergy/dataquery/api.py). 

## Installation
The easiest method for installing the package is to use the [PyPI](https://pypi.org/project/macrosynergy/) installation method:
```shell script
pip install macrosynergy
```
Alternatively, we you want to install the package directly from the [GitHub repository](https://github.com/macrosynergy/macrosynergy/tree/main) using
```shell script
pip install https://github.com/macrosynergy/macrosynergy@main
```
for the latest stable version. Alternatively for the cutting edge development version, install the package from the
 [develop](https://github.com/macrosynergy/macrosynergy/tree/develop) branch as
```shell script
pip install https://github.com/macrosynergy/macrosynergy@development
```
## Usage
### DataQuery Interface
To download data from JP Morgan DataQuery, you can use the DataQuery [Interface](./macrosynergy/dataquery/api.py)
together with your OAuth authentication credentials:
```python
import pandas as pd
from macrosynergy.dataquery import api

with api.Interface(
        oauth=True,
        client_id="<dq_client_id>",
        client_secret="<dq_client_secret>"
) as dq:
    data = dq.download(tickers="EUR_FXXR_NSA", start_date="2022-01-01")

assert isinstance(data, pd.DataFrame) and not data.empty

assert data.shape[0] > 0
data.info()
```

Alternatively, you can also the certificate and private key pair, to access DataQuery as:
```python
import pandas as pd
from macrosynergy.dataquery import api

with api.Interface(
        oauth=False,
        username="<dq_username>",
        password="<dq_password>",
        crt="<path_to_dq_certificate>",
        key="<path_to_dq_key>"
) as dq:
    data = dq.download(tickers="EUR_FXXR_NSA", start_date="2022-01-01")

assert isinstance(data, pd.DataFrame) and not data.empty

assert data.shape[0] > 0
data.info()
```

Both of the above example will download a snippet of example data from the premium JPMaQS dataset
of the daily timeseries of EUR FX excess returns.
