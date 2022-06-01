![Macrosynergy](docs/source/_static/MACROSYNERGY_Logo_Primary.png)

# Macrosynergy Quant Research
[![PyPI Latest Release](https://img.shields.io/pypi/v/macrosynergy.svg)](https://pypi.org/project/macrosynergy/)
[![Package Status](https://img.shields.io/pypi/status/macrosynergy.svg)](https://pypi.org/project/macrosynergy/)
[![License](https://img.shields.io/pypi/l/macrosynergy.svg)](https://github.com/macrosynergy/macrosynergy/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/macrosynergy?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/macrosynergy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Macrosynergy research package contains 5 subpackages:

[1] management: simulates, analyses and reshapes standard quantamental dataframes.
[2] panel: analyses and visualizes panels of quantamental data.
[3] signal: transforms quantamental indicators into trading signals and does naive analysis.
[4] pnl: constructs portfolios based on signals, applies risk management and analyses realistic PnLs.
[5] dataquery: interface for donwloading data from JP Morgan DataQuery, with main module `api.py`. 

## Installation
The easiest method for installing the package is to use the PyPI installation:
```shell script
pip install macrosynergy
```
Alternatively, we you want to install the package directly from the GitHub repository using
```shell script
pip install https://github.com/macrosynergy/macrosynergy@main
```
for the latest stable version. Alternatively for the cutting edge development version, install the package from the
 `develop` branch as
```shell script
pip install https://github.com/macrosynergy/macrosynergy@development
```
