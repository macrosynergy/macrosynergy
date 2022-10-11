![Macrosynergy](https://raw.githubusercontent.com/macrosynergy/macrosynergy/main/docs/source/_static/MACROSYNERGY_Logo_Primary.png?raw=True)

# Macrosynergy Quant Research
[![PyPI Latest Release](https://img.shields.io/pypi/v/macrosynergy.svg)](https://pypi.org/project/macrosynergy/)
[![Package Status](https://img.shields.io/pypi/status/macrosynergy.svg)](https://pypi.org/project/macrosynergy/)
[![License](https://img.shields.io/pypi/l/macrosynergy.svg)](https://github.com/macrosynergy/macrosynergy/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/macrosynergy?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/macrosynergy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/github/macrosynergy/macrosynergy/branch/develop/graph/badge.svg?token=BX4IKVD07R)](https://codecov.io/github/macrosynergy/macrosynergy)

The Macrosynergy package supports financial market research and the development of trading strategies based on formats and conventions of the J.P. Morgan Macrosynergy  Quantamental System (JPMaQS). JPMaQS provides quantitative-fundamental (quantamental) and market data in simple daily formats in accordance with the information state of markets. The Macrosynergy package consists of five sub-packages:

1. [management](./macrosynergy/management): simulates, analyses and reshapes standard quantamental dataframes.
2. [panel](./macrosynergy/panel): analyses and visualizes panels of quantamental data.
3. [signal](./macrosynergy/signal): transforms quantamental indicators into trading signals and does naive analysis.
4. [pnl](./macrosynergy/pnl): constructs portfolios based on signals, applies risk management and analyses realistic PnLs.
5. [dataquery](./macrosynergy/dataquery): interface for downloading data from JP Morgan DataQuery, with main module [api.py](./macrosynergy/dataquery/api.py). 

## Installation
The easiest method for installing the package is to use the [PyPI](https://pypi.org/project/macrosynergy/) installation method:
```shell script
pip install macrosynergy
```
 Alternatively for the cutting edge development version, install the package from the
 [develop](https://github.com/macrosynergy/macrosynergy/tree/develop) branch as
```shell script
pip install git+https://github.com/macrosynergy/macrosynergy@develop
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

Using the api you can also access a panel of tickers from different countries like so.
```python
cids = ['EUR','GBP','USD']

xcats = ['FXXR_NSA','EQXR_NSA']

tickers = [cid+"_"+xcat for cid in cids for xcat in xcats]

with api.Interface(
        oauth=True,
        username="<dq_username>",
        password="<dq_password>"
) as dq:
    data = dq.download(tickers="tickers, start_date="2022-01-01")

assert isinstance(data, pd.DataFrame) and not data.empty

assert data.shape[0] > 0
data.info()
```
### Management 
In order to use the rest of the package without access to the api you can [simulate](./macrosynergy/management/simulate_quantamental_data.py) quantamental data using the 

management sub-package. 
```python
from macrosynergy.management.simulate_quantamental_data import make_qdf

cids = ['AUD', 'GBP', 'NZD', 'USD']
xcats = ['FXXR_NSA', 'FXCRY_NSA', 'FXCRR_NSA', 'EQXR_NSA', 'EQCRY_NSA', 'EQCRR_NSA',
             'FXWBASE_NSA', 'EQWBASE_NSA']

df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                'sd_mult'])

df_cids.loc['AUD'] = ['2000-01-01', '2022-03-14', 0, 1]
df_cids.loc['GBP'] = ['2001-01-01', '2022-03-14', 0, 2]
df_cids.loc['NZD'] = ['2002-01-01', '2022-03-14', 0, 3]
df_cids.loc['USD'] = ['2000-01-01', '2022-03-14', 0, 4]

 df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])
df_xcats.loc['FXXR_NSA'] = ['2010-01-01', '2022-03-14', 0, 1, 0, 0.2]
df_xcats.loc['FXCRY_NSA'] = ['2010-01-01', '2022-03-14', 1, 1, 0.9, 0.2]
df_xcats.loc['FXCRR_NSA'] = ['2010-01-01', '2022-03-14', 0.5, 0.8, 0.9, 0.2]
df_xcats.loc['EQXR_NSA'] = ['2010-01-01', '2022-03-14', 0.5, 2, 0, 0.2]
df_xcats.loc['EQCRY_NSA'] = ['2010-01-01', '2022-03-14', 2, 1.5, 0.9, 0.5]
df_xcats.loc['EQCRR_NSA'] = ['2010-01-01', '2022-03-14', 1.5, 1.5, 0.9, 0.5]
df_xcats.loc['FXWBASE_NSA'] = ['2010-01-01', '2022-02-01', 1, 1.5, 0.8, 0.5]
df_xcats.loc['EQWBASE_NSA'] = ['2010-01-01', '2022-02-01', 1, 1.5, 0.9, 0.5]
data = make_qdf(df_cids, df_xcats, back_ar=0.75)
```
The management sub-package can also be used to [check](./macrosynergy/management/check_availability.py) which data is available
in the dataframe.


```python
from macrosynergy.management.check_availability import check_availability
filt_na = (data['cid'] == 'USD') & (data['real_date'] < '2015-01-01')
data_filt.loc[filt_na, 'value'] = np.nan
check_availability(df=data_filt, xcats=xcats, cids=cids)
```
You can also use the built-in function to [reshape](./macrosynergy/management/shape_dfs.py) the data depending on
the dates or tickers of your choice.

```python
data_reduced = reduce_df(data, xcats=xcats[:-1], cids=cids[0],
                       start='2012-01-01', end='2018-01-31')
```

### Panel
#### Basket
The basket class is used to calculate the returns and carries of financial contracts using various methods,
a [basket](./macrosynergy/panel/basket.py) is created as so.

```python
from macrosynergy.panel.basket import Basket

black = {'AUD': ['2010-01-01', '2013-12-31'], 'GBP': ['2010-01-01', '2013-12-31']}
contracts = ['AUD_FX', 'AUD_EQ', 'NZD_FX', 'GBP_EQ', 'USD_EQ']
gdp_figures = [17.0, 17.0, 41.0, 9.0, 250.0]
basket_1 = Basket(
    df=data, contracts=contracts_1, ret="XR_NSA", cry=["CRY_NSA", "CRR_NSA"],
    blacklist=black
)
basket_1.make_basket(weight_meth="equal", max_weight=0.55, basket_name="GLB_EQUAL")
```
Using the basket class you have access to the methods such as visulasing the weights associated with each contract,
or returning the weight or basket.
```python
basket_1.return_basket()
basket_1.return_weights()
basket_1.weight_visualiser(basket_name="GLB_EQUAL")
```
You can also calculate and visualise the following and more with built-in functions.
1.  [historic volume](./macrosynergy/panel/historic_vol.py)
2.  [z-scores](./macrosynergy/panel/make_zn_scores.py)
3.  [beta values](./macrosynergy/panel/return_beta.py)
4.  [timeline](./macrosynergy/panel/view_timelines.py) 
```python
from macrosynergy.panel.historic_vol import historic_vol
data_historic = historic_vol(
    data, cids=cids, xcat='FXXR_NSA', lback_periods=21, lback_meth='ma', half_life=11,
    remove_zeros=True)
```

```python
from macrosynergy.panel.make_zn_scores import make_zn_scores
z_mean = make_zn_scores(data, xcat='FXXR_NSA', sequential=True, cids=cids,
                      blacklist=black, iis=False, neutral='mean',
                      pan_weight=0.5, min_obs=261, est_freq="w")
z_median = make_zn_scores(data, xcat='FXXR_NSA', sequential=True, cids=cids,
                      blacklist=black, iis=False, neutral='median',
                      pan_weight=0.5, min_obs=261, est_freq="d")
```

```python
from macrosynergy.panel.return_beta import return_beta
benchmark_return = "USD_FXXR_NSA"
data_hedge = return_beta(df=data, xcat='FXXR_NSA', cids=cids,
                       benchmark_return=benchmark_return, start='2010-01-01',
                       end='2020-10-30',
                       blacklist=black, meth='ols', oos=True,
                       refreq='w', min_obs=24, hedged_returns=True)
print(df_hedge)
beta_display(df_hedge=df_hedge, subplots=False)
```

```python
view_timelines(data, xcats=['FXXR_NSA','FXCRY_NSA'], cids=cids[0],
                   size=(10, 5), title='AUD Return and Carry')
```
### Signal
#### Signal Return Relations
The [SignalReturnRelations](./macrosynergy/signal/signal_return.py) class analyses and visualises signal and
return series.
```python
from macrosynergy.signal.signal_return import SignalReturnRelations

srn = SignalReturnRelations(data, ret="EQXR_NSA", sig="EQCRY_NSA", rival_sigs=None,
                                sig_neg=True, cosp=True, freq="M", start="2002-01-01")
srn.summary_table()
```
In the creation of the class you can also indicate rival signals for basic relational statistics.
```python
r_sigs = [ "EQCRR_NSA"]
srn = SignalReturnRelations(data, "EQXR_NSA", sig="EQCRY_NSA", rival_sigs=r_sigs,
                            sig_neg=True, cosp=True, freq="M", start="2002-01-01")
df_sigs = srn.signals_table(sigs=['EQCRY_NSA_NEG', 'EQCRR_NSA_NEG'])

df_sigs_all = srn.signals_table()
```
Using the class you can plot accuracy bars between returns and signals.
```python
srn.accuracy_bars(type="signals", title="Accuracy measure between target return, EQXR_NSA,"
                                        " and the respective signals, ['EQCRY_NSA_NEG', "
                                        " 'EQCRR_NSA_NEG'].")
```
### PnL
#### Naive pnl
The [NaivePnL](./macrosynergy/pnl/naive_pnl.py) class computes Pnls with limited signal options and 
disregarding transaction costs.
```python
from macrosynergy.pnl.naive_pnl import NaivePnL
pnl = NaivePnL(data, ret="EQXR_NSA", sigs=["CRY", "GROWTH"], cids=cids,
        start="2000-01-01", bms=["EUR_EQXR_NSA", "USD_EQXR_NSA"])
```
You can then make the pnl and see a list of key pnl statistics.
```python
pnl.make_pnl(
        sig="GROWTH", sig_op="zn_score_pan", sig_neg=True, rebal_freq="monthly",
        vol_scale=5, rebal_slip=1, min_obs=250, thresh=2)
df_eval = pnl.evaluate_pnls(
        pnl_cats=["PNL_GROWTH_NEG"], start="2015-01-01", end="2020-12-31")
```


## Documentation

The official documentation can be found at our website: https://docs.macrosynergy.com
