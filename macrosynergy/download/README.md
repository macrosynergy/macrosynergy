# `macrosynergy.download`

## Description

This subpackage contains functionalities for downloading JPMaQS data from the JPMorgan DataQuery API,
and for retrieving the Macrosynergy's Transaction Costs data set.

## Downloading JPMaQS data

### Functionality

Downloading via the `JPMaQSDownload` class needs the following inputs:

- Authentication credentials (OAuth or Certificate)
- A DataQuery expression, a JPMaQS tickers, or lists of cross-sections and extended categories.
- A start and end date.

Example:

```python
import os
import pandas as pd
from macrosynergy.download import JPMaQSDownload

client_id: str = os.getenv("DQ_CLIENT_ID")
client_secret: str = os.getenv("DQ_CLIENT_SECRET")

tickers: List[str] = ["EUR_FXXR_NSA", "USD_EQXR_NSA"]
expressions: List[str] = ["DB(JPMAQS,GBP_EQXR_NSA,value)"]
cids: List[str] = ["CAD", "AUD"]
xcats: List[str] = ["FXXR_NSA", "RIR_NSA"]
start_date: str = "2010-01-01"
end_date: str = "2020-01-01"

with JPMaQSDownload(
        client_id=client_id,
        client_secret=client_secret,
) as jpmaqs:
    data: pd.DataFrame = jpmaqs.download(
        tickers=tickers,
        expressions=expressions,
        cids=cids, xcats=xcats,
        start_date=start_date,
        end_date=end_date,
    )

proxy: Dict[str, str] = {
    "http": "http://proxy.com:8080",
    "https": "https://proxy.com:8080",
}

with JPMaQSDownload(
    client_id=client_id,
    client_secret=client_secret,
    proxy=proxy,
) as jpmaqs:
    data: List[dict] = jpmaqs.download(
                        cids=cids, xcats=xcats,
                        as_dataframe=False)

assert data.shape[0] > 0
data.info()
```

### Design principles

The key guiding principle of this subpackage can be summarized as follows:

- Download JPMaQS data from the API as a QDF, while still allowing saving the raw JSONs.
- Allow users to download any DataQuery expression, even though the JPMaQS specific functions may not apply to all expressions.
- Allow OAuth and Certificate authentication.
- Have detailed, well documented errors and break conditions.
- Allow for concurrent downloads, adding proxy settings/alternate identity certificates, and manage retries and timeouts.
- Have a simple, concise, and easy-to-use interface.

### Implementation specifics

#### Concurrency

The `DataQueryInterface` class makes use of `concurrent.futures` to allow for concurrent downloads. This means that as requests for data are sent out, the process will not wait for the response before sending out the next request. This is done to speed up the download process, as the API can handle multiple requests at once. The number of concurrent workers at the moment are controlled by defaults of the `ThreadPoolExecutor` class. The only control over concurrency is whether or not to use it at all.

#### Authentication

While certificate authentication does not require a lot of code, OAuth authentication is a bit more involved. The `macrosynergy.download.dataquery.OAuth` class manages the OAuth token and authentication. To allow for a "neat" interface between the two types of authentication, the `DataQueryInterface` class uses one of `OAuth` or `CertAuth` classes as an attribute. This allows for it to have a single method to insert authentication information in the request.

#### Retries and Error handling

The key to good error handling with context to this module is when and how to break. When a process breaks, it must include debug information such as nature of error and case-specific debug information.

- Client ID/Certificate Username.
- Request URL, Headers, and Response body (if available) for each request.
- The number of retries attempted.
- And of course, the error type and message.

For "soft" or transient errors, the process will retry the request a number of times before breaking. The number of retries is controlled by the `max_retries` parameter of the `DataQueryInterface` class. The default is 5. The timeouts are left to the defaults of the `requests` and `concurrent.futures` libraries.
For hard errors, where the error is clearly non-transient, the process will break immediately. These are cases such as `AuthenticationError` or `HeartbeatError`.

#### Debugging SSL and Proxy issues

If you're on a corporate/work network, you may have to use a proxy to access the internet. This can cause issues with connecting to the JPMorgan DataQuery API. The `JPMaQSDownload` allows for a proxy to be passed in as a dictionary, which is ultimately passed to the `requests` library. The `requests` library will then use the proxy to connect to the API.
Please also take a look at the [FAQs section of the package README](https://github.com/macrosynergy/macrosynergy#faqs-and-troubleshooting) for more information on SSL and Proxy issues.

## Retrieving Macrosynergy's Transaction Costs data

### Functionality

The `download_transaction_costs` function allows for the retrieval of the Macrosynergy's
Transaction Costs data set. The data is available as a CSV file, and can be retrieved
using the `download_transaction_costs` method.

Also see [`TransactionCosts.download()`](https://docs.macrosynergy.com/stable/macrosynergy.pnl.transaction_costs.html#macrosynergy.pnl.transaction_costs.TransactionCosts.download) for more information.

Example:

```python
from macrosynergy.download import download_transaction_costs
import pandas as pd

tc_data: pd.DataFrame = download_transaction_costs()
tc_data.info()
```
