.. _03_dataquery:

Downloading JPMaQS Data
=======================

Downloading data from the J.P. Morgan DataQuery API
---------------------------------------------------

To download data from JP Morgan DataQuery, you can use the
JPMaQSDownload Object together with your OAuth authentication
credentials (default):

.. code:: python

   import pandas as pd
   from macrosynergy.download import JPMaQSDownload

   with JPMaQSDownload(
           client_id="<dq_client_id>",
           client_secret="<dq_client_secret>"
   ) as downloader:
       data = downloader.download(tickers="EUR_FXXR_NSA",
                                   start_date="2022-01-01")

   assert isinstance(data, pd.DataFrame) and not data.empty

   assert data.shape[0] > 0
   data.info()

Alternatively, you can also the certificate and private key pair, to
access DataQuery as:

.. code:: python

   import pandas as pd
   from macrosynergy.download import JPMaQSDownload

   with JPMaQSDownload(
           oauth=False,
           username="<dq_username>",
           password="<dq_password>",
           crt="<path_to_dq_certificate>",
           key="<path_to_dq_key>"
   ) as downloader:
       data = downloader.download(tickers="EUR_FXXR_NSA",
                                   start_date="2022-01-01")

   assert isinstance(data, pd.DataFrame) and not data.empty

   assert data.shape[0] > 0
   data.info()

Downloading the JPMaQS Data Catalogue
-------------------------------------

You can download the JPMaQS Data Catalogue as a pandas DataFrame using
the following method:

.. code:: python

   from macrosynergy.download import JPMaQSDownload

   jpmaqs = JPMaQSDownload(
           client_id="<dq_client_id>",
           client_secret="<dq_client_secret>"
   )

   catalogue = jpmaqs.get_catalogue()
   # catalogue: List[str]

Connecting via a proxy server
-----------------------------

Since a lot of institutions use a proxy server to connect to the
internet; the JPMaQSDownload object can be configured to use a proxy
server.

It is also possible to use a proxy server with the Dataquery interface.
Here’s an example:

.. code:: python

   import pandas as pd
   from macrosynergy.download import JPMaQSDownload

   cids = ['EUR','GBP','USD']
   xcats = ['FXXR_NSA','EQXR_NSA']
   tickers = [cid+"_"+xcat for cid in cids for xcat in xcats]

   oauth_proxy="https://secureproxy.example.com:port"
   proxy = {"https": oauth_proxy}
   # or proxy = {"http": "http://proxy.example.com:port"}

   # or even
   # proxy = {
   #     "http": "http://proxy.example.com:port",
   #     "https": "https://secucreproxy.example.com:port",
   # }

   with JPMaQSDownload(
           client_id = "<dq_client_id>",
           client_secret = "<dq_client_secret>",
           proxy = proxy
   ) as downloader:
       data = downloader.download(tickers = tickers, start_date="2022-01-01")

   assert isinstance(data, pd.DataFrame) and not df.empty
