
===========================================================
Welcome to Macrosynergy Quant Research Documentation!
===========================================================

Version |version|

A diagram [#footnote1]_ showing Macrosynergy's logo.

.. figure:: MACROSYNERGY_Logo_Primary.png


**A software package designed by the Macrosynergy Team to accompany JPMaQS: a novel source
of macroeconomic data pioneered by Macrosynergy & JP Morgan.**

.. note::

   The package is primarily aimed at active asset managers, and Macrosynergy is not
   liable for potential investment decisions.


Context
=======

    **Data Science**

    **Tradable Economics**

    **Investor Value**

Outline
-----------

Overview of the potential work possible with the ``macrosynergy`` package. The main authors
are Ralph Sueppel, Lasse de la Porte Simonsen, and Curran Steeds.

.. tip::

   The source and built documentation, which includes these notes,
   is (and will permanently remain) hosted on GitHub at:
   https://github.com/macrosynergy/macrosynergy

Background
----------

Macrosynergy was created in August 2009 by Robert Enserro and Nikos Makris.
In January 2010, Macrosynergy launched its first investment fund, the Macrosynergy Trading Fund, a global macro fund that grew to peak assets under management of $950 million.
In November 2010, Gavin Moule joined Macrosynergy. Nikos and Gavin co-managed the Trading Fund throughout its ten year history.
Over the last decade, Macrosynergy launched and managed three other investment funds.

Ralph Sueppel joined Macrosynergy in January 2018. At that point, the firm channelled resources to further develop the
quantamental effort first started in 2005, when Ralph, Robert, Nikos and Gavin worked together at BlueCrest Capital.

Today Ralph and Lasse Simonsen, who joined Macrosynergy in 2016, lead Macrosynergy’s
quantamental research and development effort, while Robert remains responsible for the company’s strategic
business development and operations.


An example of the code in use (Python console notation):

.. code:: python

   >>> from macrosynergy.management.dq_simplification import DataQueryInterface
   >>> dq = DataQueryInterface(username=cf["dq"]["username"], password=cf["dq"]["password"],
                               crt="../api_macrosynergy_com.crt",
                               key="../api_macrosynergy_com.key")
   >>> dq_tickers =
   >>> cids_dmca = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK', 'USD']  # DM currency areas
   >>> cids_dmec = ['DEM', 'ESP', 'FRF', 'ITL', 'NLG']  # DM euro area countries
   >>> cids_latm = ['ARS', 'BRL', 'COP', 'CLP', 'MXN', 'PEN']  # Latam countries
   >>> cids_emea = ['HUF', 'ILS', 'PLN', 'RON', 'RUB', 'TRY', 'ZAR']  # EMEA countries
   >>> cids_emas = ['CNY', 'HKD', 'IDR', 'INR', 'KRW', 'MYR', 'PHP', 'SGD', 'THB', 'TWD']  # EM Asia countries
   >>> cids_dm = cids_dmca + cids_dmec
   >>> cids_em = cids_latm + cids_emea + cids_emas
   >>> cids = sorted(cids_dm + cids_em)
   >>> dq_tickers = [cid + '_CPIXFE_SJA_P6M6ML6AR' for cid in cids]
   >>> metrics = ['value']

   >>> df_ts = dq.get_tickers(tickers=dq_tickers, original_metrics=metrics,
                              start_date="2000-01-01")

   >>> if isinstance(df_ts, pd.DataFrame):
            df_ts = df_ts.sort_values(['cid', 'xcat', 'real_date']).reset_index(drop=True)

   >>> print(df_ts)
            cid  xcat                 real_date   value
    0       AUD  CPIXFE_SJA_P6M6ML6AR 2000-01-03  0.61972
    1       AUD  CPIXFE_SJA_P6M6ML6AR 2000-01-04  0.61972
    2       AUD  CPIXFE_SJA_P6M6ML6AR 2000-01-05  0.61972
    3       AUD  CPIXFE_SJA_P6M6ML6AR 2000-01-06  0.61972
    4       AUD  CPIXFE_SJA_P6M6ML6AR 2000-01-07  0.61972
    ...     ...                   ...        ...      ...
    171025  ZAR  CPIXFE_SJA_P6M6ML6AR 2021-11-02  2.97367
    171026  ZAR  CPIXFE_SJA_P6M6ML6AR 2021-11-03  2.97367
    171027  ZAR  CPIXFE_SJA_P6M6ML6AR 2021-11-04      NaN
    171028  ZAR  CPIXFE_SJA_P6M6ML6AR 2021-11-05      NaN
    171029  ZAR  CPIXFE_SJA_P6M6ML6AR 2021-11-08      NaN

Quick links
===========

Links to Macrosynergy Website and associated pages.
---------------------------------------------------


* Python 3 documentation: https://docs.python.org/3/
* NumPy documentation: https://numpy.org/doc/stable/reference/
* Systematic-Risk, Systematic Value. Quantitative Finance Research Journal by Macrosynergy's
  Ralph Sueppel:
  http://www.sr-sv.com
* Macrosynergy's website: https://www.macrosynergy.com
* Macrosynergy's Linkedin Page: https://www.linkedin.com/company/macrosynergy-partners/


.. [#footnote1] Image sourced from `this webpage
   <https://constanzapinto.com/Macrosynergy>`_.