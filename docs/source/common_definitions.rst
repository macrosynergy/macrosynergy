.. _common_definitions:

Common Definitions
==================

Like any niche field, the world of quantamental data and its analysis
has its own terminology and definitions. This section will cover the
most common terms and definitions used within the Macrosynergy package.

The Quantamental Data Format (QDF)
----------------------------------

The Quantamental Data Format, alternatively a Quantamental DataFrame, is
a tabular schema for storing and organizing quantamental data. It is a
simple, yet flexible format that allows for ease of use and
extensibility. The Macrosynergy package is built around the QDF, and all
data is stored in this format.

An example of the QDF is shown below:

========== === ======== ===== ======= ======= =======
real_date  cid xcat     value grading eop_lag mop_lag
========== === ======== ===== ======= ======= =======
2023-01-02 AUD FXXR_NSA 3.8   1       0       10
2023-01-02 AUD RIR_NSA  2.0   1       0       10
2023-01-02 AUD EQXR_NSA -0.1  1       0       10
2023-01-02 CAD FXXR_NSA 0.1   2       0       10
2023-01-02 CAD RIR_NSA  1.3   2       0       10
2023-01-02 CAD EQXR_NSA -0.2  2       0       10
========== === ======== ===== ======= ======= =======

To get a better understanding of the QDF, please see our Research:

`How to build a quantamental system for investment
management <https://research.macrosynergy.com/how-to-build-a-quantamental-system/>`__

and

`Quantitative methods for macro information
efficiency <https://research.macrosynergy.com/quantitative-methods/>`__

on the `Macrosynergy Research <https://research.macrosynergy.com/>`__
website.

Cross-Section Identifiers
-------------------------

This would be a 3-letter string that identifies a cross-section/currency
area. In the package, these are commonly referred to as ``cid`` or
``cids``.

Examples:

-  USD - United States Dollar
-  FRA - France
-  EUR - Euro Area
-  GBP - Great Britain Pound

Extended Category Codes
-----------------------

These are strings identifying the category of a data series. In the
package, these are commonly referred to as ``xcat`` or ``xcats``.

Extended category codes denote base category tickers and their
transformations. . For example CPI_NSA would be a base category for
seasonally adjusted headline CPI and P1M1ML12 would mean % change of the
latest month versus one year ago.

An extended category code will contain the following parts:

-  Base category code (e.g. ``CPI``)
-  Adjustment (e.g. ``NSA``)
-  Transformation (e.g. ``P1M1ML12``)

Combining these parts with underscores (``_``) will give us the extended
category code. In the example above, the extended category code would be
``CPI_NSA_P1M1ML12``.


JPMaQS Indicator Names
----------------------

These are strings that identify the data series in the JPMaQS dataset. These
are unique to every time-series published in the dataset.
These are commonly referred to as ``ticker``/``tickers`` or ``indicator``
in the package. The ticker name is composed of the cross-section identifier
(``cid``) and the extended category code (``xcat``) separated by an underscore
(``_``). For example:

- Cross-Section Identifier: ``USD``
- Base Category Code: ``CPI``
- Adjustment: ``NSA``
- Transformation: ``P1M1ML12``

Which makes:

- Extended Category Code: ``CPI_NSA_P1M1ML12``

and:

- Ticker: ``USD_CPI_NSA_P1M1ML12``


JPMaQS Metrics
==============

Grading
-------

This is an integer between 1 and 3 (inclusive) that specifies the
quality of the data series. Grade 1 is the highest quality, and Grade 3
is the lowest quality. Here, "quality" means estimated proximity to the
actual value of an indicator to what was seen by the market at the
related point in time.

End-of-Period (EOP) lags
------------------------

These are integers that specify for each real-time date the number of
business days that passed since the end of the concurrent observation
period.

For example, if the series reports monthly data, and the last day of the
month would be the end of the period, then the EOP lag would be 0. When
data is published for the next date in the series, the EOP lag would be
1.

Median-of-Period (MOP) lags
---------------------------

These are integers that specify for each real-time date the number of
business days that passed since the median date of the concurrent
observation period.

For example, if the series reports monthly data, and the middle day of
the month would be the median of the period, and for a value published
on the first of the next month - the MOP lag would be approximated to 11
or 12.

Lags
----

Used as ``lag`` or ``lags`` in the package, a "lag" refers to a time delay 
or the use of past values of a variable to explain its current or future 
value. Lags are applied in the native frequency of the data (i.e. monthly
data is lagged by months, daily data is lagged by days). In simple terms, 
a lag represents the late arrival of a piece of information.

Slippage
--------

Used as ``slip`` in the package (see :func:`macrosynergy.management.utils.df_utils.apply_slip`),
slippage is a term used to the delay in acting on a signal or a piece of
information. It can also be seen as the time taken to take a position.
In the package, slippage is always measured in business days, and is always
applied before any resampling or lags.


JPMaQS Expressions on DataQuery
-------------------------------

The JPMaQS dataset is served via the `JPMorgan DataQuery <https://www.jpmorgan.com/markets/dataquery/>`__
platform. The DataQuery platform allows users to search for, discover, and analyze
financial data. DataQuery uses a consistent notation to represent the numerous datasets
and the many data series within them. The notation is as follows:

``DB(<DATASET_NAME>,<SERIES_IDENTIFIER>,<ATTRIBUTE>``

Where:

- ``DATASET_NAME`` is the name of the dataset (``JPMAQS``).
- ``SERIES_IDENTIFIER`` is the unique identifier for the data series, which is the ticker name in the case of JPMaQS (e.g. ``USD_CPI_NSA_P1M1ML12``)

- ``ATTRIBUTE`` is the attribute of the data series that you want to retrieve, which is the metric name for JPMaQS (e.g. ``value``).

Which makes the full expression:

- ``DB(JPMAQS,USD_CPI_NSA_P1M1ML12,value)``


