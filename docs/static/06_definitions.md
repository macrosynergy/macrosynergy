# Common Definitions

Like any niche field, the world of quantamental data and it's analysis has it's own
terminology and definitions. This section will cover the most common terms and definitions
used within the Macrosynergy package.

## The Quantamental Data Format (QDF)

The Quantamental Data Format, alternatively a Quantamental DataFrame, is a tabular schema
for storing and organizing quantamental data. It is a simple, yet flexible format that
allows for ease of use and extensibility. The Macrosynergy package is built around the
QDF, and all data is stored in this format.

An example of the QDF is shown below:

| real_date  | cid | xcat     | value | grading | eop_lag | mop_lag |
| :--------- | :-- | :------- | ----: | ------: | ------: | ------: |
| 2023-01-02 | AUD | FXXR_NSA |     0 |       1 |       0 |       0 |
| 2023-01-02 | AUD | RIR_NSA  |     0 |       1 |       0 |       0 |
| 2023-01-02 | AUD | EQXR_NSA |  -0.1 |       1 |       0 |       0 |
| 2023-01-02 | CAD | FXXR_NSA |     0 |       1 |       0 |       0 |
| 2023-01-02 | CAD | RIR_NSA  |     0 |       1 |       0 |       0 |
| 2023-01-02 | CAD | EQXR_NSA |  -0.2 |       1 |       0 |       0 |

To get a better understanding of the QDF, please see Ralph's Research posts:

[**How to build a quantamental system for investment management**
](https://research.macrosynergy.com/how-to-build-a-quantamental-system/)

and

[**Quantitative methods for macro information efficiency**
](https://research.macrosynergy.com/quantitative-methods/)

on the [Macrosynergy Research](https://research.macrosynergy.com/) website.

## Cross-Section Identifiers

This would be a 3-letter string that identifies a cross-section/currency area.
In the package, these are commonly referred to as `cid` or `cids`.

Examples:

- USD - United States Dollar
- FRA - France
- EUR - Euro Area
- GBP - Great Britain Pound

## Category Codes

These are strings identifying the category of a data series.

Examples:

- GDP - Gross Domestic Product
- CPI - Consumer Price Index
- FXXR - Foreign Exchange Returns

These are often appended with an additional string to identify specific transformations,
adjustments or other characteristics of the data series.

## End-of-Period (EOP) lags

These are integers that specify the number of days between the last day in the period and
the publish date of the corresponding data point.

For example, if the series reports monthly data, and the last day of the month would be
the end of the period, then the EOP lag would be 0. When data is published for
the next date in the series, the EOP lag would be 1.

## Mean-of-Period (MOP) lags

...
