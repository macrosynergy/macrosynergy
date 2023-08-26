# `macrosynergy.download`

## Description

This subpackage contains functionalities for downloading JPMaQS data from the JPMorgan DataQuery API.
Its components fall into two broad categories:

- Downloading data from the API as time-series JSONs.
- Converting and wrangling the JSONs into a JPMaQS Quantamental DataFrame.

## Functionality

### User-facing functions:

## Design principles

The key guiding principle of this subpackage can be summarized as follows: 
- Download JPMaQS data from the API as a QDF, while still allowing saving the raw JSONs.
- Allow users to download any DataQuery expression, even though the JPMaQS specific functions may not apply to all expressions.
- Allow OAuth and Certificate authentication.
- Have detailed, well documented errors and break conditions.
- Allow for concurrent downloads, adding proxy settings/alternate identity certificates, and manage retries and timeouts.
- Have a simple, concise, and easy-to-use interface.


## Implementation specifics

###
