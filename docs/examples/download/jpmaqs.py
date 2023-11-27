"""example/macrosynergy/download/jpmaqs.py"""

# Import the JPMaQSDownload class from the macrosynergy package
from macrosynergy.download import JPMaQSDownload

## Set the currency areas (cross-sectional identifiers) and categories

cids = ["AUD", "BRL", "CAD"]
xcats = ["RIR_NSA", "FXXR_NSA", "FXXR_VT10", "DU05YXR_NSA"]

# all metrics - value, grading, eop_lag, mop_lag
metrics = "all"

## Setting the start and end dates for the data
start_date = "2023-01-01"
end_date = "2023-03-20"

## Setting up proxies if needed (for example, if you are behind a firewall. Useful for institutional users)

proxies = {
    "https": "http://proxy.example.com:8080",
}

## Setting up the client_id and client_secret
# Set the client_id and client_secret.
# Ideally, these should not be stored in the script
# but rather in a config file or as environment variables
client_id = "DQ_CLIENT_ID"
client_secret = "DQ_CLIENT_SECRET"

## Creating an instance of the JPMaQSDownload class

jpmaqs_download = JPMaQSDownload(
    client_id=client_id,
    client_secret=client_secret,
    proxies=proxies,
)

## Downloading the cataogue as a list
catalogue = jpmaqs_download.get_catalogue()


## Downloading the data - with a context manager
with JPMaQSDownload(
    client_id=client_id,
    client_secret=client_secret,
) as jpmaqs:
    data = jpmaqs.download(
        cids=cids,
        xcats=xcats,
        metrics=metrics,
        start_date=start_date,
        end_date=end_date,
        get_catalogue=True,  # filters the requested data according to the catalogue
        show_progress=True,  # shows a progress bar
        suppress_warning=False,  # suppresses the warning about the data being incomplete
        report_time_taken=True,  # reports the time taken to download the data
    )

    # optionally get the catalogue within the context manager
    catalogue = jpmaqs.get_catalogue()

    print(data.head())
