"""example/macrosynergy/download/jpmaqs.py"""

cids = [
    "AUD",
    "BRL",
    "CAD",
    "CHF",
    "CLP",
    "CNY",
    "COP",
    "CZK",
    "DEM",
    "ESP",
    "EUR",
    "FRF",
    "GBP",
    "USD",
]
xcats = [
    "RIR_NSA",
    "FXXR_NSA",
    "FXXR_VT10",
    "DU05YXR_NSA",
    "DU05YXR_VT10",
]
metrics = "all"
start_date: str = "2023-01-01"
end_date: str = "2023-03-20"

client_id = os.getenv("DQ_CLIENT_ID")
client_secret = os.getenv("DQ_CLIENT_SECRET")

with JPMaQSDownload(
    client_id=client_id,
    client_secret=client_secret,
    debug=True,
) as jpmaqs:
    data = jpmaqs.download(
        cids=cids,
        xcats=xcats,
        metrics=metrics,
        start_date=start_date,
        get_catalogue=True,
        end_date=end_date,
        show_progress=True,
        suppress_warning=False,
        report_time_taken=True,
    )

    print(data.head())
