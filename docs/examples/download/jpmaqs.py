"""example/macrosynergy/download/jpmaqs.py"""
from macrosynergy.download import JPMaQSDownload

cids = ["AUD", "BRL", "CAD"]
xcats = ["RIR_NSA", "FXXR_NSA", "FXXR_VT10", "DU05YXR_NSA"]
metrics = "all"
start_date: str = "2023-01-01"
end_date: str = "2023-03-20"

client_id = "DQ_CLIENT_ID"
client_secret = "DQ_CLIENT_SECRET"

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
