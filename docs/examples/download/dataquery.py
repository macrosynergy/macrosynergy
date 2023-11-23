"""example/macrosynergy/download/dataquery.py"""
from macrosynergy.download import DataQueryInterface

client_id = "DQ_CLIENT_ID"
client_secret = "DQ_CLIENT_SECRET"

expressions = [
    "DB(JPMAQS,USD_EQXR_VT10,value)",
    "DB(JPMAQS,AUD_EXALLOPENNESS_NSA_1YMA,value)",
]

with DataQueryInterface(
    client_id=client_id,
    client_secret=client_secret,
) as dq:
    assert dq.check_connection(verbose=True)

    data = dq.download_data(
        expressions=expressions,
        start_date="2020-01-25",
        end_date="2023-02-05",
        show_progress=True,
    )

print(f"Succesfully downloaded data for {len(data)} expressions.")
