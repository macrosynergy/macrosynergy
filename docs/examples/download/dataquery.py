"""example/macrosynergy/download/dataquery.py"""

# %% [markdown]
# ## Imports
# %%

from macrosynergy.download import DataQueryInterface

# %% [markdown]
# ## Setting up the client_id and client_secret
# %%

# Set the client_id and client_secret.
# Ideally, these should not be stored in the script
# but rather in a config file or as environment variables
client_id = "DQ_CLIENT_ID"
client_secret = "DQ_CLIENT_SECRET"

# %% [markdown]
# ## Setting up proxies 
# %%

# if needed (for example, if you are behind a firewall. Useful for institutional users)
proxies = {
    "https": "http://proxy.example.com:8080",
}


# %% [markdown]
# ## List DataQuery expressions
# %%

expressions = [
    "DB(JPMAQS,USD_EQXR_VT10,value)",
    "DB(JPMAQS,AUD_EXALLOPENNESS_NSA_1YMA,value)",
]

# %% [markdown]
# ## Downloading the data - with a context manager
# %%

with DataQueryInterface(
    client_id=client_id,
    client_secret=client_secret,
) as dq:
    assert dq.check_connection(verbose=True)

    # optionally get the catalogue within the context manager
    group_catalogue = dq.get_catalogue()

    data = dq.download_data(
        expressions=expressions,
        start_date="2020-01-25",
        end_date="2023-02-05",
        show_progress=True,
    )

print(f"Succesfully downloaded data for {len(data)} expressions.")

print(f"Downloaded a catalogue with {len(group_catalogue)} expressions.")
