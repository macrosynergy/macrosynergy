"""
Module for downloading data from the JPMorgan Fusion API.
"""

from macrosynergy.management.types import QuantamentalDataFrame
import pandas as pd
import tempfile

import fusion

from typing import Optional

# FUSION_ROOT_URL = "rootURLPlaceholder"
# FUSION_RESOURCE_ID = "resourceIdPlaceholder"
# FUSION_CLIENT_ID = "clientIdPlaceholder"
# FUSION_AUTH_URL = "authURLPlaceholder"


def convert_ticker_based_parquet_to_qdf(
    df: pd.DataFrame, categorical: bool = False
) -> pd.DataFrame:
    """
    Convert Parquet DataFrame with ticker entries to a QDF with cid & xcat columns.
    """
    df[["cid", "xcat"]] = df["ticker"].str.split("_", n=1, expand=True)
    df = df.drop(columns=["ticker"])
    df = QuantamentalDataFrame(df, categorical=categorical)
    return df


class JPMaQSFusionAdapter:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        root_url: Optional[str] = None,
        resource: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        auth_url: Optional[str] = None,
        download_folder: str = None,
    ):
        self.username = username
        self.password = password
        self.root_url = root_url
        self.resource = resource
        self.client_id = client_id
        self.client_secret = client_secret
        self.auth_url = auth_url
        self.download_folder = download_folder
        self.setup_fusion_client()

    def setup_fusion_client(self, download_folder: str = None):
        download_folder = download_folder or self.download_folder
        assert download_folder is not None, "Download folder must be specified."
        self.fusion = fusion.Fusion(
            root_url=self.root_url,
            download_folder=download_folder,
            credentials=fusion.FusionCredentials(
                username=self.username,
                password=self.password,
                client_id=self.client_id,
                resource=self.resource,
                auth_url=self.auth_url,
                grant_type=(
                    "password"
                    if (self.username and self.password)
                    else "client_credentials"
                ),
            ),
        )

    def teardown_fusion_client(self):
        self.fusion = None

    def list_datasets(self, product: str = "JPMAQS_DATA") -> pd.DataFrame:
        """
        List datasets available in the Fusion API for a given product.
        """
        return self.fusion.list_datasets(product=product)

    def _download(
        self,
        dataset: str,
        dt_str: str = None,  # YYYYMMDD or YYYYMMD1:YYYYMMD2 (20240101:20240131)
        show_progress: bool = False,
        **kwargs,
    ):
        """
        Wrapper for the `fusion.Fusion.download` method.
        """
        kwargs["return_paths"] = True
        # TODO Fusion SDK does not check dt_str
        with tempfile.TemporaryDirectory() as temp_dir:
            self.setup_fusion_client(temp_dir)
            snapshots_paths = self.fusion.download(
                dataset=dataset,
                dt_str=dt_str,
                show_progress=show_progress,
                **kwargs,
            )

            self.teardown_fusion_client()
            paths = [ptuple[1] for ptuple in snapshots_paths]
            # TODO filtering is most likely not needed, but just in case?
            # paths = [path for path in paths if path.endswith(".parquet")]

            dfs_list = [
                convert_ticker_based_parquet_to_qdf(pd.read_parquet(path))
                for path in paths
            ]

        return QuantamentalDataFrame.from_qdf_list(dfs_list)

    def download_full_snapshot(
        self,
        dataset: str = "JPMAQS_DATA_FULL_SNAPSHOT",
        **kwargs,
    ):
        """
        Download the full snapshot of the dataset.
        """
        return self._download(dataset=dataset, **kwargs)


if __name__ == "__main__":
    # Example usage
    # fusion_adapter = JPMaQSFusionAdapter(
    #     username="your_username",
    #     password="your_password",
    # )
    # datasets = fusion_adapter.list_datasets()
    # print(datasets)
    # df = fusion_adapter.download_full_snapshot(dataset="JPMAQS_DATA_FULL_SNAPSHOT")
    # print(df.head())
    import json

    creds = "./data/fusion_client_credentials.json"
    download_folder = "./data/fusion_downloads"
    with open(creds) as f:
        credentials = json.load(f)

    client_id = credentials["client_id"]
    client_secret = credentials["client_secret"]
    resource = credentials["resource"]
    application_name = credentials["application_name"]
    root_url = credentials["root_url"] + "/"
    auth_url = credentials["auth_url"]
    username = None
    password = None
    fs = fusion.Fusion(
        root_url=root_url,
        download_folder=download_folder,
        credentials=fusion.FusionCredentials(
            # username=username,
            # password=password,
            client_id=client_id,
            client_secret=client_secret,
            resource=resource,
            auth_url=auth_url,
            # grant_type="password",
            grant_type="client_credentials",
        ),
    )
    x = fs.list_datasets(catalog="common")
    print(x)
