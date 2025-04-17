"""
Module for downloading data from the JPMorgan Fusion API.
"""

from macrosynergy.management.types import QuantamentalDataFrame
import pandas as pd
import tempfile

import fusion

from typing import Optional, Union, List

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
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        root_url: Optional[str] = None,
        resource: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        auth_url: Optional[str] = None,
        credentials: Optional[Union[str, fusion.FusionCredentials]] = None,
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
        self.credentials = credentials
        self.setup_fusion_client()

    def setup_fusion_client(self, download_folder: str = None):
        download_folder = download_folder or self.download_folder
        assert download_folder is not None, "Download folder must be specified."
        creds = fusion.FusionCredentials(
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
        )
        creds = self.credentials or creds
        root_url_arg = dict(root_url=self.root_url) if self.root_url else {}
        self.fusion = fusion.Fusion(
            **root_url_arg, download_folder=download_folder, credentials=creds
        )

    def teardown_fusion_client(self):
        self.fusion = None

    def list_datasets(self, product: str = "JPMAQS", **kwargs) -> pd.DataFrame:
        """
        List datasets available in the Fusion API for a given product.
        """
        return self.fusion.list_datasets(product=product, **kwargs)

    def _download(
        self,
        dataset: str,
        dt_str: str = "latest",  # YYYYMMDD or YYYYMMD1:YYYYMMD2 (20240101:20240131)
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
        catalog: str = "common",
        **kwargs,
    ):
        """
        Download the full snapshot of the dataset.
        """
        datasets = self.list_datasets()
        identifiers: List[str] = datasets["identifier"].tolist()
        # return self._download(dataset=dataset, **kwargs)
        results = []
        for dataset in identifiers:
            try:
                df = self._download(dataset=dataset, catalog=catalog, **kwargs)
                results.append(df)
            except Exception as e:
                print(f"Failed to download {dataset}: {e}")

        if len(results) == 0:
            raise ValueError("No datasets were downloaded successfully.")

        return results


if __name__ == "__main__":
    import json

    creds = "./data/fusion_client_credentials.json"
    download_folder = "./data/fusion_downloads"
    # with open(creds) as f:
    #     credentials = json.load(f)

    # client_id = credentials["client_id"]
    # client_secret = credentials["client_secret"]
    # resource = credentials["resource"]
    # application_name = credentials["application_name"]
    # root_url = credentials["root_url"] + "/"
    # auth_url = credentials["auth_url"]
    # username = None
    # password = None

    # fs = fusion.Fusion(
    #     root_url=root_url,
    #     download_folder=download_folder,
    #     credentials=fusion.FusionCredentials(
    #         # username=username,
    #         # password=password,
    #         client_id=client_id,
    #         client_secret=client_secret,
    #         resource=resource,
    #         auth_url=auth_url,
    #         # grant_type="password",
    #         grant_type="client_credentials",
    #     ),
    # )
    # x = fs.list_datasets(catalog="common")
    # print(x)

    jpmaqs_fusion = JPMaQSFusionAdapter(
        credentials=creds, download_folder=download_folder
    )
    jpmaqs_fusion.list_datasets()
    jpmaqs_fusion.download_full_snapshot()
