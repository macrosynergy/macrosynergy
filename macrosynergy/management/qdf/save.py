import os
import pickle
from typing import Any, Dict
import pandas as pd
from macrosynergy.management.types import QuantamentalDataFrame


class Saver:
    @staticmethod
    def save_single_csv_to_disk(df: pd.DataFrame, path: str) -> None:
        """
        Save a single csv file to disk.
        """
        # reset index, and ensure the real_date is a column
        if not "real_date" in df.columns:
            df = df.reset_index()

        assert "real_date" in df.columns, "real_date column not found in the dataframe"
        assert len(df.columns) >= 2, "At least one ticker column is required"
        df.to_csv(path, index=False)

    @staticmethod
    def save_single_qdf_to_disk(
        df: QuantamentalDataFrame, path: str, infer_filename: bool = True
    ) -> None:
        """
        Save a single qdf file to disk.
        """
        # reset index, and ensure the real_date is a column
        if not "real_date" in df.columns:
            df = df.reset_index()

        assert "real_date" in df.columns, "real_date column not found in the dataframe"
        cidx = df["cid"].unique()
        xcatx = df["xcat"].unique()
        assert len(cidx) == 1, f"Multiple cids found: {cidx}"
        assert len(xcatx) == 1, f"Multiple xcats found: {xcatx}"
        if infer_filename:
            # check if the base path is a directory
            path = path if os.path.isdir(path) else os.path.dirname(path)
            path = os.path.join(path, f"{cidx[0]}_{xcatx[0]}.csv")

        df.to_csv(path, index=False)

    @staticmethod
    def save_pkl_to_disk(obj: Any, path: str) -> None:
        """
        Save an object to disk.
        """
        with open(path, "wb") as f:
            pickle.dump(obj, f)
