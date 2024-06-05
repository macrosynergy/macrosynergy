import pandas as pd
from macrosynergy.management.qdf.methods import qdf_to_df_dict, expression_df_to_df_dict
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import get_cid, get_xcat, concat_qdfs
import joblib
import glob
import os
from tqdm import tqdm
from typing import Any, Dict, List
import pickle


class Loader:
    @staticmethod
    def load_single_csv_from_disk(csv_file: str) -> pd.DataFrame:
        """
        Load a single csv file from disk.
        """
        return pd.read_csv(csv_file, parse_dates=["real_date"]).set_index("real_date")

    @staticmethod
    def load_single_qdf_from_disk(csv_file: str) -> QuantamentalDataFrame:
        """
        Load a single qdf file from disk.
        """
        ticker = os.path.basename(csv_file).split(".")[0]
        return pd.read_csv(csv_file, parse_dates=["real_date"]).assign(
            cid=get_cid(ticker), xcat=get_xcat(ticker)
        )

    @staticmethod
    def get_csv_files(path: str) -> List[str]:
        """
        Load all the csv files in the path recursively.
        """
        listx = sorted(glob.glob(path + "/**/*.csv", recursive=True))
        if not listx:
            raise FileNotFoundError(f"No CSV files found in {path}")
        return listx

    @staticmethod
    def load_csv_batch_from_disk(
        csv_files: List[str],
    ) -> pd.DataFrame:
        """
        Load a batch of csv files from disk, only for CSV (non-qdf) files.
        """
        return pd.concat(
            joblib.Parallel()(
                joblib.delayed(Loader.load_single_csv_from_disk)(csv_file)
                for csv_file in tqdm(csv_files, desc="Loading CSVs", leave=False)
            ),
            axis=1,
        )

    @staticmethod
    def load_qdf_batch_from_disk(
        csv_files: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """
        Load a batch of qdf files from disk.
        """
        return qdf_to_df_dict(
            concat_qdfs(
                joblib.Parallel()(
                    joblib.delayed(Loader.load_single_qdf_from_disk)(csv_file)
                    for csv_file in tqdm(csv_files, desc="Loading QDFs", leave=False)
                )
            )
        )

    @staticmethod
    def load_csvs_from_disk(
        path: str,
        show_progress: bool = True,
        batch_size: int = 100,
    ) -> pd.DataFrame:
        """
        All the csv files in the path recursively and try to load them as time series data.
        """
        csvs_list = Loader.get_csv_files(path)
        csv_batches = [
            csvs_list[i : i + batch_size] for i in range(0, len(csvs_list), batch_size)
        ]

        dfs = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(Loader.load_csv_batch_from_disk)(csv_batch)
            for csv_batch in tqdm(
                csv_batches, disable=not show_progress, desc="Loading CSVs"
            )
        )

        return pd.concat(dfs, axis=1)

    @staticmethod
    def load_csv_from_disk_as_df_dict(
        path: str,
        show_progress: bool = True,
        batch_size: int = 100,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all the csv files in the path recursively.
        """
        return expression_df_to_df_dict(
            Loader.load_csvs_from_disk(
                path=path,
                show_progress=show_progress,
                batch_size=batch_size,
            )
        )

    @staticmethod
    def load_qdfs_from_disk_as_df_dict(
        path: str,
        show_progress: bool = True,
        batch_size: int = 100,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all the qdf files in the path recursively.
        """
        csvs_list = Loader.get_csv_files(path)
        csv_batches = [
            csvs_list[i : i + batch_size] for i in range(0, len(csvs_list), batch_size)
        ]

        df_dicts: List[Dict[str, pd.DataFrame]] = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(Loader.load_qdf_batch_from_disk)(csv_batch)
            for csv_batch in tqdm(
                csv_batches, disable=not show_progress, desc="Loading QDFs"
            )
        )
        df_dict: Dict[str, pd.DataFrame] = {}
        for _ in range(len(df_dicts)):
            d = df_dicts.pop()
            for k, v in d.items():
                if k in df_dict:
                    df_dict[k] = pd.concat([df_dict[k], v], axis=1)
                else:
                    df_dict[k] = v

        for metric in df_dict.keys():
            df_dict[metric] = df_dict[metric][sorted(df_dict[metric].columns)]

        return df_dict

    @staticmethod
    def load_pkl_from_disk(path: str) -> Any:
        """
        Load a pickle file from disk.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_pkl(path: str) -> Any:
        """
        Load a pickle file from disk.
        """
        obj = Loader.load_pkl_from_disk(path)
        if isinstance(obj, pd.DataFrame):
            return expression_df_to_df_dict(obj)
        elif isinstance(obj, dict):
            return obj
        else:
            raise NotImplementedError(
                f"Invalid object type: {type(obj)}. Must be either `pd.DataFrame` or "
                "`Dict[str, pd.DataFrame]`"
            )

    @staticmethod
    def load_from_disk(
        path: str,
        format: str = "csvs",
    ) -> Dict[str, pd.DataFrame]:
        """
        Load a `QuantamentalDataFrame` from disk.

        Parameters
        :param <str> path: The path to the directory containing the data.
        :param <str> format: The format of the data. Options are "csvs", "csv",
            "pkl", or "qdfs".
        """

        def load_single_csv_from_disk_as_df_dict(
            csv_file: str,
        ) -> Dict[str, pd.DataFrame]:
            """
            Load a single csv file from disk.
            """
            return expression_df_to_df_dict(Loader.load_single_csv_from_disk(csv_file))

        if format == "pkl":
            return Loader.load_pkl(path)

        fmt_dict = {
            "csv": load_single_csv_from_disk_as_df_dict,
            "csvs": Loader.load_csv_from_disk_as_df_dict,
            "qdfs": Loader.load_qdfs_from_disk_as_df_dict,
        }
        if format not in fmt_dict:
            raise NotImplementedError(
                f"Invalid format: {format}. Options are {fmt_dict.keys()}"
            )
        return fmt_dict[format](path)
