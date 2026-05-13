import os
import logging
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def save_data(path: str | Path, data: Any) -> Path:
    """Save data locally, choosing the file format based on the data type.

    DataFrames are saved as Parquet, dicts and lists as JSON, and all other
    objects as pickle. The extension in "path" is replaced with the appropriate
    suffix regardless of what was originally supplied.

    Parameters
    ----------
    path : str or Path
        Destination path. The extension is overridden based on the data type.
    data : Any
        Object to persist. Its type determines the output format.

    Returns
    -------
    Path
        Path of the file that was written, with the correct extension applied:
        ".parquet" for DataFrames, ".json" for dicts and lists, and ".pkl" for
        everything else.
    """
    path = Path(path)
    if not path.parent.exists():
        logger.error("Directory does not exist: %s", path.parent)
        raise FileNotFoundError(f"Directory does not exist: {path.parent}")

    if isinstance(data, pd.DataFrame):
        out = path.with_suffix(".parquet")
        logger.info("Saving DataFrame as Parquet to %s", out)
        data.to_parquet(out)
    elif isinstance(data, (dict, list)):
        out = path.with_suffix(".json")
        logger.info("Saving %s as JSON to %s", type(data).__name__, out)
        with open(out, "w") as f:
            json.dump(data, f)
    else:
        out = path.with_suffix(".pkl")
        logger.warning(
            "Saving %s as pickle to %s. Pickle files are not portable across "
            "Python versions or environments.",
            type(data).__name__,
            out,
        )
        with open(out, "wb") as f:
            pickle.dump(data, f)

    logger.info("Data saved successfully to %s", out)
    return out


def load_data(path: str | Path) -> Any:
    """Load data previously saved by "save_data", inferring the format from the file extension.

    If "path" has no extension, the function probes for ".parquet", ".json",
    and ".pkl" files in that order and loads the first match found.

    Parameters
    ----------
    path : str or Path
        Path to the saved file. The extension may be omitted if the file was
        written by "save_data".

    Returns
    -------
    Any
        The deserialized data: a DataFrame for ".parquet" files, a dict or list
        for ".json" files, and the original pickled object for ".pkl" files.
    """
    path = Path(path)
    if not path.exists():
        logger.debug("No file at %s, probing for known extensions.", path)
        for suffix in (".parquet", ".json", ".pkl"):
            candidate = path.with_suffix(suffix)
            if candidate.exists():
                logger.debug("Found %s, using it.", candidate)
                path = candidate
                break
        else:
            logger.error("No cached file found at %s", path)
            raise FileNotFoundError(f"No cached file found at {path}")

    logger.info("Loading data from %s", path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    else:
        logger.warning(
            "Loading pickle file %s. Pickle files are not portable across "
            "Python versions or environments.",
            path,
        )
        with open(path, "rb") as f:
            return pickle.load(f)
