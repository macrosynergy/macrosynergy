import os
import logging
import json
import pickle
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def save_data(path: str | Path, data) -> Path:
    """Save data locally. Format is chosen by type: parquet, JSON, or pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        out = path.with_suffix(".parquet")
        data.to_parquet(out)
    elif isinstance(data, (dict, list)):
        out = path.with_suffix(".json")
        with open(out, "w") as f:
            json.dump(data, f)
    else:
        out = path.with_suffix(".pkl")
        with open(out, "wb") as f:
            pickle.dump(data, f)

    return out


def load_data(path: str | Path):
    """Load data saved by save_data. Infers format from file extension."""
    path = Path(path)
    if not path.exists():
        # Try known extensions if no extension provided
        for suffix in (".parquet", ".json", ".pkl"):
            candidate = path.with_suffix(suffix)
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"No cached file found at {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)
