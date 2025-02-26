from abc import ABC, abstractmethod
import pandas as pd

from macrosynergy.management.types.qdf.classes import QuantamentalDataFrame


class BaseTransformer(ABC):
    @abstractmethod
    def transform(self, data, **kwargs):
        pass

    @staticmethod
    def get_mapping_value(d, keys):
        if not isinstance(keys, list):
            return d[keys]

        for key in keys:
            d = d[key]
        return d


class DataFrameTransformer(BaseTransformer):
    """
    This transformer is used to transform a pandas DataFrame into a QuantamentalDataFrame.
    """
    def transform(self, data, mapping):
        if data.empty:
            raise ValueError("DataFrame is empty.")
        if not isinstance(mapping, dict):
            raise TypeError("Mapping must be a dictionary.")

        # Case 1 - Mapping is a dictionary of columns to rename.
        if set(mapping.keys()).issubset(data.columns):
            df = data.rename(columns=mapping).drop(
                columns=set(data.columns) - set(mapping.keys())
            )
            return QuantamentalDataFrame(df)

        # Case 2 - Dataframe has some multiindexing in the columns
        elif isinstance(data.columns, pd.MultiIndex):
            df = data.stack(level=[0, 1]).reset_index()
            if "value" not in mapping.values():
                mapping[0] = "value"
            df.rename(columns=mapping, inplace=True)
            return QuantamentalDataFrame(df)

        # Case 3 - Dataframe has some multiindexing in the indexes
        elif isinstance(data.index, pd.MultiIndex):
            df = data.stack(level=0).reset_index()
            if "value" not in mapping.values():
                mapping[0] = "value"
            df.rename(columns=mapping, inplace=True)
            return QuantamentalDataFrame(df)
        
        # Case 4 - Need to split into xcat and cid
        else:
            raise ValueError(
                "Mapping keys must be a subset of the DataFrame columns and names" 
                " of indexes/multiindexes."
            )

class JSONTransformer(BaseTransformer):
    """
    This transformer is used to transform a list of JSON objects into a QuantamentalDataFrame.

    Any JSON object is assumed to be a dictionary in python.
    """
    def transform(self, data, mapping):

        rows = []

        for json in data:
            cid = BaseTransformer.get_mapping_value(json, mapping["cid"])
            xcat = BaseTransformer.get_mapping_value(json, mapping["xcat"])
            time_series = BaseTransformer.get_mapping_value(json, mapping["time-series"])
            for date, value in time_series:
                rows.append({"real_date": date, "cid": cid, "xcat": xcat, "value": value})

        df = pd.DataFrame(rows)
        df["real_date"] = pd.to_datetime(df["real_date"])

        return QuantamentalDataFrame(df)

def transform_to_qdf(data, **kwargs):
    if isinstance(data, pd.DataFrame):
        return DataFrameTransformer().transform(
            data, mapping=kwargs.get("mapping", {})
        )
    elif isinstance(data, list) and all(isinstance(x, dict) for x in data):
        return JSONTransformer().transform(
            data, mapping=kwargs.get("mapping", {})
        )
    else:
        raise TypeError(
            "data format has not been implemented, please create a new transformer "
            "class that inherits from BaseTransformer and use the transform method "
            "directly."
        )


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["s", "g", "q"],
            "c": ["baap", "boop", "beep"],
            "date": ["2021-01-01", "2021-01-02", "2021-01-03"],
        }
    )

    new_df = transform_to_qdf(
        df, mapping={"a": "value", "b": "cid", "c": "xcat", "date": "real_date"}
    )

    print(new_df)

    import os
    from macrosynergy.download import DataQueryInterface

    client_id: str = os.getenv("DQ_CLIENT_ID")
    client_secret: str = os.getenv("DQ_CLIENT_SECRET")

    expressions = ["DB(CFX,GBP,)", "DB(CFX,USD,)"]

    with DataQueryInterface(
        client_id=client_id,
        client_secret=client_secret,
    ) as dq:
        assert dq.check_connection(verbose=True)

        data = dq.download_data(
            expressions=expressions,
            start_date="2024-01-25",
            end_date="2024-02-05",
            show_progress=True,
        )

    new_df = transform_to_qdf(data, mapping = {
        "cid": ["attributes", 0, "expression"],
        "xcat": ["attributes", 0, "expression"],
        "time-series": ["attributes", 0, "time-series"]
    })

    print(new_df)
