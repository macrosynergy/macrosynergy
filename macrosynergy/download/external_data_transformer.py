from abc import ABC, abstractmethod
import pandas as pd

from macrosynergy.management.types.qdf.classes import QuantamentalDataFrame


class BaseTransformer(ABC):
    @abstractmethod
    def transform(self, data, **kwargs):
        pass


class DataFrameTransformer(BaseTransformer):
    def transform(self, data, mapping):
        if data.empty:
            raise ValueError("DataFrame is empty.")
        if not isinstance(mapping, dict):
            raise TypeError("Mapping must be a dictionary.")

        # Case 1 - Mapping is a dictionary of columns to rename.
        if set(mapping.keys()).issubset(data.columns):
            data = data.rename(columns=mapping).drop(
                columns=set(data.columns) - set(mapping.keys())
            )
            return QuantamentalDataFrame(data)

        # Case 2 - Dataframe has some multiindexing in the columns
        elif isinstance(data.columns, pd.MultiIndex):
            df.stack(level=[0, 1]).reset_index().rename(columns=mapping)
            return QuantamentalDataFrame(data)

        # Case 3 - Dataframe has some multiindexing in the indexes
        elif isinstance(data.index, pd.MultiIndex):
            df.stack(level=0).reset_index().rename(columns=mapping)
            return QuantamentalDataFrame(data)
        
        # Case 4 - Need to split into xcat and cid
        else:
            raise ValueError(
                "Mapping keys must be a subset of the DataFrame columns and names" 
                " of indexes/multiindexes."
            )



def transform_to_qdf(data, **kwargs):
    if isinstance(data, pd.DataFrame):
        return DataFrameTransformer().transform(
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
