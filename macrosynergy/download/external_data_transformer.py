import pandas as pd

class ExternalDataTransformer(object):
    def __init__(self):
        pass

    def transform(self, data: object, mapping) -> pd.DataFrame:
        """
        Transform the external data to the desired format.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    

class DataframeTransformer(ExternalDataTransformer):
    def __init__(self):
        super().__init__()

    def transform(self, df: pd.DataFrame, mapping) -> pd.DataFrame:
        """
        Transform the external data to the desired format.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not isinstance(mapping, dict):
            raise TypeError("mapping must be a dictionary")
        if df.empty:
            raise ValueError("df cannot be empty")
        
        # Possible dataframe inputs
        # 1. df with columns that need to be renamed
        # 2. df with columns that need to be dropped
        # 3. df with columns that need to be renamed and dropped
        # 4. df where columns and rows need to be pivoted
        # 5. df where columns and rows need to be pivoted and renamed
        # 6. df where columns and rows need to be pivoted, renamed and dropped

        # For cid and xcat columns, possibility of having to split by a character after
        # transformation of dataframe

