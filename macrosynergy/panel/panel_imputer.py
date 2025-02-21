from typing import List
import warnings

import pandas as pd
from datetime import datetime

from macrosynergy.management.types.qdf.classes import QuantamentalDataFrame
from macrosynergy.management.utils.df_utils import reduce_df


class BasePanelImputer:
    """
    Base class for imputing missing values in a panel DataFrame. Defines an overall
    structure for how the imputation should be performed, without the imputation method.
    Separate subclasses should be created for each imputation method, which will have a
    defined impute technique.

    Parameters
    ----------
    df : ~pandas.DataFrame
        DataFrame containing the panel data.
    xcats : List[str]
        List of extended categories.
    cids : List[str]
        List of cross sections.
    start : str
        Start date in ISO format.
    end : str
        End date in ISO format.
    min_cids : int
        Minimum number of cross sections required to perform imputation on a specific
        real date. Default is len(cids) // 2.
    postfix : str
        Postfix to add to the extended categories after imputation. Default is "F".
    """

    def __init__(
        self,
        df: pd.DataFrame,
        xcats: List[str],
        cids: List[str],
        start: str = None,
        end: str = None,
        min_cids: int = None,
        postfix: str = "F",
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        else:
            df = QuantamentalDataFrame(df)
        if not isinstance(xcats, list):
            raise TypeError("xcats must be a list")
        if not isinstance(cids, list):
            raise TypeError("cids must be a list")
        if start is None:
            start = df["real_date"].min().strftime("%Y-%m-%d")
        if not isinstance(start, str):
            raise TypeError("start must be a string")
        else:
            try:
                start = datetime.strptime(start, "%Y-%m-%d")
            except ValueError:
                raise ValueError("start must be in the format 'YYYY-MM-DD'")
        if end is None:
            end = df["real_date"].max().strftime("%Y-%m-%d")
        if not isinstance(end, str):
            raise TypeError("end must be a string")
        else:
            try:
                end = datetime.strptime(end, "%Y-%m-%d")
            except ValueError:
                raise ValueError("end must be in the format 'YYYY-MM-DD'")
        if min_cids is None:
            min_cids = len(cids) // 2
        elif not isinstance(min_cids, int) or min_cids < 0:
            raise TypeError("min_cids must be a non-negative integer")
        if not isinstance(postfix, str):
            raise TypeError("postfix must be a string")

        if min_cids > df["cid"].nunique():
            raise ValueError(
                "min_cids must be less than or equal to the number of unique cids in df"
            )

        self.cids = cids
        self.xcats = xcats
        self.start = start
        self.end = end
        self.min_cids = min_cids
        self.postfix = postfix
        self.imputed = False
        self.blacklist = {xcat: {} for xcat in xcats}

        self.df = QuantamentalDataFrame(
            reduce_df(df, xcats=xcats, start=self.start, end=self.end).dropna()
        )
        self._as_categorical = df.InitializedAsCategorical

    def impute(self):
        """
        Returns the imputed DataFrame.
        """
        business_dates = (
            pd.date_range(start=self.start, end=self.end, freq="B")
            .strftime("%Y-%m-%d")
            .tolist()
        )

        full_idx = pd.MultiIndex.from_product(
            [business_dates, self.xcats, self.cids], names=["real_date", "xcat", "cid"]
        )

        imputed_df = pd.DataFrame(index=full_idx).reset_index()
        imputed_df["real_date"] = pd.to_datetime(imputed_df["real_date"])
        imputed_df = imputed_df.merge(
            self.df,
            how="outer",
            on=["real_date", "xcat", "cid"],
            suffixes=("", ""),
        )

        imputed_df["value"] = imputed_df.groupby(["real_date", "xcat"], observed=True)[
            "value"
        ].transform(self.get_impute_function)

        diff = self.df.merge(
            imputed_df, how="outer", on=["real_date", "xcat", "cid"], suffixes=("", "_")
        )
        diff["imputed"] = diff["value"].isnull() & ~diff["value_"].isnull()

        self.generate_blacklist(diff)

        if not self.imputed:
            warnings.warn(
                "No imputation was performed. Consider changing the impute_method or min_cids."
            )

        imputed_df = reduce_df(
            imputed_df, cids=self.cids, xcats=self.xcats, start=self.start, end=self.end
        )
        imputed_df["xcat"] = imputed_df["xcat"] + self.postfix
        imputed_df.dropna(inplace=True)
        return QuantamentalDataFrame(imputed_df, categorical=self._as_categorical)

    def get_impute_function(self, group):
        """
        Abstract method that should be implemented in a subclass. Defines the imputation
        technique to be used on a group of values.
        """
        raise NotImplementedError(
            "get_impute_function must be implemented in a subclass"
        )

    def generate_blacklist(self, diff: pd.DataFrame):
        """
        Generates a dictionary of cross sections and dates that have been imputed in the
        same format as a blacklist dictionary. For each cross section it stores a date
        range where imputation has been performed.

        Parameters
        ----------
        diff : ~pandas.DataFrame
            DataFrame containing the differences between the original and imputed
            DataFrames.
        """
        grouped = diff.groupby(["xcat", "cid"], observed=True)

        for (xcat, cid), group in grouped:
            group = group.sort_values(["real_date"]).reset_index(drop=True)
            imputed_group = group[group["imputed"]]
            if imputed_group.empty:
                continue

            consecutive_groups = (imputed_group.index.to_series().diff() != 1).cumsum()

            date_ranges = (
                imputed_group.groupby(consecutive_groups, observed=True)
                .agg(start=("real_date", "first"), end=("real_date", "last"))
                .reset_index(drop=True)
            )

            if len(date_ranges) == 1:
                self.blacklist[xcat][cid] = (
                    date_ranges.iloc[0]["start"],
                    date_ranges.iloc[0]["end"],
                )
            else:
                for i, row in date_ranges.iterrows():
                    self.blacklist[xcat][f"{cid}_{i+1}"] = (row["start"], row["end"])

    def return_blacklist(self, xcat: str = None):
        if not self.imputed:
            warnings.warn(
                "No imputation was performed. The blacklist is empty.", RuntimeWarning
            )
        return self.blacklist if xcat is None else self.blacklist[xcat]


class MeanPanelImputer(BasePanelImputer):
    """
    Imputer class that fills missing values with the global cross-sectional mean.
    If the group has less than min_cids non-missing values, the group is left as is.
    """

    def get_impute_function(self, group):
        if group.count() >= self.min_cids:
            self.imputed = True
            return group.fillna(group.mean())
        return group


class MedianPanelImputer(BasePanelImputer):
    """
    Imputer class that fills missing values with the global cross-sectional median.
    If the group has less than min_cids non-missing values, the group is left as is.
    """

    def get_impute_function(self, group):
        if group.count() >= self.min_cids:
            self.imputed = True
            return group.fillna(group.median())
        return group
