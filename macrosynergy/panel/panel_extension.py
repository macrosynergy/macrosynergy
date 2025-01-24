from typing import List
import warnings

import pandas as pd
from datetime import datetime

from macrosynergy.management.types.qdf.classes import QuantamentalDataFrame
from macrosynergy.management.utils.df_utils import reduce_df


class PanelExtension(object):
    def __init__(
        self,
        df: pd.DataFrame,
        xcats: List[str],
        cids: List[str],
        start: str,
        end: str,
        impute_method: str = "mean",
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
        if not isinstance(start, str):
            raise TypeError("start must be a string")
        else:
            try:
                start = datetime.strptime(start, "%Y-%m-%d")
            except ValueError:
                raise ValueError("start must be in the format 'YYYY-MM-DD'")
        if not isinstance(end, str):
            raise TypeError("end must be a string")
        else:
            try:
                end = datetime.strptime(end, "%Y-%m-%d")
            except ValueError:
                raise ValueError("end must be in the format 'YYYY-MM-DD'")
        if not isinstance(impute_method, str):
            raise TypeError("impute_method must be a string")
        elif impute_method not in ["mean", "median"]:
            raise ValueError("impute_method must be either 'mean' or 'median'")
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
        self.impute_method = impute_method
        self.min_cids = min_cids
        self.postfix = postfix
        self.imputed = False
        self.blacklist = {xcat: {} for xcat in xcats}

        complete_df = reduce_df(
            df, xcats=xcats, start=self.start, end=self.end
        ).dropna()
        complete_df = QuantamentalDataFrame(complete_df)
        _as_categorical = complete_df.InitializedAsCategorical

        business_dates = (
            pd.date_range(start=self.start, end=self.end, freq="B")
            .strftime("%Y-%m-%d")
            .tolist()
        )

        # Create a multiindex from the product of business dates, xcats and cids
        full_idx = pd.MultiIndex.from_product(
            [business_dates, xcats, cids], names=["real_date", "xcat", "cid"]
        )

        # Create a dataframe with the full index and the value column filled with the values in complete_df and if it doesn't exist, fill it with imputed value and then nan
        self.df = pd.DataFrame(index=full_idx).reset_index()
        self.df["real_date"] = pd.to_datetime(self.df["real_date"])
        self.df = self.df.merge(
            complete_df,
            how="outer",
            on=["real_date", "xcat", "cid"],
            suffixes=("", ""),
        )

        self.df["value"] = self.df.groupby(["real_date", "xcat"])["value"].transform(
            self.get_impute_function
        )
        # Get difference between the two dataframes
        diff = complete_df.merge(
            self.df, how="outer", on=["real_date", "xcat", "cid"], suffixes=("", "_")
        )
        diff["imputed"] = diff["value"].isnull() & ~diff["value_"].isnull()
        # diff = diff[diff["imputed"]].drop(columns=["value", "value_"])
        self.generate_blacklist(diff)

        if not self.imputed:
            warnings.warn(
                "No imputation was performed. Consider changing the impute_method or min_cids."
            )
        self.df = reduce_df(
            self.df, cids=cids, xcats=xcats, start=self.start, end=self.end
        )

        self.df.dropna(inplace=True)
        self.df = QuantamentalDataFrame(self.df, categorical=_as_categorical)

    def get_impute_function(self, group):
        if self.impute_method == "mean":
            return self.impute_mean(group)
        elif self.impute_method == "median":
            return self.impute_median(group)

    def impute_mean(self, group):
        if group.count() >= self.min_cids:
            self.imputed = True
            return group.fillna(group.mean())
        else:
            return group

    def impute_median(self, group):
        if group.count() >= self.min_cids:
            self.imputed = True
            return group.fillna(group.median())
        else:
            return group

    def generate_blacklist(self, diff: pd.DataFrame):
        grouped = diff.groupby(["xcat", "cid"])

        for (xcat, cid), group in grouped:
            imputed_group = group[group["imputed"]]
            if imputed_group.empty:
                continue

            imputed_group = imputed_group.sort_values("real_date")

            consecutive_groups = (imputed_group.index.to_series().diff() == 1).cumsum()

            date_ranges = (
                imputed_group.groupby(consecutive_groups)
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
        return self.blacklist if xcat is None else self.blacklist[xcat]

    def return_filled_df(self):
        return self.df
