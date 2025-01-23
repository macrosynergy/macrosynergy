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
            suffixes=("", self.postfix),
        )

        self.df["value"] = self.df.groupby(["real_date", "xcat"])["value"].transform(
            self.get_impute_function
        )
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
            self.add_to_blacklist(timestamp=group.name[0], xcat=group.name[1])
            return group.fillna(group.mean())
        else:
            return group

    def impute_median(self, group):
        if group.count() >= self.min_cids:
            self.imputed = True
            self.add_to_blacklist(timestamp=group.name[0], xcat=group.name[1])
            return group.fillna(group.median())
        else:
            return group

    def add_to_blacklist(self, cid: str, timestamp: str, xcat: str):
        xcat_blacklist = self.blacklist[xcat]
        if cid not in xcat_blacklist and cid + "_1" not in xcat_blacklist:
            xcat_blacklist[cid] = [(timestamp, timestamp)]
        elif cid in xcat_blacklist:
            # If current timestamp is one business day after the last timestamp in the blacklist
            # then replace last timestamp with current timestamp
            # Else change name of key to cid_1 and add current timestamp to blacklist under cid_2
            last_timestamp = xcat_blacklist[cid][-1][-1]

    def return_blacklist(self, xcat: str = None):
        blacklist_output = {}
        xcat_blacklist = self.blacklist.get(xcat)
        for key, date_ranges in xcat_blacklist.items():
            if len(date_ranges) == 1:
                blacklist_output[key] = (date_ranges[0][0], date_ranges[0][-1])
            else:
                for i, date_range in enumerate(date_ranges, start=1):
                    blacklist_output[key + "_" + str(i)] = (date_range[0], date_range[-1])
        return blacklist_output

    def return_filled_df(self):
        return self.df
