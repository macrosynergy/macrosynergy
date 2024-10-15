import numpy as np
import pandas as pd

from macrosynergy.management import update_df


def impute_panel(
    df: pd.DataFrame, cids: list, xcats: list, threshold: float = 0.5
) -> pd.DataFrame:
    """
    Imputes missing values for each category in a long-format panel dataset by a
    cross-sectional mean, conditional on the number of available cross-sections at each
    concerned date exceeding a fraction `threshold` of the total number of cross-sections.

    :param <pd.DataFrame> df: the long-format panel dataset
    :param <list> cids: the list of cross sections to be considered in the imputation
    :param <list> xcats: the list of categories to be imputed
    :param <float> threshold: the fraction of available cross-sections at each date

    :return <pd.DataFrame>: the imputed long-format panel data, with the relevant xcats and cids

    .. note::

      This class is still **experimental**: the predictions
      and the API might change without any deprecation cycle.
    """
    # Checks
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input `df` must be a pandas DataFrame.")
    if not isinstance(xcats, list):
        raise TypeError("The input `xcats` must be a list.")
    for xcat in xcats:
        if not isinstance(xcat, str):
            raise TypeError("The elements of `xcats` must be strings.")
    if not isinstance(threshold, float):
        raise TypeError("The input `threshold` must be a float.")
    if not 0 <= threshold <= 1:
        raise ValueError("The input `threshold` must be between 0 and 1.")

    complete_df = df.copy()
    complete_df = complete_df.set_index(["cid", "real_date", "xcat"])
    full_idx = pd.MultiIndex.from_frame(
        pd.concat(
            [
                pd.MultiIndex.from_product(
                    [
                        complete_df.index.levels[0],
                        complete_df.index.levels[1],
                        pd.Series([xcat], name="xcat"),
                    ]
                ).to_frame(index=False)
                for xcat in complete_df.index.levels[2].unique()
            ],
            axis=0,
            ignore_index=True,
        ),
    )

    # reindexing to align all the CIDs on the same timeseries of dates
    complete_df = complete_df.reindex(full_idx).reset_index(drop=False)

    # subsetting to keep only the relevant XCATs across CIDs and dates
    incomplete_mask = (complete_df["xcat"].isin(xcats)) & (
        complete_df["cid"].isin(cids)
    )
    incomplete_df = complete_df.loc[incomplete_mask, :]
    # computing the data availability stats
    incomplete_df["mean_val"] = incomplete_df.groupby(["xcat", "real_date"])[
        "value"
    ].transform("mean")
    incomplete_df["tot"] = incomplete_df.groupby(["xcat", "real_date"])[
        "value"
    ].transform("size")
    incomplete_df["avail"] = incomplete_df.groupby(["xcat", "real_date"])[
        "value"
    ].transform("count")

    # Filling CID-specific values only for the appropriate cids and conditional on representative sample
    mask = (incomplete_df["avail"].div(incomplete_df["tot"]) > threshold) & (
        incomplete_df["value"].isna()
    )
    incomplete_df.loc[mask, "value"] = incomplete_df.loc[mask, "value"].fillna(
        incomplete_df.loc[mask, "mean_val"]
    )

    return incomplete_df.loc[:, complete_df.columns]
