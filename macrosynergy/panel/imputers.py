import numpy as np
import pandas as pd

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.simulate import make_qdf


def impute_panel(
    df: pd.DataFrame,
    cids: list,
    xcats: list,
    threshold: float = 0.5,
    start_date: str = None,
    impute_empty_tickers: bool = False,
) -> pd.DataFrame:
    """
    Imputes missing values for each category in a long-format panel dataset by a cross-
    sectional mean, conditional on the number of available cross-sections at each
    concerned date exceeding a fraction `threshold` of the total number of cross-
    sections.

    Parameters
    ----------
    df : ~pandas.DataFrame
        the long-format panel dataset
    cids : list
        the list of cross sections to be considered in the imputation
    xcats : list
        the list of categories to be imputed
    threshold : float
        the fraction of available cross-sections at each date
    start_date : str
        the starting date for the imputation
    impute_empty_tickers : bool
        boolean flag for whether to impute missing values for empty tickers

    Returns
    -------
    ~pandas.DataFrame
        the imputed long-format panel data with columns


    .. note::
        This class is still **experimental**: the predictions and the API might change
        without any deprecation cycle.
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
    if not isinstance(start_date, str) and start_date is not None:
        raise TypeError("The input `start_date` must be a string.")
    if not isinstance(impute_empty_tickers, bool):
        raise TypeError("The input `impute_empty_tickers` must be a boolean.")

    if start_date is not None:
        df = df[df["real_date"] >= start_date]

    complete_df = QuantamentalDataFrame(df)
    _as_categorical = complete_df.InitializedAsCategorical
    complete_df = complete_df.set_index(["cid", "real_date", "xcat"])
    complete_df = complete_df
    if impute_empty_tickers:
        cids_series = pd.Series(cids, name="cid")
    else:
        cids_series = complete_df.index.levels[0]
    full_idx = pd.MultiIndex.from_frame(
        pd.concat(
            [
                pd.MultiIndex.from_product(
                    [
                        cids_series,
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

    return QuantamentalDataFrame(
        incomplete_df[complete_df.columns].reset_index(drop=True),
        categorical=_as_categorical,
    )

if __name__ == "__main__":
    cids = ["AUD", "CAD", "GBP", "NZD", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2000-01-01", "2020-12-31", 0.1, 1]
    df_cids.loc["CAD"] = ["2001-01-01", "2020-11-30", 0, 1]
    # df_cids.loc["BRL"] = ["2001-01-01", "2020-11-30", -0.1, 2]
    df_cids.loc["GBP"] = ["2002-01-01", "2024-12-30", 0, 2]
    df_cids.loc["NZD"] = ["2002-01-01", "2024-12-30", -0.1, 2]
    df_cids.loc["USD"] = ["2003-01-01", "2024-12-31", -0.1, 2]

    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]
    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2000-01-01", "2024-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-10-30", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2001-01-01", "2020-10-30", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    dfx = impute_panel(dfd, cids + ["BRL"], ["XR"], threshold=0.0, start_date="2000-01-01", impute_empty_tickers=True)

    print(dfx)
