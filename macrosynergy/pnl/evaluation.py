from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from macrosynergy.management import reduce_df
from macrosynergy.management.utils import _map_to_business_day_frequency


def evaluate_pnl(
    df: pd.DataFrame,
    pnl_cats: List[str],
    pnl_cids: List[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    label_dict: Optional[Dict[str, str]] = None,
    blacklist: Optional[Dict[str, Tuple]] = None,
    benchmarks: Optional[List[str]] = None,
):
    """
    """
    error_cids = "List of cross-sections expected."
    error_xcats = "List of PnL categories expected."
    if not isinstance(pnl_cids, (list, type(None))):
        raise TypeError(error_cids)
    if not isinstance(pnl_cats, list):
        raise TypeError(error_xcats)
    if not all(isinstance(elem, str) for elem in pnl_cats):
        raise TypeError(error_xcats)
    if pnl_cids is not None and not all(isinstance(elem, str) for elem in pnl_cids):
        raise TypeError(error_cids)

    unique_xcats = set(df["xcat"].unique())
    missing_pnl_cats = set(pnl_cats).difference(unique_xcats)
    if missing_pnl_cats:
        raise ValueError(f"Input data has no pnl category: {missing_pnl_cats}")

    # if not (len(pnl_cats) == 1) | (len(pnl_cids) == 1):
    #     len_err = (
    #         "One can only have multiple PnL categories or multiple cross-sections, "
    #         "not both."
    #     )
    #     raise ValueError(len_err)

    dfx = reduce_df(
        df=df,
        xcats=pnl_cats,
        cids=pnl_cids,
        start=start,
        end=end,
        blacklist=blacklist,
        out_all=False,
    )

    groups = "xcat" if pnl_cids is None or len(pnl_cids) == 1 else "cid"
    stats = [
        "Return %",
        "St. Dev. %",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max 21-Day Draw %",
        "Max 6-Month Draw %",
        "Peak to Trough Draw %",
        "Top 5% Monthly PnL Share",
        "Traded Months",
    ]

    if benchmarks:
        missing_benchmarks = set(benchmarks).difference(unique_xcats)

        if missing_benchmarks:
            bm_err = f"Input data is missing benchmark tickers: {missing_benchmarks}"
            raise ValueError(bm_err)

        for bm in benchmarks:
            stats.insert(len(stats) - 1, f"{bm} correl")

    dfw = dfx.pivot(index="real_date", columns=groups, values="value")
    df = pd.DataFrame(columns=dfw.columns, index=stats)

    # Step 1: annualized mean and std
    df.iloc[0, :] = dfw.mean(axis=0) * 261
    df.iloc[1, :] = dfw.std(axis=0) * np.sqrt(261)

    # Step 2: sharpe + sortino ratios
    df.iloc[2, :] = df.iloc[0, :] / df.iloc[1, :]
    dsd = dfw.apply(lambda x: np.sqrt(np.sum(x[x < 0] ** 2) / len(x))) * np.sqrt(261)
    df.iloc[3, :] = df.iloc[0, :] / dsd

    # Step 3: draws
    df.iloc[4, :] = dfw.rolling(21).sum().min()
    df.iloc[5, :] = dfw.rolling(6 * 21).sum().min()

    cum_pnl = dfw.cumsum()
    high_watermark = cum_pnl.cummax()
    drawdown = high_watermark - cum_pnl

    df.iloc[6, :] = -drawdown.max()

    # Step 4: monthly PnL share
    mfreq = _map_to_business_day_frequency("M")
    monthly_pnl = dfw.resample(mfreq).sum()
    total_pnl = monthly_pnl.sum(axis=0)
    top_5_percent_cutoff = int(np.ceil(len(monthly_pnl) * 0.05))
    top_months = pd.DataFrame(columns=monthly_pnl.columns)
    for column in monthly_pnl.columns:
        top_months[column] = (
            monthly_pnl[column].nlargest(top_5_percent_cutoff).reset_index(drop=True)
        )

    df.iloc[7, :] = top_months.sum() / total_pnl

    # Step 5: benchmark correlations
    if benchmarks:
        df_bm = _compute_benchmark_df(df, benchmarks)
        for i, bm in enumerate(benchmarks):
            index = dfw.index.intersection(df_bm.index)
            correlation = dfw.loc[index].corrwith(
                other=df_bm.loc[index].loc[:, bm],
                axis=0,
                method="pearson",
                drop=True,
            )
            df.iloc[8 + i, :] = correlation

    # Step 6: number of traded months
    traded_months_idx = 8 if benchmarks is None else 8 + len(benchmarks)
    df.iloc[traded_months_idx, :] = (
        dfw.notna().resample(mfreq).sum().ne(0).sum()
    )

    # Step 7: optionally apply label dict
    if label_dict is not None:
        if not isinstance(label_dict, dict):
            raise TypeError("label_dict must be a dictionary.")
        if not all([isinstance(k, str) for k in label_dict.keys()]):
            raise TypeError("Keys in label_dict must be strings.")
        if not all([isinstance(v, str) for v in label_dict.values()]):
            raise TypeError("Values in label_dict must be strings.")
        if len(label_dict) != len(df.columns):
            raise ValueError(
                "label_dict must have the same number of keys as columns in the "
                "DataFrame."
            )
        df.rename(columns=label_dict, inplace=True)
        df = df[label_dict.values()]

    return df


def _compute_benchmark_df(
    df: pd.DataFrame, benchmarks: List[str]
) -> pd.DataFrame:

    bm_dfs = []
    for bm in benchmarks:
        # Accounts for appending "_NEG" to the ticker.
        bm_s = bm.split("_", maxsplit=1)
        cid = bm_s[0]
        xcat = bm_s[1]

        dfa = df[(df["cid"] == cid) & (df["xcat"] == xcat)]

        if dfa.shape[0] == 0:
            print(f"{bm} has no observations in the DataFrame.")
        else:
            df_single_bm = dfa.pivot(index="real_date", columns="xcat", values="value")
            df_single_bm.columns = [bm]

            bm_dfs.append(df_single_bm)

    return pd.concat(bm_dfs, axis=1)
