"""
Functionality to create contract-specific target positions from signals.
"""

import numpy as np
import pandas as pd
from typing import List, Union
from macrosynergy.management.utils import reduce_df
from macrosynergy.management.simulate import make_qdf
from macrosynergy.panel.historic_vol import historic_vol
from macrosynergy.panel.make_zn_scores import make_zn_scores
from macrosynergy.panel.basket import Basket
from macrosynergy.compat import PD_FUTURE_STACK
import random


def weight_dataframes(df: pd.DataFrame, basket_names: Union[str, List[str]] = None):
    """
    Helper function used to break up the original dataframe and create the wide weight
    dataframes.

    :param <pd.Dataframe> df: standardized DataFrame.
    :param <str or List[str]> basket_names: string or list of basket names.

    :return <List[pd.DataFrames], dict>:
    """
    if isinstance(basket_names, str):
        basket_names = [basket_names]

    xcats = df["xcat"].to_numpy()
    wgt_indices = lambda index: index.split("_")[-1] == "WGT"
    boolean = list(map(wgt_indices, xcats))

    r_df = df[boolean].copy()
    b_column = lambda index: "_".join(index.split("_")[1:-1])

    r_df.loc[:, "basket_name"] = np.array(list(map(b_column, r_df["xcat"])))

    df_c_wgts = []
    b_dict = {}
    contr_func = lambda index: "_".join(index.split("_")[0:2])

    r_df_copy = r_df.copy()

    for b in basket_names:
        b_df = r_df[r_df["basket_name"] == b]
        b_df_copy = r_df_copy[r_df_copy["basket_name"] == b]
        b_df_copy = b_df_copy.drop(["basket_name"], axis=1)
        column = b_df["cid"] + "_" + b_df["xcat"]
        b_df_copy["xcat"] = np.array(list(map(contr_func, column)))
        b_wgt = b_df_copy.pivot(index="real_date", columns="xcat", values="value")
        b_wgt = b_wgt.reindex(sorted(b_wgt.columns), axis=1)
        df_c_wgts.append(b_wgt)
        contracts = set(b_df_copy["xcat"])
        b_dict[b] = sorted(list(contracts))

    return df_c_wgts, b_dict


def modify_signals(
    df: pd.DataFrame,
    cids: List[str],
    xcat_sig: str,
    start: str = None,
    end: str = None,
    scale: str = "prop",
    min_obs: int = 252,
    thresh: float = None,
):
    """
    Calculate modified cross-section signals based on zn-scoring (proportionate method)
    or conversion to signs (digital method).

    :param <pd.Dataframe> df: standardized DataFrame containing the following columns:
        'cid', 'xcat', 'real_date' and 'value'.
    :param <List[str]> cids: cross sections of markets or currency areas in which
        positions should be taken.
    :param <str> xcat_sig: category that serves as signal across markets.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        for which the signal category is available is used.
    :param <str> end: latest date in ISO format. Default is None and latest date
        for which the signal category is available is used.
    :param <str> scale: method to translate signals into target positions:
        [1] Default is 'prop', means proportionate. In this case zn-scoring is applied
            to the signal based on the panel, with the neutral level set at zero.
            A 1 SD value translates into a USD1 position in the contract.
        [2] Method 'dig' means 'digital' and sets the individual position to either USD1
            long or short, depending on the sign of the signal.
            Note that a signal of zero translates into a position of zero.
    :param <int> min_obs: the minimum number of observations required to calculate
        zn_scores. Default is 252.
        Note: For the initial period of the signal time series in-sample
        zn-scoring is used.
    :param <float> thresh: threshold value beyond which zn-scores for propotionate
        position taking are winsorized. The threshold is the maximum absolute
        score value in standard deviations. The minimum is 1 standard deviation.

    :return <pd.Dataframe>: standardized dataframe, of modified signaks, using the
        columns 'cid', 'xcat', 'real_date' and 'value'.

    """

    options = ["prop", "dig"]
    assert scale in options, f"The scale parameter must be either {options}"

    if scale == "prop":
        df_ms = make_zn_scores(
            df,
            xcat=xcat_sig,
            sequential=True,
            cids=cids,
            start=start,
            end=end,
            neutral="zero",
            pan_weight=1,
            min_obs=min_obs,
            iis=True,
            thresh=thresh,
        )
    else:
        df_ms = reduce_df(
            df=df, xcats=[xcat_sig], cids=cids, start=start, end=end, blacklist=None
        )
        df_ms["value"] = np.sign(df_ms["value"])

    return df_ms


def cs_unit_returns(
    df: pd.DataFrame,
    contract_returns: List[str],
    sigrels: List[str],
    ret: str = "XR_NSA",
):
    """
    Calculate returns of composite unit positions (that jointly depend on one signal).

    :param <pd.Dataframe> df: standardized DataFrame containing the following columns:
        'cid', 'xcat', 'real_date' and 'value'.
    :param <List[str]> contract_returns: list of the contract return types.
    :param <List[float]> sigrels: respective signal for each contract type.
    :param <str> ret: postfix denoting the returns in % applied to the contract types.

    :return <pd.Dataframe>: standardized dataframe with the summed portfolio returns
        which are used to calculate the evolving volatility, using the columns 'cid',
        'xcat', 'real_date' and 'value'.

    """

    error_message = "Each individual contract requires an associated signal."
    assert len(contract_returns) == len(sigrels), error_message
    df_c_rets = None

    for i, c_ret in enumerate(contract_returns):
        # Filter the DataFrame 'df' to just c_ret xcats
        df_c_ret = df[df["xcat"] == c_ret]
        # Reshape the DataFrame by setting 'real_date' as index, 'cid' as columns, and
        # 'value' as values.
        df_c_ret = df_c_ret.pivot(index="real_date", columns="cid", values="value")

        # Sort the dataframe and multiply by the respective signal.
        df_c_ret = df_c_ret.sort_index(axis=1)
        df_c_ret *= sigrels[i]

        df_c_rets = df_c_ret if i == 0 else df_c_rets + df_c_ret

    # Any dates not shared by all categories will be removed.
    df_c_rets.dropna(how="all", inplace=True)

    df_rets = df_c_rets.stack().to_frame("value").reset_index()
    df_rets["xcat"] = ret

    cols = ["cid", "xcat", "real_date", "value"]

    return df_rets[cols].sort_values(by=cols[:3])


def basket_handler(
    df_mods_w: pd.DataFrame, df_c_wgts: pd.DataFrame, contracts: List[str]
):
    """
    Function designed to compute the target positions for the constituents of a basket.
    The function will return the corresponding basket dataframe of positions.

    :param <pd.DataFrame> df_mods_w: target position dataframe. Will be multiplied by the
        weight dataframe to establish the positions for the basket of constituents.
    :param <pd.DataFrame> df_c_wgts: weight dataframe used to adjust the positions of
        the basket of contracts.
    :param <dict> contracts: the constituents that make up each basket.

    :return <pd.Dataframe>: basket positions weight-adjusted.
    """

    error_1 = "df_c_wgts expects to receive a pd.DataFrame."
    assert isinstance(df_c_wgts, pd.DataFrame), error_1
    error_2 = (
        "df_c_wgts expects a pivoted pd.DataFrame - each column corresponds to the"
        " contract's weight."
    )
    assert df_c_wgts.index.name == "real_date", error_2
    error_3 = (
        f"df_c_wgts column names must correspond to the received contract: "
        f"{contracts}."
    )
    assert all(df_c_wgts.columns == contracts), error_3

    split = lambda b: b.split("_")[0]

    cross_sections = list(map(split, contracts))

    # Sort the columns to align via cross-sections to conduct the multiplication. The
    # weight dataframe is formed using the respective contracts, so additional checks are
    # not required.
    dfw_wgs = df_c_wgts.reindex(sorted(df_c_wgts.columns), axis=1)

    # Reduce to the cross-sections held in the respective basket.
    df_mods_w = df_mods_w[cross_sections]
    df_mods_w = df_mods_w.reindex(sorted(df_mods_w.columns), axis=1)

    # Adjust the target positions to reflect the weighting method. Align the pandas names
    # to allow for pd.DataFrame.multiply().
    dfw_wgs.columns = df_mods_w.columns
    df_mods_w = df_mods_w.multiply(dfw_wgs)

    return df_mods_w


def date_alignment(panel_df: pd.DataFrame, basket_df: pd.DataFrame):
    """
    Method used to align the panel position dataframe and the basket dataframe of
    weight-adjusted positions to the same timeframe.

    :param <pd.DataFrame> panel_df:
    :param <pd.DataFrame> basket_df:

    :return <Tuple[pd.DataFrame, pd.DataFrame]>: returns the two received dataframes
        defined over the same period.
    """

    p_dates = panel_df["real_date"].to_numpy()
    b_dates = basket_df["real_date"].to_numpy()
    if p_dates[0] > b_dates[0]:
        index = np.searchsorted(b_dates, p_dates[0])
        basket_df = basket_df.iloc[index:, :]
    elif p_dates[0] < b_dates[0]:
        index = np.searchsorted(p_dates, b_dates[0])
        panel_df = panel_df.iloc[index:, :]
    else:
        pass

    if p_dates[-1] > b_dates[-1]:
        index = np.searchsorted(p_dates, b_dates[-1], side="right")
        panel_df = panel_df.iloc[:index, :]
    elif p_dates[-1] < b_dates[-1]:
        index = np.searchsorted(b_dates, p_dates[-1], side="right")
        basket_df = basket_df.iloc[:index, :]
    else:
        pass

    return panel_df, basket_df


def consolidation_help(panel_df: pd.DataFrame, basket_df: pd.DataFrame):
    """
    The function receives a panel dataframe and a basket of cross-sections of the same
    contract type. Therefore, aim to consolidate the targeted positions across the shared
    contracts.

    :param <pd.DataFrame> panel_df:
    :param <pd.DataFrame> basket_df:

    :return <Tuple[pd.DataFrame, pd.DataFrame]>: returns the consolidated and reduced
        dataframes.
    """

    basket_cids = basket_df["cid"].unique()
    panel_cids = panel_df["cid"].unique()

    panel_copy = []
    for cid in panel_cids:
        indices = panel_df["cid"] == cid
        temp_df = panel_df[indices]

        if cid in basket_cids:
            basket_indices = basket_df["cid"] == cid
            basket_rows = basket_df[basket_indices]
            temp_df, basket_rows = date_alignment(temp_df, basket_rows)
            b_values = basket_rows["value"].to_numpy()

            panel_values = temp_df["value"].to_numpy()
            consolidation = panel_values + b_values
            temp_df["value"] = consolidation
            panel_copy.append(temp_df)

            basket_indices = ~basket_indices
            basket_df = basket_df[basket_indices]
        else:
            panel_copy.append(temp_df)

    return pd.concat(panel_copy)


def consolidate_positions(data_frames: List[pd.DataFrame], ctypes: List[str]):
    """
    Method used to consolidate positions if baskets are used. The constituents of a
    basket will be a subset of one of the panels.

    :param <List[pd.DataFrame]> data_frames: list of the target position dataframes.
    :param <List[str]> ctypes:

    :return <List[pd.DataFrame]>: list of dataframes having consolidated positions.
    """

    no_ctypes = len(ctypes)
    dict_ = dict(zip(ctypes[:no_ctypes], data_frames[:no_ctypes]))
    df_baskets = data_frames[no_ctypes:]
    # Iterating exclusively through the basket dataframes.
    for df in df_baskets:
        category = df["xcat"].str.split("_", expand=True)[1]
        c_type = category.iloc[0]

        panel_df = dict_[c_type]
        dict_[c_type] = consolidation_help(panel_df, basket_df=df)

    return list(dict_.values())


def target_positions(
    df: pd.DataFrame,
    cids: List[str],
    xcat_sig: str,
    ctypes: Union[List[str], str],
    sigrels: List[float],
    basket_names: Union[str, List[str]] = [],
    ret: str = "XR_NSA",
    start: str = None,
    end: str = None,
    scale: str = "prop",
    min_obs: int = 252,
    thresh: float = None,
    cs_vtarg: float = None,
    lback_periods: int = 21,
    lback_meth: str = "ma",
    half_life: int = 11,
    posname: str = "POS",
):
    """
    Converts signals into contract-specific target positions.

    :param <pd.Dataframe> df: standardized DataFrame containing at least the following
        columns: 'cid', 'xcat', 'real_date' and 'value'.
    :param <List[str]> cids: cross-sections of markets or currency areas in which
        positions should be taken.
    :param <str> xcat_sig: category that serves as signal across markets.
    :param <List[str]> ctypes: contract types that are traded across markets. They should
        correspond to return categories in the dataframe if the `ret` argument is
        appended. Examples are 'FX' or 'EQ'.
    :param <str or List[str]> basket_names: single string or list of the names of several
        baskets. The weight dataframes will be appended to the main dataframe. Therefore,
        use the basket name to isolate the corresponding weights. The default value for
        the parameter is an empty list, which mean that no baskets are traded.
    :param <List[float]> sigrels: values that translate the single signal into contract
        type and basket signals in the order defined by keys.
    :param <str> ret: postfix denoting the returns in % associated with contract types.
        For JPMaQS derivatives return data this is typically "XR_NSA" (default).
        Returns are required for volatility targeting.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        for which the signal category is available is used.
    :param <str> end: latest date in ISO format. Default is None and latest date
        for which the signal category is available is used.
    :param <str> scale: method that translates signals into unit target positions:
        [1] Default is 'prop' for proportionate. In this case zn-scoring is applied
            to the signal based on the panel, with the neutral level set at zero.
            A 1 SD value translates into a USD1 position in the contract.
            This translation may apply winsorization through the `thresh` argument
        [2] Method 'dig' means 'digital' and sets the individual position to either USD1
            long or short, depending on the sign of the signal.
            Note that a signal of zero translates into a position of zero.
        Note that unit target positions may subsequently be calibrated to meet cross-
        section volatility targets using the `cs_targ` argument.
    :param <int> min_obs: the minimum number of observations required to calculate
        zn_scores. Default is 252.
        Note: For the initial minimum period of the signal time series in-sample
        zn-scoring is used.
    :param <float> thresh: threshold value beyond which zn-scores for proportionate
        position taking are winsorized. The threshold is the maximum absolute
        score value in standard deviations. The minimum is 1 standard deviation.
    :param <float> cs_vtarg: Value for volatility targeting at the cross-section level.
        The purpose of this operation is to relate signal to risk rather than notional.
        Default is None and means no volatility targeting.
        If a value is chosen then for each cross-section a unit position is defined as a
        position for which the annual return standard deviation is equal to that value.
        For example, a target of 10 and a cross-section signal of 0.5 standard deviations
        would translate into a target position that carries a recent historical
        annualized standard deviation of 5 dollars (or other currency units).
    :param <int>  lback_periods: Number of lookback periods over which volatility is
        calculated. Default is 21. Typically this refers to days.
    :param <str> lback_meth: Lookback method to calculate the volatility.
        Default is "ma", which means simple moving average.
        Alternative is "ema", which means exponential moving average.
    :param <int> half_life: Refers to the half-time for "xma". Default is 11.
    :param <str> posname: postfix added to contract to denote position name.

    :return <pd.Dataframe>: standardized dataframe with daily target positions
        in USD, using the columns 'cid', 'xcat', 'real_date' and 'value'.

    Note: A target position differs from a signal insofar as it is a dollar amount and
        determines to what extent the size of signal (as opposed to direction) matters.
        Further, if the modified signal has a NaN value, the target position will be
        converted to zero: a position will not be taken given the signal was not
        available for that respective date.
        A target position also differs from an actual position in two ways. First,
        the actual position can only be aligned with the target with some lag. Second,
        the actual position will be affected by other considerations, such as
        risk management and assets under management.
    """

    # A. Initial checks

    if basket_names:
        df_c_wgts, baskets = weight_dataframes(df=df, basket_names=basket_names)
        df_c_wgts = iter(df_c_wgts)
        no_panels = len(sigrels) - len(basket_names)
        panel_sigrels = sigrels[:no_panels]  # sigrels only for regular panels.
    else:
        panel_sigrels = sigrels

    if isinstance(ctypes, str):
        ctypes = [ctypes]
    ctypes_baskets = ctypes + basket_names

    cols = ["cid", "xcat", "real_date", "value"]
    assert set(cols) <= set(df.columns), f"df columns must contain {cols}."

    categories = set(df["xcat"].unique())
    error_1 = "Signal category missing from the standardised dataframe."
    assert xcat_sig in categories, error_1
    error_2 = "Volatility Target must be numeric value."
    if cs_vtarg is not None:
        assert isinstance(cs_vtarg, (float, int)), error_2

    error_3 = (
        "The number of signal relations must be equal to the number of contracts "
        "and, if defined, the number of baskets defined in 'ctypes'."
    )
    clause = len(ctypes_baskets)
    assert len(sigrels) == clause, error_3
    assert isinstance(min_obs, int), "Minimum observation parameter must be an integer."

    # B. Reduce frame to necessary data.

    df = df.loc[:, cols]
    contract_returns = [c + ret for c in ctypes]
    xcats = contract_returns + [xcat_sig]

    dfx = reduce_df(df=df, xcats=xcats, cids=cids, start=start, end=end, blacklist=None)

    # C. Calculate and reformat modified cross-sectional signals.

    df_mods = modify_signals(
        df=dfx,
        cids=cids,
        xcat_sig=xcat_sig,
        start=start,
        end=end,
        scale=scale,
        min_obs=min_obs,
        thresh=thresh,
    )  # (USD 1 per SD or sign).

    df_mods_w = df_mods.pivot(index="real_date", columns="cid", values="value")

    # D. Volatility target ratios (if required).

    use_vtr = False
    if isinstance(cs_vtarg, (int, float)):
        # D.1. Composite signal-related positions as basis for volatility targeting.

        df_csurs = cs_unit_returns(
            dfx, contract_returns=contract_returns, sigrels=panel_sigrels
        )  # Gives cross-section returns.

        # D.2. Calculate volatility adjustment ratios.

        df_vol = historic_vol(
            df_csurs,
            xcat=ret,
            cids=cids,
            lback_periods=lback_periods,
            lback_meth=lback_meth,
            half_life=half_life,
            start=start,
            end=end,
            remove_zeros=True,
            postfix="",
        )  # Gives unit position vols.

        dfw_vol = df_vol.pivot(index="real_date", columns="cid", values="value")
        dfw_vol = dfw_vol.sort_index(axis=1)
        dfw_vtr = 100 * cs_vtarg / dfw_vol  # vol-target ratio to be applied.
        use_vtr = True

    # E. Actual position calculation.

    df_pos_cons = []
    ctypes_sigrels = dict(zip(ctypes_baskets, sigrels))

    for k, v in ctypes_sigrels.items():
        df_mods_copy = df_mods_w.copy()

        if use_vtr:
            dfw_pos_vt = df_mods_copy.multiply(dfw_vtr)
            dfw_pos_vt.dropna(how="all", inplace=True)
            df_mods_copy = dfw_pos_vt

        if k in basket_names:
            contracts = baskets[k]
            df_c_weights = next(df_c_wgts)
            df_mods_copy = basket_handler(
                df_mods_w=df_mods_copy, df_c_wgts=df_c_weights, contracts=contracts
            )

        # Allows for the signal being applied to the basket constituents on the original
        # dataframe.
        df_mods_copy *= v  # modified signal x sigrel = post-VT position.

        df_posi = df_mods_copy.stack(**PD_FUTURE_STACK).to_frame("value").reset_index()
        df_posi = df_posi.fillna(0)
        df_posi["xcat"] = k
        df_posi = df_posi.sort_values(["cid", "xcat", "real_date"])[cols]
        df_pos_cons.append(df_posi)

    # Consolidate the positions across the formed panels and baskets (baskets will be a
    # subset of the panels).
    if basket_names:
        df_pos_cons = consolidate_positions(df_pos_cons, ctypes)
    df_tpos = pd.concat(df_pos_cons, axis=0, ignore_index=True)

    df_tpos["xcat"] += "_" + posname
    df_tpos = df_tpos[cols]

    df_tpos = reduce_df(df=df_tpos, xcats=None, cids=None, start=start, end=end)
    df_tpos = df_tpos.sort_values(["cid", "xcat", "real_date"])[cols]

    return df_tpos.reset_index(drop=True)


if __name__ == "__main__":
    # A. Example dataframe

    cids = ["AUD", "GBP", "NZD", "USD"]
    xcats = ["FXXR_NSA", "EQXR_NSA", "SIG_NSA"]

    ccols = ["earliest", "latest", "mean_add", "sd_mult"]
    df_cids = pd.DataFrame(index=cids, columns=ccols)
    df_cids.loc["AUD"] = ["2010-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2010-01-01", "2020-12-31", 0, 2]
    df_cids.loc["NZD"] = ["2010-01-01", "2020-12-31", 0, 3]
    df_cids.loc["USD"] = ["2010-01-01", "2020-12-31", 0, 4]

    xcols = ccols + ["ar_coef", "back_coef"]
    df_xcats = pd.DataFrame(index=xcats, columns=xcols)
    df_xcats.loc["FXXR_NSA"] = ["2010-01-01", "2020-12-31", 0, 1, 0, 0.2]
    df_xcats.loc["EQXR_NSA"] = ["2010-01-01", "2020-12-31", 0.5, 2, 0, 0.2]
    df_xcats.loc["SIG_NSA"] = ["2010-01-01", "2020-12-31", 0, 10, 0.4, 0.2]

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd_copy = dfd.copy()
    black = {"AUD": ["2000-01-01", "2003-12-31"], "GBP": ["2018-01-01", "2100-01-01"]}

    # B. Target positions without basket

    df1 = target_positions(
        df=dfd,
        cids=cids,
        xcat_sig="SIG_NSA",
        ctypes=["FX", "EQ"],
        sigrels=[1, 0.5],
        ret="XR_NSA",
        start="2012-01-01",
        end="2020-10-30",
        scale="prop",
        min_obs=252,
        cs_vtarg=5,
        posname="POS",
    )

    df2 = target_positions(
        df=dfd,
        cids=cids,
        xcat_sig="FXXR_NSA",
        ctypes=["FX", "EQ"],
        sigrels=[1, -1],
        ret="XR_NSA",
        start="2012-01-01",
        end="2020-10-30",
        scale="dig",
        cs_vtarg=0.1,
        posname="POS",
    )

    df3 = target_positions(
        df=dfd,
        cids=cids,
        xcat_sig="FXXR_NSA",
        ctypes=["FX", "EQ"],
        sigrels=[1, -1],
        ret="XR_NSA",
        start="2010-01-01",
        end="2020-12-31",
        scale="prop",
        cs_vtarg=None,
        posname="POS",
    )

    # C. Target position with one basket

    apc_contracts = ["AUD_FX", "NZD_FX"]
    basket_1 = Basket(
        df=dfd, contracts=apc_contracts, ret="XR_NSA", cry=None, blacklist=black
    )
    basket_1.make_basket(weight_meth="equal", max_weight=0.55, basket_name="APC_FX")
    df_weight = basket_1.return_weights("APC_FX")

    df_weight = df_weight[["cid", "xcat", "real_date", "value"]]
    dfd = dfd[["cid", "xcat", "real_date", "value"]]
    dfd_concat = pd.concat([dfd_copy, df_weight])

    df4 = target_positions(
        df=dfd_concat,
        cids=cids,
        xcat_sig="SIG_NSA",
        ctypes=["FX", "EQ"],
        basket_names=["APC_FX"],
        sigrels=[1, -1, -0.5],
        ret="XR_NSA",
        start="2010-01-01",
        end="2020-12-31",
        scale="prop",
        cs_vtarg=10,
        posname="POS",
    )
