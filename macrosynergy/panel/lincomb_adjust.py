import pandas as pd
from typing import Optional, List
from macrosynergy.management import reduce_df
from macrosynergy.management.types import QuantamentalDataFrame


def lincomb(
    df_zns: pd.DataFrame,
    min_score: Optional[float] = None,
    coeff_new: Optional[float] = 0.5,
) -> pd.DataFrame:
    """
    Linear combination of the parameters.
    """
    min_score = -3 if min_score is None else min_score
    assert coeff_new >= 0 and coeff_new <= 1, "`coeff_new` must be between 0 and 1"

    # new_weight_basis[i, t] = max(adj_zns[i, t] - min_score, 0)
    nwb = df_zns.apply(lambda x: max(x - min_score, 0))

    # new_weight[i, t] = new_weight_basis[i, t] / sum(new_weight_basis[t])
    nw = nwb.div(nwb.sum(axis="columns"), axis="index")

    # output_raw_weight[i, t] = (1 - coeff_new) * old_weight[i, t] + coeff_new * new_weight[i, t]
    orw = (1 - coeff_new) * df_zns + coeff_new * nw

    # output_weight[i, t] = output_raw_weight[i, t] / sum(output_raw_weight[i, t]))
    ow = orw.div(orw.sum(axis="columns"), axis="index")

    return ow


def lincomb_adjust(
    df: QuantamentalDataFrame,
    zns_xcat: str,
    cids: Optional[List[str]] = None,
    min_score: Optional[float] = None,
    coeff_new: Optional[float] = 0.5,
) -> QuantamentalDataFrame:
    """
    Adjust the weights of the zns scores based on the linear combination of the cross-sectional values.

    Parameters
    ----------
    df : QuantamentalDataFrame
        The input dataframe.

    zns_xcat : str
        The category of the zns scores to adjust. This category should be present in the `df`.

    min_score : float, optional
        The minimum score to consider. Default is None, where it is set to the minimum
        score discovered in the panel of `zns_xcat`.

    coeff_new : float, optional
        The coefficient to use for the new weights. Default is 0.5.

    Returns
    -------
    QuantamentalDataFrame
        The adjusted dataframe.
    """
    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("`df` must be a QuantamentalDataFrame")

    df = QuantamentalDataFrame(df)
    result_as_categorical: bool = df.InitializedAsCategorical

    if not isinstance(zns_xcat, str):
        raise TypeError("`zns_xcat` must be a string")

    if not isinstance(min_score, (int, float, type(None))):
        raise TypeError("`min_score` must be a number or None")

    if not isinstance(coeff_new, (int, float)):
        raise TypeError("`coeff_new` must be a number")
    if not 0 <= coeff_new <= 1:
        raise ValueError("`coeff_new` must be between 0 and 1")

    df, r_xcats, r_cids = reduce_df(df, xcats=[zns_xcat], cids=cids)

    if cids is not None:
        if set(r_cids).issubset(set(cids)):
            raise ValueError("The `cids` provided are not present in the dataframe")

    if set(r_xcats) != set([zns_xcat]):
        raise ValueError(f"The `zns_xcat` provided is not present in the dataframe")

    if min_score is None:
        min_score = df["value"].min()

    dfw = df.to_wide()

    dfw = lincomb(dfw, min_score=min_score, coeff_new=coeff_new)

    return QuantamentalDataFrame.from_wide(dfw, categorical=result_as_categorical)


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_test_df

    df = make_test_df(xcats=["weights", "adj_zns"], cids=["cid1", "cid2", "cid3"])
    dfb = make_test_df(xcats=["some_xcat", "other_xcat"], cids=["cid1", "cid2", "cid4"])
    df = pd.concat([df, dfb], axis=0)

    df_res = lincomb_adjust(df, zns_xcat="adj_zns", min_score=-3, coeff_new=0.5)
    print(df_res)
