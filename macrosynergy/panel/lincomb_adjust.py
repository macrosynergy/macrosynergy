import pandas as pd
import numpy as np
from numbers import Number
import warnings
from typing import Optional, List

from macrosynergy.management import reduce_df
from macrosynergy.management.types import QuantamentalDataFrame

from macrosynergy.panel.adjust_weights import (
    split_weights_adj_zns,
    check_missing_cids_xcats,
    normalize_weights,
)


def _lincomb_backend(
    dfw_adj_zns: pd.DataFrame,
    dfw_weights: pd.DataFrame,
    min_score: Optional[float] = None,
    coeff_new: Optional[float] = 0.5,
) -> pd.DataFrame:
    """
    Linear combination of the parameters.
    """
    assert min_score is not None, "`min_score` must be provided"
    assert coeff_new >= 0 and coeff_new <= 1, "`coeff_new` must be between 0 and 1"

    # new_weight_basis[i, t] = max(adj_zns[i, t] - min_score, 0)
    nwb = dfw_adj_zns.apply(lambda s: s.apply(lambda x: max(x - min_score, 0)))

    # new_weight[i, t] = new_weight_basis[i, t] / sum(new_weight_basis[t])
    nw = nwb.div(nwb.sum(axis="columns"), axis="index")

    # output_raw_weight[i, t] = (1 - coeff_new) * old_weight[i, t] + coeff_new * new_weight[i, t]
    orw = (1 - coeff_new) * dfw_weights + coeff_new * nw

    # output_weight[i, t] = output_raw_weight[i, t] / sum(output_raw_weight[i, t]))
    ow = orw.div(orw.sum(axis="columns"), axis="index")

    assert np.allclose(ow[~ow.isna().any(axis="columns")].sum(axis=1), 1)
    return ow


def linear_combination_adjustment(
    df: QuantamentalDataFrame,
    adj_zns_xcat: str,
    weights_xcat: str,
    cids: Optional[List[str]] = None,
    min_score: Optional[float] = None,
    coeff_new: Optional[float] = 0.5,
    normalize: bool = True,
    normalize_to_pct: bool = False,
    adj_name: str = "lincomb",
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

    normalize : bool, optional
        Whether to normalize the weights. Default is True.

    normalize_to_pct : bool, optional
        Whether to normalize the weights to percentages. Default is False.

    adj_name : str, optional
        The name of the new category. Default is "lincomb".

    Returns
    -------
    QuantamentalDataFrame
        The adjusted dataframe.
    """
    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("`df` must be a QuantamentalDataFrame")

    df = QuantamentalDataFrame(df)
    result_as_categorical: bool = df.InitializedAsCategorical

    if not isinstance(adj_zns_xcat, str):
        raise TypeError("`zns_xcat` must be a string")

    if not isinstance(min_score, (Number, type(None))):
        raise TypeError("`min_score` must be a number or None")

    if not isinstance(coeff_new, Number):
        raise TypeError("`coeff_new` must be a number")
    if not 0 <= coeff_new <= 1:
        raise ValueError("`coeff_new` must be between 0 and 1")

    df, r_xcats, r_cids = reduce_df(
        df, xcats=[adj_zns_xcat, weights_xcat], cids=cids, out_all=True
    )

    if cids is not None:
        if set(r_cids).issubset(set(cids)):
            raise ValueError("The `cids` provided are not present in the dataframe")

    if set(r_xcats) != set([adj_zns_xcat, weights_xcat]):
        raise ValueError(f"The `zns_xcat` provided is not present in the dataframe")

    dfw_weights, dfw_adj_zns = split_weights_adj_zns(
        df=df, weights=weights_xcat, adj_zns=adj_zns_xcat
    )

    if min_score is None:
        warnings.warn(
            "`min_score` is not provided. Setting it to the minimum score in the panel."
        )
        min_score = dfw_adj_zns.min().min()

    dfw = _lincomb_backend(
        dfw_adj_zns=dfw_adj_zns,
        dfw_weights=dfw_weights,
        min_score=min_score,
        coeff_new=coeff_new,
    )

    if normalize:
        dfw = normalize_weights(dfw, normalize_to_pct=normalize_to_pct)

    dfw.columns = dfw.columns + "_" + adj_name

    return QuantamentalDataFrame.from_wide(
        dfw, categorical=result_as_categorical
    ).to_original_dtypes()


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_test_df

    df = make_test_df(xcats=["weights", "adj_zns"], cids=["cid1", "cid2", "cid3"])
    dfb = make_test_df(xcats=["some_xcat", "other_xcat"], cids=["cid1", "cid2", "cid4"])
    df = pd.concat([df, dfb], axis=0)

    df_res = linear_combination_adjustment(
        df,
        adj_zns_xcat="adj_zns",
        weights_xcat="weights",
        min_score=-3,
        coeff_new=0.5,
        adj_name="lincomb",
    )
    print(df_res)
