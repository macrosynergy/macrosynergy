from typing import Optional, List

from macrosynergy.management.types import QuantamentalDataFrame

from macrosynergy.panel.adjust_weights import adjust_weights


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
    return adjust_weights(
        df=df,
        weights_xcat=weights_xcat,
        adj_zns_xcat=adj_zns_xcat,
        cids=cids,
        method="lincomb",
        params=dict(min_score=min_score, coeff_new=coeff_new),
        normalize=normalize,
        normalize_to_pct=normalize_to_pct,
        adj_name=adj_name,
    )


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_test_df

    df = make_test_df(
        xcats=["weights", "adj_zns", "other_xcat"],
        cids=["cid1", "cid2", "cid3", "cid4"],
    )

    df_res = linear_combination_adjustment(
        df,
        adj_zns_xcat="adj_zns",
        weights_xcat="weights",
        min_score=-3,
        coeff_new=0.5,
        adj_name="lincomb",
    )
    print(df_res)
