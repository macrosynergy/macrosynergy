from typing import List, Optional, Dict

from pandas import DataFrame

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.panel import panel_calculator


def cross_asset_effects(
        df: QuantamentalDataFrame,
        cids: List[str],
        effect_name: str,
        signal_xcats: Dict[str, str],
        weights_xcats: Dict[str, str],
        signal_signs: Optional[Dict[str, int]] = None,
) -> DataFrame:
    """
    Linear combination of a set of categories with corresponding weights and, optionally, signs.
    Corresponding assets' volatilities are generally used as weights.

    Parameters
    ----------
    df : str
        QuantamentalDataFrame with time-series of categories for both values and weights for all cross-sections.
    cids : List[str]
        List of cross-sections to compute the new quantamental category for.
    effect_name : str
        Name of the new quantamental xcat.
    signal_xcats : Dict[str, str]
        Dictionary of asset class names and related signals' time-series specified as xcats, part of df.
    weights_xcats : Dict[str, str]
        Dictionary of asset class names and related weights' time-series specified as xcats, part of df.
    signal_signs : Dict[str, int], optional
        Dictionary of asset class names and related signs in form of +1 / -1.
        Default is None, hence we assume all components contribute positively and proportionately to the final average

    Returns
    -------
    pd.DataFrame
    """
    assert isinstance(signal_xcats, dict), "Please provide a dictionary for assets' xcats"
    assert isinstance(weights_xcats, dict), "Please provide a dictionary for assets' weights"
    assert signal_signs is None or isinstance(signal_signs,
                                              dict), "Please provide a dictionary for assets' signs if needed."

    assert set(signal_xcats.keys()) == set(
        weights_xcats.keys()), "The keys of provided dictionaries for xcats and weights do not match."
    assert len(signal_xcats) == len(
        weights_xcats), "The size of provided dictionaries for xcats and weights do not match."

    if signal_signs is not None:
        assert set(signal_xcats.keys()) == set(
            signal_signs.keys()), "The keys of provided dictionaries for xcats and signs do not match."
        assert len(signal_xcats) == len(
            signal_signs), "The size of provided dictionaries for xcats and signs do not match."
    else:
        signal_signs = {
            k: 1 for k, v in signal_xcats.items()
        }

    weighted_parts_calc = [
        f"( {k.upper()}_SHARE ) * ( {str(signal_signs.get(k))} ) * {v}" for k, v in signal_xcats.items()
    ]

    calcs = [
                # Computing the total weight assigned across xcat elements
                f"WSUM = {' + '.join(weights_xcats.values())}"
            ] + [
                # Computing each category's share
                f"{k.upper()}_SHARE = {v} / WSUM" for k, v in weights_xcats.items()
            ] + [
                # Computing the final indicator as weighted average
                f"{effect_name} = {' + '.join(weighted_parts_calc)}"
            ]

    return panel_calculator(df, calcs=calcs, cids=cids)
