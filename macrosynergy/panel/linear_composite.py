import numpy as np
import pandas as pd
from typing import *
from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.management.simulate_quantamental_data import make_qdf


def linear_composite(df: pd.DataFrame, xcats: List[str], weights=None, signs=None,
                     cids: List[str] = None, start: str = None, end: str = None,
                     complete_xcats: bool = True, nan_treatment: Optional[Union[str, int]] = None,
                     new_xcat="NEW"):
    """
    Returns new category panel as linear combination of others as standard dataframe

    :param <pd.DataFrame> df:  standardized JPMaQS DataFrame with the necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
    :param <List[str]> xcats: all extended categories used for the linear combination.
    :param <List[float]> weights: weights of all categories in the linear combination.
        These must correspond to the order of xcats and the sum will be coerced to unity.
        Default is equal weights.
    :param <List[float]> signs: signs with which the categories are combined.
        These must be 1 or -1 for positive and negative and correspond to the order of
        xcats. Default is all positive.
    :param <List[str]> cids: cross-sections for which the linear combination is ti be
        calculated. Default is all cross-section available for the respective category.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        for which the respective category is available is used.
    :param <str> end: latest date in ISO format. Default is None and latest date for
        which the respective category is available is used.
    :param <bool> complete_xcats: If True combinations are only calculated for
        observation dates on which all xcats are available. If False a combination of the
        available categories is used.
    :param <str> nan_treatment: If an integer is passed, the NaNs are replaced by the integer.
        Passing "drop" will drop all rows with NaNs. Default is None and no treatment is done.        
    :param <str> new_xcat: name of new composite xcat. Default is "NEW".

    """
    listtypes = (list, np.ndarray, pd.Series)

    def make_new_xcat(cid : Union[str, List[str]], new_xcat : str = new_xcat) -> Union[str, List[str]]:
        if isinstance(cid, str):
            return f"{cid}_{new_xcat}"
        elif isinstance(cid, pd.Series):
            return cid.apply(make_new_xcat)
        elif isinstance(cid, listtypes):
            return [make_new_xcat(c) for c in cid]
        
    # checking inputs; casting weights and signs to np.array    
    if weights is not None:
        assert isinstance(weights, listtypes), "weights must be list, np.ndarray or pd.Series"
        if isinstance(weights, np.ndarray):
            assert weights.ndim == 1, "weights must be 1-dimensional if passed as np.ndarray"
        else:
            weights = np.array(weights)
    else:
        weights = np.ones(len(xcats)) * (1 / len(xcats))
    
    if signs is not None:
        assert isinstance(signs, listtypes), "signs must be list, np.ndarray or pd.Series"
        if isinstance(signs, np.ndarray):
            assert signs.ndim == 1, "signs must be 1-dimensional if passed as np.ndarray"
        if isinstance(signs, list):
            signs = np.array(signs)    
    else:
        signs = np.ones(len(xcats))
        
    assert len(xcats) == len(weights) == len(signs), "xcats, weights, and signs must have same length"
    if not np.isclose(np.sum(weights), 1):
        print("WARNING: weights do not sum to 1. They will be coerced to sum to 1. w←w/∑w")
        weights = weights / np.sum(weights)
    if not np.all(np.isin(signs, [1, -1])):
        print("WARNING: signs must be 1 or -1. They will be coerced to 1 or -1.")
        signs = np.abs(signs) / signs # should be faster?
        
    # main function is here and below.
    weights = pd.Series(weights * signs, index=xcats)
    
    if start is None:
        start = df['real_date'].min()
    if end is None:
        end = df['real_date'].max()
    
    dfc: pd.DataFrame = reduce_df(df, cids=cids, xcats=xcats, start=start, end=end)
    
    # @mikiinterfiore 's version
    # creating a dataframe with the xcats as columns. each row is an observation for a combination CID-Date
    dfc_wide = dfc.set_index(['cid', 'real_date', 'xcat'])['value'].unstack(level=2)
    # creating a dataframe for the weights with the same index as dfc_wide. 
    # each column will be a weight, with the same value all along
    weights_wide = pd.DataFrame(data=[weights.sort_index()], 
                    index=dfc_wide.index, columns=dfc_wide.columns)
    # boolean mask to help us work out the calcs
    mask = dfc_wide.isna()
    # pandas series with an index equal to the index of dfc_wide, and a value equal to the sum of the weights
    weights_sum = weights_wide[~mask].abs().sum(axis=1)
    # re-weighting the weights to sum to 1 considering the available xcats
    adj_weights_wide = weights_wide[~mask].div(weights_sum, axis=0)
    # final single series: the linear combination of the xcats and the weights
    
    out_df = (dfc_wide * adj_weights_wide).sum(axis=1)
    
    if complete_xcats:
        out_df[mask.any(axis=1)] = np.NaN
    else:
        out_df[mask.all(axis=1)] = np.NaN

    out_df = out_df.reset_index().rename(columns={0: 'value'})
    out_df['xcat'] = new_xcat # make_new_xcat(out_df['cid'])
    out_df = out_df[['cid', 'xcat', 'real_date', 'value']]

    return out_df    
    
if __name__ == "__main__":
    cids = ['AUD', 'CAD', 'GBP']
    xcats = ['XR', 'CRY', 'INFL']
    dates  = pd.date_range('2000-01-01', '2000-01-03')
    total_entries = len(cids) * len(xcats) * len(dates)
    randomints = list(np.arange(total_entries) - total_entries // 2)
    lx = [[cid, xcat, date, randomints.pop()] 
            for cid in cids 
            for xcat in xcats 
            for date in dates]
    dfst = pd.DataFrame(lx, columns=['cid', 'xcat', 'real_date', 'value'])
    missing_idx = [9, 18, 19, 20, 23, 25, 26]
    dfst.loc[missing_idx, 'value'] = np.NaN

    weights = [1, 2, 3]
    signs = [-1, 1, 1]
    
    dflc = linear_composite(df=dfst, xcats=xcats, cids=cids, 
                            weights=weights, signs=signs,
                            complete_xcats=True)
    print(dflc)
