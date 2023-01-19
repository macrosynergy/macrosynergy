import numpy as np
import pandas as pd
from typing import *
from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.management.simulate_quantamental_data import make_qdf


def linear_composite(df: pd.DataFrame, xcats: List[str], weights=None, signs=None,
                     cids: List[str] = None, start: str = '2001-01-01', end: str = None,
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
    def new_cid(cid, new_xcat=new_xcat):
        if isinstance(cid, str):
            return f"{cid}_{new_xcat}"
        elif isinstance(cid, listtypes):
            return [new_cid(c) for c in cid]
        
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
        # signs = np.where(signs >= 0, 1, -1)
        signs  = abs(signs) / signs # should be faster?
        
    # main function is here and below.
    
    weights = weights * signs
    
    if end is None:
        end = df['real_date'].max()
    
    dfc: pd.DataFrame = reduce_df(df, cids=cids, xcats=xcats, start=start, end=end)
    
    out_df = pd.DataFrame(data=0, columns=new_cid(cids),
                          index=df['real_date'].unique(),)
    
  
    uxcats_set = set([f"{c}_{x}" for c in cids for x in xcats]) # user specified cids_xcats
    dfxcats_set = set(dfc['cid'] + '_' + dfc['xcat']) # available cids_xcats in df
    
    for ic, cid in enumerate(cids):
        cid_mask = (dfc['cid'] == cid)
        curr_weights = weights
        if not(set(dfc['xcat'][cid_mask].unique()) == set(xcats)):
            print(f"WARNING: {cid} does not have all xcats. Weights will be adjusted.")
            avail_bools = np.isin(xcats, dfc['xcat'][cid_mask].unique())
            new_weights = weights + weights[~avail_bools] / np.sum(avail_bools)
            new_weights[~avail_bools] = 0
            curr_weights = new_weights
            
        for ix, xcat in enumerate(xcats):
                xcat_mask = cid_mask & (dfc['xcat'] == xcat)
                dfcurr = dfc.loc[xcat_mask, ['real_date', 'value']].set_index('real_date')
                if complete_xcats:
                    if dfcurr.isna().any().any():
                        print(f"WARNING: {cid} does not have {xcat} for all dates. Skipping.")
                        continue # skip next lines and goto next iteration (xcat)
                out_df.loc[dfcurr.index, new_cid(cid)] += dfcurr['value'] * curr_weights[ix]
                
    return out_df    
    
if __name__ == "__main__":
    cids = ['AUD', 'CAD', 'GBP', 'NZD']
    xcats = ['XR', 'CRY', 'INFL', 'BXBGDPRATIO']
    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                'sd_mult'])
    # df_cids.loc['AUD', ] = ['2010-01-01', '2020-12-31', 0.2, 0.2]
    # df_cids.loc['CAD', ] = ['2011-01-01', '2020-11-30', 0, 1]
    # df_cids.loc['GBP', ] = ['2012-01-01', '2020-11-30', 0, 2]
    # df_cids.loc['NZD', ] = ['2012-01-01', '2020-09-30', -0.1, 3]

    df_cids.loc['CAD', ] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP', ] = ['2011-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD', ] = ['2011-01-01', '2020-11-30', -0.1, 3]
    df_cids.loc['AUD', ] = ['2011-01-01', '2020-11-30', 0.2, 0.2]
    

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])

    df_xcats.loc['XR', ] = ['2010-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['INFL', ] = ['2015-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY', ] = ['2013-01-01', '2020-10-30', 1, 2, 0.95, 0.5]
    df_xcats.loc['BXBGDPRATIO', ] = ['2013-01-01', '2020-10-30', 1, 2, 0.95, 0.5]
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    
    # drop rows where cids=AUD and xcats=XR as a test
    dfd = dfd.loc[~((dfd['cid'] == 'GBP') & (dfd['xcat'] == 'XR')), :]
    weights = [1, 100, 150, 200]

    df = linear_composite(df=dfd, xcats=xcats, cids=cids, start='2015-01-01', end='2020-12-31', 
                          weights=weights, complete_xcats=True)
    
    print(df)