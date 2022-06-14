
import random
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.cluster.hierarchy as sch
from matplotlib import pyplot as plt
from typing import List, Union, Tuple

from macrosynergy.management.check_availability import reduce_df
from macrosynergy.management.simulate_quantamental_data import make_qdf


def correl_matrix(df: pd.DataFrame, xcats: Union[str, List[str]] = None,
                  cids: List[str] = None, start: str = '2000-01-01',
                  end: str = None, val: str = 'value', freq: str = None,
                  cluster: bool = False, lags: List[int] = None,
                  title: str = None, size: Tuple[float] = (14, 8),
                  max_color: float = None):
    """
    Visualize correlation across categories or cross-sections of panels.

    :param <pd.Dataframe> df: standardized JPMaQS dataframe with the necessary columns:
        'cid', 'xcats', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: extended categories to be correlated. Default is all in the
        dataframe. If xcats contains only one category the correlation coefficients
        across cross sections are displayed. If xcats contains more than one category the
        correlation coefficients across categories are displayed.
    :param <List[str]> cids: cross sections to be correlated. Default is all in the
        dataframe.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df
        is used.
    :param <str> val: name of column that contains the values of interest. Default is
        'value'.
    :param <str> freq: frequency option. Per default the correlations are calculated
        based on the native frequency of the datetimes in 'real_date', which is business
        daily. Down-sampling options include weekly ('W'), monthly ('M'), or quarterly
        ('Q') mean.
    :param <bool> cluster: if True the series in the correlation matrix are reordered
        by hierarchical clustering. Default is False.
    :param <List[int]> lags: allows for looking at lagged correlation between a set of
        features & targets.
        The parameter requires passing the number of lags corresponding to the numeracy
        of cross-sections, if modelling on a single category, or categories if the number
        of categories is greater than one. If a feature is not being lagged, pass in
        zero.
    :param <str> title: chart heading. If none is given, a default title is used.
    :param <Tuple[float]> size: two-element tuple setting width/height of figure. Default
        is (14, 8).
    :param <float> max_color: maximum values of positive/negative correlation
        coefficients for color scale. Default is none. If a value is given it applies
        symmetrically to positive and negative values.

    N.B:. The function displays the heatmap of a correlation matrix across categories or
    cross-sections (depending on which parameter has received multiple elements).
    """

    if freq is not None:
        freq_options = ['W', 'M', 'Q']
        error_message = f"Frequency parameter must be one of the following options:" \
                        f"{freq_options}."
        assert freq in freq_options, error_message

    xcats = xcats if isinstance(xcats, list) else [xcats]

    min_color = None if max_color is None else -max_color

    df, xcats_post, cids = reduce_df(df, xcats, cids, start, end, out_all=True)

    s_date = df['real_date'].min().strftime('%Y-%m-%d')
    e_date = df['real_date'].max().strftime('%Y-%m-%d')

    if len(xcats) == 1:
        df_w = df.pivot(index='real_date', columns='cid', values=val)

        if freq is not None:
            df_w = df_w.resample(freq).mean()

        if title is None:
            title = f'Cross-sectional correlation of {xcats[0]} from {s_date} to ' \
                    f'{e_date}'

    else:

        df_w = df.pivot(index=('cid', 'real_date'), columns='xcat', values=val)

        # Down-sample according to the passed frequency.
        if freq is not None:
            df_w = df_w.groupby([pd.Grouper(level='cid'),
                                 pd.Grouper(level='real_date', freq=freq)]
                                ).mean()

        # Apply the lag mechanism, to the respective categories, after the down-sampling.
        if lags is not None:
            lag_xcats = "The number of defined lags must match the number of " \
                        "categories the correlation matrix is defined over."
            assert len(lags) == len(xcats), lag_xcats

            lag_xcat = dict(zip(xcats, lags))
            i = 1
            for xcat, shift in lag_xcat.items():

                # Handle for multi-index DataFrame. The interior index represents the
                # timestamps.
                df_w[xcat] = df_w.groupby(level=0)[xcat].shift(shift)
                df_w = df_w.rename(columns={xcat: xcat + f"_L{i}"})
                i += 1

        if title is None:
            title = f'Cross-category correlation from {s_date} to {e_date}'

    sns.set(style="ticks")

    corr = df_w.corr()

    if cluster:
        # Pairwise distances between observations in n-dimensional space. If y is a 1-D
        # condensed distance matrix, then y must be a nCk sized vector (k = 2, pairwise).
        d = sch.distance.pdist(corr)
        # Perform hierarchical / agglomerative clustering. The clustering method used is
        # Farthest Point Algorithm.
        L = sch.linkage(d, method='complete')
        # Returns an (n - 1) by four matrix. The first two columns represent the "nodes"
        # being merged. The third column is the euclidean distance between the "nodes"
        # being merged and the fourth column equates to the number of original
        # observations in the newly formed cluster.
        # The second parameter is the distance threshold, t, which will determine the
        # "number" of clusters. If the distance threshold is too small, none of the data
        # points will form a cluster, so n different clusters are returned.
        # If there are any clusters, the categories contained in the cluster will be
        # adjacent.
        ind = sch.fcluster(L, 0.5 * d.max(), 'distance')
        columns = [corr.columns.tolist()[i] for i in list((np.argsort(ind)))]
        corr = corr.loc[columns, columns]

    # Mask for the upper triangle.
    # Return a copy of an array with the elements below the k-th diagonal zeroed. The
    # mask is implemented because correlation coefficients are symmetric.
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=size)
    sns.heatmap(corr, mask=mask, cmap='vlag_r', center=0,
                vmin=min_color, vmax=max_color,
                square=False, linewidths=.5,
                cbar_kws={"shrink": .5})

    ax.set(xlabel='', ylabel='')
    ax.set_title(title, fontsize=14)

    plt.show()


if __name__ == "__main__":

    # Un-clustered correlation matrices.

    random.seed(1)
    cids = ['AUD', 'CAD', 'GBP', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']
    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest',
                                                'mean_add', 'sd_mult'])
    df_cids.loc['AUD', ] = ['2010-01-01', '2020-12-31', 0.1, 1]
    df_cids.loc['CAD', ] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP', ] = ['2012-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD', ] = ['2012-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR', ] = ['2010-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY', ] = ['2011-01-01', '2020-10-30', 1, 2, 0.95, 0.5]
    df_xcats.loc['GROWTH', ] = ['2010-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
    df_xcats.loc['INFL', ] = ['2010-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    # Lag inflation by 60 business days - 3 months.
    correl_matrix(dfd, xcats=['GROWTH', 'INFL'], cids=cids,
                  lags= [0, 60], max_color=0.1)
    # Down-sample to monthly frequency and then apply the lags.
    correl_matrix(dfd, xcats=['GROWTH', 'INFL'], cids=cids, freq='M',
                  lags= [0, 3], max_color=0.1)

    correl_matrix(dfd, xcats=xcats, cids=cids,
                  lags=[0, 0, 60, 0], freq="Q")

    # Clustered correlation matrices.

    random.seed(1)
    cids = ['AUD', 'CAD', 'GBP', 'USD', 'NZD', 'EUR']
    cids_dmsc = ["CHF", "NOK", "SEK"]
    cids_dmec = ["DEM", "ESP", "FRF", "ITL", "NLG"]
    cids += cids_dmec
    cids += cids_dmsc
    xcats = ['XR', 'CRY']

    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest',
                                                'mean_add', 'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2012-01-01', '2020-11-30', -0.2, 0.5]
    df_cids.loc['USD'] = ['2010-01-01', '2020-12-30', -0.2, 0.5]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]
    df_cids.loc['EUR'] = ['2002-01-01', '2020-09-30', -0.2, 2]
    df_cids.loc['DEM'] = ['2003-01-01', '2020-09-30', -0.3, 2]
    df_cids.loc['ESP'] = ['2003-01-01', '2020-09-30', -0.1, 2]
    df_cids.loc['FRF'] = ['2003-01-01', '2020-09-30', -0.2, 2]
    df_cids.loc['ITL'] = ['2004-01-01', '2020-09-30', -0.2, 0.5]
    df_cids.loc['NLG'] = ['2003-01-01', '2020-12-30', -0.1, 0.5]
    df_cids.loc['CHF'] = ['2003-01-01', '2020-12-30', -0.3, 2.5]
    df_cids.loc['NOK'] = ['2010-01-01', '2020-12-30', -0.1, 0.5]
    df_cids.loc['SEK'] = ['2010-01-01', '2020-09-30', -0.1, 0.5]

    df_xcats = pd.DataFrame(index = xcats, columns = ['earliest', 'latest', 'mean_add',
                                                      'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2012-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2010-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    start = '2012-01-01'
    end = '2020-09-30'
    correl_matrix(df=dfd, xcats='XR', cids=cids, start=start, end=end,
                  val='value', freq=None, cluster=True,
                  title='Correlation Matrix', size=(14, 8), max_color=None)

