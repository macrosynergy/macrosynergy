import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf, make_qdf_black
from macrosynergy.management.shape_dfs import reduce_df

def negate_interpolation(df: pd.DataFrame, xcats: List[str] = None):
    """
    Negate Seaborn's default approach to handling missing dates by reinstating the
    missing dates, blacklist periods, with np.inf values. The values will allow for a
    discontinuous graph demonstrating that blacklist periods are excluded from analysis.

    :param <pd.Dataframe> df: standardized dataframe with the necessary columns:
        'cid', 'xcats', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: extended categories to plot. Default is all in dataframe.

    :return <pd.DataFrame>: updated DataFrame with the previously missing dates populated
        with values.
    """

    first_date = min(df['real_date'])
    last_date = max(df['real_date'])

    complete_dates = pd.date_range(start=first_date, end=last_date, freq='B')
    print(complete_dates)

def view_timelines(df: pd.DataFrame, xcats: List[str] = None,  cids: List[str] = None,
                   intersect: bool = False, val: str = 'value',
                   cumsum: bool = False, start: str = '2000-01-01', end: str = None,
                   ncol: int = 3, same_y: bool = True, all_xticks: bool = False,
                   title: str = None, title_adj: float = 0.95,
                   xcat_labels: List[str] = None, label_adj: float = 0.05,
                   size: Tuple[float] = (12, 7), aspect: float = 1.7, height: float = 3):

    """Displays a facet grid of time line charts of one or more categories

    :param <pd.Dataframe> df: standardized dataframe with the necessary columns:
        'cid', 'xcats', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: extended categories to plot. Default is all in dataframe.
    :param <List[str]> cids: cross sections to plot. Default is all in dataframe.
        If this contains only one cross section a single line chart is created.
    :param <bool> intersect: if True only retains cids that are available for all xcats.
        Default is False.
    :param <str> val: name of column that contains the values of interest.
        Default is 'value'.
    :param <bool> cumsum: plot cumulative sum of the values over time. Default is False.
    :param <str> start: earliest date in ISO format. Default is earliest date available.
    :param <str> end: latest date in ISO format. Default is latest date available.
    :param <int> ncol: number of columns in facet grid. Default is 3.
    :param <bool> same_y: if True (default) all plots in facet grid share same y axis.
    :param <bool> all_xticks:  if True x-axis tick labels are added to all plots in grid.
        Default is False, i.e only the lowest row displays the labels.
    :param <str> title: chart heading. Default is no title.
    :param <float> title_adj: parameter that sets top of figure to accommodate title.
        Default is 0.95.
    :param <List[str]> xcat_labels: labels to be used for xcats if not identical to
        extended categories.
    :param <float> label_adj: parameter that sets bottom of figure to fit the label.
        Default is 0.05.
    :param <Tuple[float]> size: two-element tuple setting width/height of single cross
        section plot. Default is (12, 7). This is irrelevant for facet grid.
    :param <float> aspect: width-height ratio for plots in facet. Default is 1.7.
    :param <float> height: height of plots in facet. Default is 3.

    """

    df, xcats, cids = reduce_df(df, xcats, cids, start, end, out_all=True,
                                intersect=intersect)
    negate_interpolation(df=df, xcats=xcats)

    # Handle for blacklist periods. The application of blacklist removes the respective
    # dates from the dataframe which results in the intermediary, missing, dates being
    # computed via linear interpolation. This is misleading and is an erroneous
    # representation of the data: the missing dates should not be populated with values
    # but instead there should be a break in the graph.

    if cumsum:
        df[val] = df.sort_values(['cid', 'xcat', "real_date"])[['cid', 'xcat', val]].\
            groupby(['cid', 'xcat']).cumsum()

    sns.set(style='darkgrid')
    if len(cids) == 1:
        sns.set(rc={'figure.figsize': size})
        ax = sns.lineplot(data=df, x='real_date', y=val, hue='xcat', ci=None,
                          sizes=size)
        plt.axhline(y=0, c=".5")
        handles, labels = ax.get_legend_handles_labels()
        label = labels[0:] if xcat_labels is None else xcat_labels
        ax.legend(handles=handles[0:], labels=label)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if title is not None:
            plt.title(title)
    else:
        fg = sns.FacetGrid(data=df, col='cid', col_wrap=ncol, sharey=same_y,
                           aspect=aspect, height=height,
                           col_order=cids)
        fg.map_dataframe(sns.lineplot, x='real_date', y=val, hue='xcat',
                         legend=False,
                         hue_order=xcats, ci=None)
        fg.map(plt.axhline, y=0, c=".5")
        fg.set_titles(col_template='{col_name}')
        fg.set_axis_labels('', '')
        if title is not None:
            fg.fig.suptitle(title, fontsize=20)
            fg.fig.subplots_adjust(top=title_adj)
        if len(xcats) > 1:
            handles = fg._legend_data.values()
            if xcat_labels is None:
                labels = fg._legend_data.keys()
            else:
                labels = xcat_labels
            fg.fig.legend(handles=handles, labels=labels, loc='lower center',
                          ncol=3)  # add legend to bottom of figure
            fg.fig.subplots_adjust(bottom=label_adj,
                                   top=title_adj)  # lift bottom to respect legend

    if all_xticks:  # add x-axis tick labels to all axes in grid
        for ax in fg.axes.flatten():
            ax.tick_params(labelbottom=True, pad=0)

    plt.show()


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'NZD']
    xcats = ['XR', 'CRY']
    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                'sd_mult'])
    df_cids.loc['AUD', ] = ['2010-01-01', '2020-12-31', 0.2, 3]
    df_cids.loc['CAD', ] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP', ] = ['2012-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD', ] = ['2012-01-01', '2020-09-30', -0.1, 3]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR', ] = ['2010-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY', ] = ['2012-01-01', '2020-10-30', 1, 2, 0.95, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfdx = dfd[~((dfd['cid'] == 'AUD') & (dfd['xcat'] == 'XR'))]

    black = {'AUD': ['2010-01-03', '2013-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    dfd_reduce = reduce_df(df=dfd, xcats=xcats,  cids=cids, start = '2010-01-01',
                           end = '2020-12-31', blacklist=black)
    print(dfd_reduce)

    blackout = {'AUD': ('2010-01-03', '2013-12-31'), 'GBP': ('2018-01-01', '2100-01-01')}
    dfd_black = make_qdf_black(df_cids, df_xcats, blackout=blackout)

    # Test the interpolation that is being applied to populate missing values: ultimately
    # inactive periods should be excluded from the graphical display.
    view_timelines(dfd_reduce, xcats=['XR', 'CRY'], cids=cids, ncol=2,
                   xcat_labels=['Return', 'Carry'],
                   title='Carry and return', title_adj=0.9, label_adj=0.1,
                   aspect=1, height=5)

    view_timelines(dfd, xcats=['XR', 'CRY'], cids=cids[0], ncol=1, size=(10, 5),
                   title='AUD return and carry')
    view_timelines(dfd, xcats=['XR', 'CRY'], cids=cids[0], ncol=1,
                   xcat_labels=['Return', 'Carry'],
                   title='AUD return and carry')
    view_timelines(dfd, xcats=['CRY'], cids=cids, ncol=2, title='Carry')
    view_timelines(dfd, xcats=['XR', 'CRY'], cids=cids, ncol=2, title='Return and carry',
                   all_xticks=True)
    view_timelines(dfd, xcats=['XR'], cids=cids, ncol=2, cumsum=True, same_y=False,
                   aspect=2)

    dfd.info()