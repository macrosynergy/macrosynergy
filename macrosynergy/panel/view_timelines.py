
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.check_availability import reduce_df


def view_timelines(df: pd.DataFrame, xcats: List[str] = None,  cids: List[str] = None,
                   intersect: bool = False, val: str = 'value',
                   cumsum: bool = False, start: str = '2000-01-01', end: str = None,
                   ncol: int = 3, same_y: bool = True, all_xticks: bool = False,
                   title: str = None, title_adj: float = 0.95,
                   xcat_labels: List[str] = None, label_adj: float = 0.05,
                   cs_mean: bool = False, size: Tuple[float] = (12, 7),
                   aspect: float = 1.7, height: float = 3):

    """Displays a facet grid of time line charts of one or more categories.

    :param <pd.Dataframe> df: standardized DataFrame with the necessary columns:
        'cid', 'xcats', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: extended categories to plot. Default is all in DataFrame.
    :param <List[str]> cids: cross sections to plot. Default is all in DataFrame.
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
    :param <List[str]> xcat_labels: labels to be used for xcats. If not defined, the
        labels will be identical to extended categories.
    :param <float> label_adj: parameter that sets bottom of figure to fit the label.
        Default is 0.05.
    :param <bool> cs_mean: if True this adds a line of cross-sectional averages to
        the line charts. This is only allowed for function calls with a single
        category. Default is False.
    :param <Tuple[float]> size: two-element tuple setting width/height of single cross
        section plot. Default is (12, 7). This is irrelevant for facet grid.
    :param <float> aspect: width-height ratio for plots in facet. Default is 1.7.
    :param <float> height: height of plots in facet. Default is 3.

    """

    df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")

    cs_mean_error = f"cs_mean parameter must be a Boolean Object."
    assert isinstance(cs_mean, bool), cs_mean_error
    error = f"cs_mean can only be set to True if a single category is passed. The " \
            f"received categories are {xcats}."
    if cs_mean:
        assert (len(xcats) == 1), error

    df, xcats, cids = reduce_df(df, xcats, cids, start, end,
                                out_all=True, intersect=intersect)

    if cumsum:
        df[val] = df.sort_values(['cid', 'xcat',
                                  "real_date"])[['cid', 'xcat',
                                                 val]].groupby(['cid', 'xcat']).cumsum()

    sns.set(style='darkgrid')
    if len(cids) == 1:
        sns.set(rc={'figure.figsize': size})
        ax = sns.lineplot(data=df, x='real_date', y=val,
                          hue='xcat', estimator=None, sizes=size)

        plt.axhline(y=0, c=".5")
        handles, labels = ax.get_legend_handles_labels()
        label = labels[0:] if xcat_labels is None else xcat_labels
        ax.legend(handles=handles[0:], labels=label)
        ax.set_xlabel("")
        ax.set_ylabel("")

        if title is not None:
            plt.title(title)
    else:
        # Utilise a Facet Grid for instances where a large number of cross-sections are
        # defined & plotted. Otherwise the line chart becomes too congested.
        fg = sns.FacetGrid(data=df, col='cid', col_wrap=ncol,
                           sharey=same_y, aspect=aspect,
                           height=height, col_order=cids)
        fg.map_dataframe(sns.lineplot, x='real_date', y=val,
                         hue='xcat', hue_order=xcats, estimator=None)

        if cs_mean:
            axes = fg.axes.flatten()

            dfw = df.pivot(index='real_date', columns='cid',
                           values='value')
            cross_mean = dfw.mean(axis=1)
            cross_mean = pd.DataFrame(data=cross_mean.to_numpy(),
                                      index=cross_mean.index,
                                      columns=['average'])
            cross_mean = cross_mean.reset_index(level=0)
            for ax in axes:
                sns.lineplot(data=cross_mean, x='real_date', y='average', color='red',
                             ax=ax, label=f"cross-sectional average of {xcats[0]}.")

            handles, labels = ax.get_legend_handles_labels()

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

        if len(xcats) > 1 or cs_mean:
            fg.fig.legend(handles=handles, labels=labels,
                          loc='lower center', ncol=3)
            fg.fig.subplots_adjust(bottom=label_adj,
                                   top=title_adj)

    # Add x-axis tick labels to all axes in grid.
    if all_xticks:
        for ax in fg.axes.flatten():
            ax.tick_params(labelbottom=True, pad=0)

    plt.show()


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'NZD']
    xcats = ['XR', 'CRY', 'INFL']
    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                'sd_mult'])
    df_cids.loc['AUD', ] = ['2010-01-01', '2020-12-31', 0.2, 0.2]
    df_cids.loc['CAD', ] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP', ] = ['2012-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD', ] = ['2012-01-01', '2020-09-30', -0.1, 3]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])

    df_xcats.loc['XR', ] = ['2010-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['INFL', ] = ['2015-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY', ] = ['2013-01-01', '2020-10-30', 1, 2, 0.95, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfdx = dfd[~((dfd['cid'] == 'AUD') & (dfd['xcat'] == 'XR'))]

    view_timelines(dfd, xcats=['XR', 'CRY'], cids=cids[0],
                   size=(10, 5), title='AUD Return and Carry')

    view_timelines(dfd, xcats=['XR', 'CRY', 'INFL'], cids=cids[0],
                   xcat_labels=['Return', 'Carry', 'Inflation'],
                   title='AUD Return, Carry & Inflation')

    view_timelines(dfd, xcats=['CRY'], cids=cids, ncol=2, title='Carry',
                   cs_mean=True)

    view_timelines(dfd, xcats=['XR'], cids=cids, ncol=2,
                   cumsum=True, same_y=False, aspect=2)

    dfd.info()