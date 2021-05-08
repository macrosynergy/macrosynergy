import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.check_availability import reduce_df


def view_timelines(df: pd.DataFrame, xcats: List[str] = None,  cids: List[str] = None,
                   start: str = '2000-01-01', end: str = None, val: str = 'value', cumsum: bool = False,
                   title: str = None,
                   ncol: int = 3, same_y: bool = True, size: Tuple[float] = (12, 7), aspect: float = 1.7):

    """Display facet grid of time lines of one or more categories"""

    df, xcats, cids = reduce_df(df, xcats, cids, start, end, out_all=True)

    if cumsum:
        df[val] = df[['cid', 'xcat', val]].groupby(['cid', 'xcat']).cumsum()

    sns.set(rc={'figure.figsize': size}, style='darkgrid')
    fg = sns.FacetGrid(df, col='cid', col_wrap=ncol, sharey=same_y, aspect=aspect, col_order=cids)
    fg.map_dataframe(sns.lineplot, x='real_date', y=val, hue='xcat', ci=None)
    fg.map(plt.axhline, y=0, c=".5")
    fg.set_titles(col_template='{col_name}')
    fg.add_legend()
    if title is not None:
        fg.fig.subplots_adjust(top=0.9)
        fg.fig.suptitle(title)
    plt.show()


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'NZD']
    xcats = ['XR', 'CRY']
    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
    df_cids.loc['AUD', ] = ['2010-01-01', '2020-12-31', 0.5, 0.2]
    df_cids.loc['CAD', ] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP', ] = ['2012-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD', ] = ['2012-01-01', '2020-09-30', -0.5, 3]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR', ] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY', ] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    view_timelines(dfd, xcats=['XR'], cids=cids, ncol=2, title='Returns')
    view_timelines(dfd, xcats=['CRY'], cids=cids)

    dfd.info()