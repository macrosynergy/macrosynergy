import numpy as np
import pandas as pd
from typing import List
from itertools import groupby
import random
from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.management.simulate_quantamental_data import make_qdf_black

tuple_ = lambda dates, index_tr, length: (dates[index_tr], dates[index_tr + (length - 1)])

def make_blacklist(df: pd.DataFrame, xcat: str, cids: List[str] = None,
                   start: str = None, end: str = None):

    """
    Converts binary category of standardized dataframe into a standardized dictionary
    that can serve as a blacklist for cross-sections in further analyses

    :param <pd.Dataframe> df: standardized DataFrame with following columns:
        'cid', 'xcats', 'real_date' and 'value'.
    :param <str> xcat: category with binary values, where 1 means blacklisted and 0 means
        not blacklisted.
    :param List<str> cids: list of cross-sections which are considered in the formation
        of the blacklist. Per default, all available cross sections are considered.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        for which the respective category is available is used.
    :param <str> end: latest date in ISO format. Default is None and latest date
        for which the respective category is available is used.

    :return <dict>: standardized dictionary with cross-sections as keys and tuples of
        start and end dates of the blacklist periods in ISO formats as values.
        If one cross section has multiple blacklist periods, numbers are added to the
        keys (i.e. TRY_1, TRY_2, etc.)
    """

    assert all(list(map(lambda val: val == 1 or val == 0, df['value'].to_numpy())))

    dfd = reduce_df(df=df, xcats=xcat, cids=cids, start=start, end=end)

    df_pivot = dfd.pivot(index='real_date', columns='cid', values='value')

    dates = df_pivot.index
    cids_df = list(df_pivot.columns)

    dates_dict = {}
    for cid in cids_df:
        count = 0

        column = df_pivot[cid]
        cut_off_start = column.first_valid_index()
        cut_off_end = column.last_valid_index()

        condition_1 = np.where(dates == cut_off_start)[0]
        condition_2 = np.where(dates == cut_off_end)[0]

        cut_off_start = next(iter(condition_1))
        cut_off_end = next(iter(condition_2))

        column = column.to_numpy()[cut_off_start:(cut_off_end + 1)]
        # To handle for the NaN values, the datatype will be floating point values.

        column = column.astype(dtype=np.uint8)

        index_tr = cut_off_start
        for k, v in groupby(column):
            v = list(v)  # Instantiate the iterable in memory.
            length = len(v)

            if not sum(v) ^ 0:
                if count == 0:
                    dates_dict[cid] = (dates[index_tr], dates[index_tr + (length - 1)])
                elif count == 1:
                    val = dates_dict.pop(cid)
                    dates_dict[cid + '_1'] = val
                    count += 1
                    dates_dict[cid + '_' + str(count)] = tuple_(dates, index_tr, length)
                else:
                    dates_dict[cid + '_' + str(count)] = tuple_(dates, index_tr, length)

                count += 1

            index_tr += length

    return dates_dict


if __name__ == "__main__":

    cids = ['AUD', 'GBP', 'CAD', 'USD']

    xcats = ['FXXR_NSA']

    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31']
    df_cids.loc['GBP'] = ['2011-01-01', '2020-11-30']
    df_cids.loc['CAD'] = ['2011-01-01', '2021-11-30']
    df_cids.loc['USD'] = ['2011-01-01', '2020-12-30']

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest'])
    df_xcats.loc['FXXR_NSA'] = ['2010-01-01', '2021-11-30']

    blackout = {'AUD': ('2010-01-12', '2010-06-14'),
                'USD': ('2011-08-17', '2011-09-20'),
                'CAD_1': ('2011-01-04', '2011-01-23'),
                'CAD_2': ('2013-01-09', '2013-04-10'),
                'CAD_3': ('2015-01-12', '2015-03-12'),
                'CAD_4': ('2021-11-01', '2021-11-20')}

    print(blackout)
    random.seed(2)
    df = make_qdf_black(df_cids, df_xcats, blackout=blackout)

    dates_dict = make_blacklist(df, xcat=['FXXR_NSA'], cids=None,
                                start=None, end=None)

    # If the output, from the below printed dictionary, differs from the above defined
    # dictionary, it should be by a date or two, as the construction of the dataframe,
    # using make_qdf_black(), will account for the dates received, in the dictionary,
    # being weekends. Therefore, if any of the dates, for the start or end of the
    # blackout period are Saturday or Sunday, the date for will be shifted to the
    # following Monday. Hence, a break in alignment from "blackout" to "dates_dict".
    print(dates_dict)