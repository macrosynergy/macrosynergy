import unittest
import os
import random
from itertools import groupby
import numpy as np
import pandas as pd
import datetime
from macrosynergy.panel.make_blacklist import make_blacklist


class TestAll(unittest.TestCase):

    @staticmethod
    def dataframe_concatenation(arr, cids, n_vals, n_cids):

        tracker = 0
        n_vals -= n_cids

        for cid in cids[:-1]:
            index = random.randint(1, n_vals)
            arr[tracker:(tracker + index)] = np.repeat(cid, index)

            tracker += index
            n_vals -= index

        residual = arr.size - tracker
        arr[tracker:] = np.repeat(cids[-1], residual)

        return arr

    @staticmethod
    def dates_generator(date, n_days):
        start = datetime.datetime.strptime(date, "%Y-%m-%d")
        dates = [start + datetime.timedelta(days=x) for x in range(0, n_days)]
        return dates

    def dataframe_generator(self):

        cids = ['AUD', 'USD', 'GBP', 'CAD']
        xcat = ['FXXR_NSA']
        timestamps = 1000
        sequence = 20

        n_cids = len(cids)

        value = [int(round(random.random())) for i in range(timestamps - sequence)]
        value = [1 for i in range(sequence)] + value
        n_vals = len(value)

        cids_ = np.empty(n_vals, dtype=object)

        cids_ = self.dataframe_concatenation(cids_, cids, n_vals, n_cids)
        # Itertools groubpy Class is a heavily optimised algorithm written in
        # high-performance C code by Enthought.
        cids_list = [len(list(v)) for k, v in groupby(cids_)]

        dates_l = []
        for no in cids_list:
            dates_l += self.dates_generator("2000-01-01", no)

        category = np.repeat(xcat[0], timestamps)
        data = np.column_stack((cids_, category, np.array(dates_l), value))
        cols = ['cid', 'xcat', 'real_date', 'value']

        self.df = pd.DataFrame(data=data, columns=cols)

    @staticmethod
    def sequence_ones(values):

        copy_values = values.copy()

        values = iter(values)
        value = next(values)

        i = 0
        while not value:
            count = 0
            try:
                value = next(values)
            except StopIteration:
                return count
            i += 1

        count = 1
        for val in values:
            i += 1
            if not val:
                continue
            elif val != copy_values[(i - 1)]:
                count += 1

            elif val:
                continue

        return count

    @staticmethod
    def dict_counter(dict_, cids_):
        dict_keys = list(dict_.keys())

        dict_tracker = dict(zip(cids_, list(np.repeat(0, len(cids_)))))
        for cross in dict_keys:
            dict_tracker[cross[:3]] += 1

        return dict_tracker

    def test_blackout(self):

        self.dataframe_generator()
        dfd = self.df
        shape = dfd.shape
        cross_sections = dfd['cid'].unique()
        cross_sections.sort()

        dfd['value'] = np.arange(0, shape[0])

        with self.assertRaises(AssertionError):
            make_blacklist(dfd, 'FXXR_NSA', cids=list(cross_sections))

        dfd['value'] = np.repeat(0, shape[0])

        dict_ = make_blacklist(dfd, xcat='FXXR_NSA', cids=list(cross_sections))
        # Validate the dictionary is empty if every row in the 'value' column contains
        # a zero.
        self.assertTrue(not bool(dict_))

        # Refresh the DataFrame for the next testing framework.
        self.dataframe_generator()
        dfd = self.df

        dfd['value'] = np.repeat(1, shape[0])
        # If the entire 'value' column is equated to one, the number of keys in the
        # dictionary should equal the number of cross-sections in the dataframe.
        dict_ = make_blacklist(dfd, 'FXXR_NSA', cids=list(cross_sections))

        cross_sections = dfd['cid'].unique()
        cross_sections.sort()
        no_cross_sections = cross_sections.size
        keys = list(dict_.keys())
        self.assertTrue(len(keys) == no_cross_sections)
        # If the dataframe contains only a single blackout period for a cross-section,
        # the key should just be the name of the cross-section.
        # In this situation, indexing is not required: 'AUD_1', 'AUD_2' etc..
        self.assertTrue(sorted(keys) == list(cross_sections))

        df_pivot = dfd.pivot(index='real_date', columns='cid', values='value')

        # Check the start & end dates. All cross-sections have the same start date but
        # varying end dates depending on the number of periods in the series.
        for cid in cross_sections:
            self.assertTrue(str(dict_[cid][0])[:10] == "2000-01-01")

            end_date = df_pivot[cid].last_valid_index()
            self.assertTrue(dict_[cid][1] == end_date)

        # Refresh the DataFrame for the next testing framework.
        self.dataframe_generator()
        dfd = self.df
        df_pivot = dfd.pivot(index='real_date', columns='cid', values='value')
        dates = df_pivot.index

        dict_ = make_blacklist(dfd, 'FXXR_NSA', cids=list(cross_sections))
        dict_tracker = self.dict_counter(dict_, cross_sections)

        # Test the number of keys in the dictionary per cross_section.
        # The number of keys in the dictionary should correspond to the number of
        # sequences of one held in the cross-section's column (on the pivoted dataframe).
        # Unable to validate the dictionary's values, tuples of dates, are correct but if
        # the number of keys match the number of sequences of one, it minimises the
        # scope for potential error in the function.
        # The only aspect that would still be required to check is that the actual
        # "start_date" & "end_date", the two values in the tuple, are correct. But the
        # final check would require a more involved algorithm that might be superfluous
        # given the current checks.
        for cid in cross_sections:
            column = df_pivot[cid]
            last_index = column.last_valid_index()
            last_index = np.where(dates == last_index)[0]
            last_index = next(iter(last_index))

            values = column.tolist()[:(last_index + 1)]

            count = self.sequence_ones(values)
            self.assertTrue(count == dict_tracker[cid])


if __name__ == '__main__':

    print(os.getcwd())
    unittest.main()