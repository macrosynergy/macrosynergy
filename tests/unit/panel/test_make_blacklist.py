import unittest
from macrosynergy.panel.make_blacklist import blacklist

class TestAll(unittest.TestCase):

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

    def dates_generator(date, n_days):
        start = datetime.datetime.strptime(date, "%Y-%m-%d")
        dates = [start + datetime.timedelta(days = x) for x in range(0, n_days)]
        return dates

    def dataframe_generator(cids, xcats, timestamps, sequence):

        n_cids = len(cids)

        value = [int(round(random.random())) for i in range(timestamps - sequence)]
        value = [1 for i in range(sequence)] + value
        n_vals = len(value)

        cids_ = np.empty(n_vals, dtype = object)

        cids_ = dataframe_concatenation(cids_, cids, n_vals, n_cids)
        ## Itertools groubpy Class is a heavily optimised algorithm written in high-performance C code by Enthought.
        cids_list = [len(list(v)) for k, v in groupby(cids_)]

        dates_l = []
        for no in cids_list:

            dates_l += dates_generator("2000-01-01", no)

        category = np.repeat('FXXR_NSA', timestamps)
        data = np.column_stack((cids_, category, np.array(dates_l), value))
        cols = ['cid', 'xcats', 'real_date', 'value']

        df = pd.DataFrame(data = data, columns = cols)

        return df

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

if __name__ == '__main__':
    unittest.main()