import unittest
import pandas as pd
from typing import List, Tuple, Optional, Dict
from tests.simulate import make_qdf
from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.panel import view_metrics
import matplotlib

class TestAll(unittest.TestCase):

    def dataframe_construction(self):
        self.cids: List[str] = ['AUD', 'CAD', 'GBP', 'NZD']
        self.xcats: List[str] = ['XR', 'CRY', 'INFL']
        self.metrics: List[str] = ['value', 'grading', 'eop_lag', 'mop_lag']


if __name__ == '__main__':
    unittest.main()