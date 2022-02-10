
import numpy as np
import pandas as pd
import random
from typing import List
from macrosynergy.panel.historic_vol import expo_weights, expo_std, flat_std
from macrosynergy.management.shape_dfs import reduce_df_by_ticker
from macrosynergy.panel.converge_row import ConvergeRow
from macrosynergy.management.simulate_quantamental_data import make_qdf

class Basket(object):

    def __init__(self, df:pd.DataFrame, contracts: List[str], ret: str = "XR_NSA",
                 weight_meth: str = 'equal', cry: str = None,
                 start: str = None, end: str = None, blacklist: dict = None,
                 wgt: str = None, basket_tik: str = "GLB_ALL"):


        self.dfx = reduce_df_by_ticker(df, start=start, end=end, ticks=self.tickers,
                                       blacklist=blacklist)
        self.contract = contracts
        self.tickers = self.ticker_list(self)
        self.ret = ret
        self.cry = cry
        self.cry_flag = (cry is not None)
        self.wgt_flag = (wgt is not None) and (weight_meth in ["values", "inv_values"])
        self.wgt = wgt
        self.basket_tik = basket_tik

    def ticker_list(self):

        ticks_ret = [c + self.ret for c in self.contracts]
        tickers = ticks_ret.copy()  # Initiates general tickers list.

        # Boolean for carry being used.
        if self.cry_flag:
            self.__dict__['ticks_cry'] = [c + self.cry for c in self.contracts]
            tickers += self.ticks_cry

        if self.wgt_flag:
            error = f"'wgt' must be a string and not a {type(self.wgt)}."
            assert isinstance(self.wgt, str), error
            self.__dict__['ticks_wgt'] = [c + self.wgt for c in self.contracts]
            tickers += self.ticks_wgt

        return tickers