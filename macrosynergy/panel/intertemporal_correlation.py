import pandas as pd
import numpy as np
from typing import List
from macrosynergy.management.simulate.simulate_quantamental_data import make_qdf
from macrosynergy.management.utils import categories_df

"""
Implementation of correlation functions for quantamental data.
The functions calculates and visualize for two categories or cross-sections the following
  - Probabilities of Non-Stationarity
    - Augmented Dickey-Fuller (ADF) test
    - Phillips-Perron (PP) test
    - KPSS test
    - ADF-GLS test
    - Plot time trends, random walks and seasonalities
  - Autocorrelation function (ACF, PACF)
  - Cross-Correlation function (CCF)   
  - Granger Causality
  - Cross Recurrence Plots 
"""

# Step 1: Sample the data via "D", "W", "M", "Q", "A"
# Step 2: Calculate the probabilities of non-stationarity
# Step 3: Plot the time trends, random walks and seasonalities
# Step 4: Plot the ACF and PACF for statistically significant lags
# Step 5: Plot the CCF for two different signals for  statistically significant lags
# Step 6: Perform Granger Causality test


class IntertemporalCorrelation(object):
    def __init__(
        self,
        df: pd.DataFrame,
        xcats: List[str],
        cids: List[str],
        start: str = None,
        end: str = None,
        blacklist: dict = None,
        freq: str = None,
        lag: int = None,
        fwin: int = None,
        xcat_aggs: List[str] = None,
    ):
        self.df = categories_df(
            df,
            xcats=xcats,
            cids=cids,
            start=start,
            end=end,
            blacklist=blacklist,
            freq=freq,
            lag=lag,
            fwin=fwin,
            xcat_aggs=xcat_aggs,
        )

    def calc_probabilities_of_non_stationarity(
        self, df: pd.DataFrame, statistical_test: str
    ) -> pd.DataFrame:
        """
        Calculate the probabilities of non-stationarity
        """

        # ADF test HERE

        # Phillips-Perron (PP) test HERE

        # KPSS test HERE

        # ADF-GLS test HERE

        pass

    def plot_time_series_components(self):
        """
        Plot the time trends, random walks and seasonalities
        """
        pass

    def calculate_acf(self, lags: int):
        """
        Calculate the ACF and for statistically significant lags
        """
        pass

    def calculate_pacf(self, lags: int):
        """
        Calculate the PACF and for statistically significant lags
        """
        pass

    def plot_acf(self):
        """
        Plot the ACF and PACF for statistically significant lags
        """
        pass

    def calculate_ccf(self, lags: int):
        """
        Calculate the CCF for two different signals for statistically significant lags
        """
        pass

    def plot_ccf(self):
        """
        Plot the CCF for two different signals for statistically significant lags
        """
        pass

    def calculate_granger_causality(self):
        """
        Perform Granger Causality test
        """
        pass

    def cross_recurrence_plots(self):
        """
        Cross Recurrence Plots
        """
        pass


if __name__ == "__main__":
    cids = ["AUD", "CAD", "GBP", "NZD", "USD"]
    xcats = ["XR", "XRH", "CRY", "GROWTH", "INFL"]
    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2000-01-01", "2020-12-31", 0.1, 1]
    df_cids.loc["CAD"] = ["2001-01-01", "2020-11-30", 0, 1]
    df_cids.loc["BRL"] = ["2001-01-01", "2020-11-30", -0.1, 2]
    df_cids.loc["GBP"] = ["2002-01-01", "2020-11-30", 0, 2]
    df_cids.loc["NZD"] = ["2002-01-01", "2020-09-30", -0.1, 2]
    df_cids.loc["USD"] = ["2003-01-01", "2020-12-31", -0.1, 2]

    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]
    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["XRH"] = ["2000-01-01", "2020-12-31", 0.2, 1, 0, 0.25]
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-10-30", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2001-01-01", "2020-10-30", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {"AUD": ["2000-01-01", "2003-12-31"], "GBP": ["2018-01-01", "2100-01-01"]}

    itc = IntertemporalCorrelation(
        dfd,
        xcats=xcats,
        cids=cids,
        start="2000-01-01",
        end="2020-12-31",
        blacklist=black,
        freq="M",
    )

    print(itc.df.head())