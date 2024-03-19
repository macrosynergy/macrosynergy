import pandas as pd
import numpy as np
import typing

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
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def sample(self, freq: str = "M") -> pd.DataFrame:
        """
        Sample the data via "D", "W", "M", "Q", "A"
        """
        return self.df.resample(freq).mean()
    
    def calc_probabilities_of_non_stationarity(self, df: pd.DataFrame, statistical_test: str) -> pd.DataFrame:
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
