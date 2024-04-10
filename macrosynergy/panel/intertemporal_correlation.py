import pandas as pd
import numpy as np
from typing import List
from macrosynergy.management.simulate.simulate_quantamental_data import make_qdf
from macrosynergy.management.utils import categories_df
from arch.unitroot import PhillipsPerron, ADF, DFGLS, KPSS, ZivotAndrews
from statsmodels.tsa.stattools import pacf

"""
Implementation of correlation functions for quantamental data.
The functions calculates and visualize for two categories or cross-sections the following
  - Probabilities of Non-Stationarity
    - Augmented Dickey-Fuller (ADF) test
    - Phillips-Perron (PP) test
    - KPSS test
    - ADF-GLS test
    - Zivot-Andrews test
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
        lag: int = 0,
        fwin: int = 1,
        xcat_aggs: List[str] = ["mean", "mean"],
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
        self.xcats = xcats
        self.cids = cids

    def calc_prob_of_non_stationarity(
        self,
        statistical_tests: List[str] = None,
        lags: int = None,
        trend="c",
        max_lags=None,
    ) -> pd.DataFrame:
        """
        Calculate the probabilities of non-stationarity
        """
        stat_test_dict = {
            "ADF": self._calc_adf,
            "PP": self._calc_pp,
            "DFGLS": self._calc_dfgls,
            "KPSS": self._calc_kpss,
            "ZA": self._calc_za,
        }

        if statistical_tests is None:
            statistical_tests = ["ADF", "PP", "DFGLS", "KPSS", "ZA"]
        else:
            if not set(statistical_tests).issubset(
                {"ADF", "PP", "DFGLS", "KPSS", "ZA"}
            ):
                raise ValueError(
                    "statistical_tests must be a subset of {'ADF', 'PP', 'DFGLS', 'KPSS', 'ZA'}"
                )

        index = pd.MultiIndex.from_product(
            [self.cids, self.xcats], names=["cid", "xcat"]
        )
        df_result = pd.DataFrame(index=index)

        for idx in index:
            cid = idx[0]
            xcat = idx[1]
            for test in statistical_tests:
                df_result.loc[idx, f"{test} P Value"] = stat_test_dict[test](
                    xcat, cid, lags, trend, max_lags
                )

        return df_result

    def _calc_adf(self, xcat, cid, lags, trend, max_lags, *args, **kwargs):
        adf_result = ADF(
            self.df[xcat][cid].dropna().values,
            lags=lags,
            trend=trend,
            max_lags=max_lags,
            method="aic",
        )
        return "{:.6f}".format(adf_result.pvalue)

    def _calc_pp(self, xcat, cid, lags, trend, *args, **kwargs):
        PP_result = PhillipsPerron(
            self.df[xcat][cid].dropna().values, lags=lags, trend=trend
        )
        return "{:.6f}".format(PP_result.pvalue)

    def _calc_dfgls(self, xcat, cid, lags, trend, max_lags, *args, **kwargs):
        dfgls_result = DFGLS(
            self.df[xcat][cid].dropna().values,
            lags=lags,
            max_lags=max_lags,
            trend=trend,
            method="aic",
        )
        return "{:.6f}".format(dfgls_result.pvalue)

    def _calc_kpss(self, xcat, cid, lags, trend, *args, **kwargs):
        kpss_result = KPSS(self.df[xcat][cid].dropna().values, lags=lags, trend=trend)
        return "{:.6f}".format(kpss_result.pvalue)

    def _calc_za(self, xcat, cid, lags, trend, max_lags, *args, **kwargs):
        za_result = ZivotAndrews(
            self.df[xcat][cid].dropna().values,
            lags=lags,
            trend=trend,
            max_lags=max_lags,
            method="aic",
        )
        return "{:.6f}".format(za_result.pvalue)

    # def plot_time_series_components(self):
    #     """
    #     Plot the time trends, random walks and seasonalities
    #     """
    #     pass

    def calculate_acf(self, xcat: str, cid: str, max_lag: int, detrend: bool = True):
        """
        Calculate the ACF and for statistically significant lags
        """
        time_series = self.df[xcat][cid].dropna()
        
        if detrend:
            time_series = time_series - time_series.mean()
        
        autocorr_values = []
        
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorr_values.append(1.0)
            else:
                lag_correlation = np.corrcoef(time_series[:-lag], time_series[lag:])[0, 1]
                autocorr_values.append(lag_correlation)
                
        return pd.Series(autocorr_values, index=np.arange(max_lag + 1))

    def calculate_pacf(self, xcat: str, cid: str, max_lag: int):
        """
        Calculate the PACF and for statistically significant lags
        """
        pacf_values = pacf(self.df[xcat][cid].dropna(), nlags=max_lag)
    
        return pacf_values

    def plot_acf(self):
        """
        Plot the ACF and PACF for statistically significant lags
        """
        pacf_values = self.calculate_pacf("FXXR_NSA", "AUD", )

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
    from macrosynergy.download import JPMaQSDownload
    import os

    # ids_dmlc = ["EUR", "JPY", "MXN", "TRY", "INR"]  # DM large currency areas
    cids_dmsc = [
        "AUD",
        "CAD",
        "CHF",
        "GBP",
        "NOK",
        "NZD",
        "SEK",
    ]  # DM small currency areas

    cids = cids_dmsc
    rets = [
        "FXXR_NSA",
        "FXXR_VT10",
        # "FXXRHvGDRB_NSA",
        # "MTBGDPRATIO_SA_3MMAv120MMA"
    ]

    xcats = rets

    # Resultant tickers

    tickers = [cid + "_" + xcat for cid in cids for xcat in xcats]
    print(f"Maximum number of tickers is {len(tickers)}")

    start_date = "2000-01-01"
    end_date = "2023-05-01"

    # Retrieve credentials

    client_id: str = os.getenv("DQ_CLIENT_ID")
    client_secret: str = os.getenv("DQ_CLIENT_SECRET")

    with JPMaQSDownload(client_id=client_id, client_secret=client_secret) as dq:
        df = dq.download(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            suppress_warning=True,
            metrics=["value"],
            report_time_taken=True,
            show_progress=True,
        )

    itc = IntertemporalCorrelation(
        df,
        xcats=xcats,
        cids=cids,
        start="2000-01-01",
        end="2023-01-01",
        # blacklist=black,
        freq="M",
    )

    print(itc.df.head())
    print(itc.calc_prob_of_non_stationarity(trend="c"))
    print(itc.calc_prob_of_non_stationarity(trend="ct"))

    print(itc.calculate_acf("FXXR_NSA", "AUD", 10))  # ACF for AUD
    print(itc.calculate_pacf("FXXR_NSA", "AUD", 10))  # PACF for AUD
