from typing import Dict, List

import pandas as pd


class ScoreVisualizers:
    """
    Class for displaying heatmaps of normalized quantamental categories, including a weighted composite

    :param <pd.DataFrame> df: A DataFrame with the following columns:
        'cid', 'xcat', 'real_date', and at least one metric from -
        'value', 'grading', 'eop_lag', or 'mop_lag'.

    :param <List[str]> cids: A list of cids to select from the DataFrame.
        If None, all cids are selected.

    :param <List[str]> xcats: A list of xcats to select from the DataFrame.
        These should be normalized indicators.

    :param <Dict[str, str]> xcat_labels: Optional list of custom labels for xcats
        Default is None.

    :param <str> xcat_comp: Label for composite score, if one is to be calculated and used. Default is “Composite”/
        Calculations are done based for the linear_composite function

    :param <Dict[str, str]> weights: A list of weights as large as the xcats list for calculating the composite.
        Default is equal weights.
        [we need passthrough arguments for linear_composite]
"""
    def __init__(self, df: pd.Dataframe, cids: List[str] = None, xcats: List[str] = None, xcat_labels: Dict[str, str]=None, xcat_comp: str ="Composite", weights: Dict[str, str] = None):
        self.df = df
        self.cids = cids
        self.xcats = xcats
        self.xcat_labels = xcat_labels
        self.xcat_comp = xcat_comp
        self.weights = weights


    def view_snapshot(cids: List[str], xcats: List[str], transpose: bool = False, start: str = None):
        """
        Display a multiple scores for multiple countries for the latest available or any previous date

        Parameters

        :param <List[str]> cids: A list of cids whose values are displayed. Default is all in the class

        :param <List[str]> xcats: A list of xcats to be displayed in the given order. Default is all in the class, including the composite, with the latter being the first row (or column).

        :param <bool> transpose: If False (default) rows are cids and columns are xcats. If True rows are xcats and columns are cids.

        : param <str> start: ISO-8601 formatted date string giving the date (or nearest previous if not available). Default is latest day in the dataframe,
        """
        pass
    

    def view_score_evolution(cids: List[str], xcats: List[str], freq: str, include_latest_period: bool = True, include_latest_day: bool = True, start: str = None, transpose: bool = False):
        """
        :param <List[str]> cids: A list of cids whose values are displayed. Default is all in the class

        :param <str> xcat: Single xcat to be displayed. Default is xcat_comp.

        :param<str> freq: frequency to which values are aggregated, i.e. averaged. Default is annual (A). The alternative is quarterly (Q) or bi-annnual (BA)

        :param <bool> include_latest_period: include the latest period average as defined by freq, even if it is not complete. Default is True.

        :param <bool> include_latest_day: include the latest working day date as defined by freq, even if it is not complete. Default is True.

        :param <str> start: ISO-8601 formatted date string. Select data from
            this date onwards. If None, all dates are selected.

        :param <bool> transpose: If False (default) rows are cids and columns are time periods. If True rows are time periods and columns are cids.
        """

        pass


    def view_cid_evolution(cid: str, xcats: List[str], freq: str, include_latest_period: bool = True, include_latest_day: bool = True, start: str = None, transpose: bool = False):
        """
        :param <str> cid: Single cid to be displayed

        :param <List[str]> xcats: A list of xcats to be displayed in the given order. Default is all in the class, including the composite, with the latter being the first row (or column).

        :param<str> freq: frequency to which values are aggregated, i.e. averaged. Default is annual (A). The alternative is quarterly (Q) or bi-annnual (BA)

        :param <bool> include_latest_period: include the latest period average as defined by freq, even if it is not complete. Default is True.

        :param <bool> include_latest_day: include the latest working day date as defined by freq, even if it is not complete. Default is True.

        :param <str> start: ISO-8601 formatted date string. Select data from
            this date onwards. If None, all dates are selected.

        :param <bool> transpose: If False (default) rows are xcats and columns are time periods. If True rows are time periods and columns are xcats.
        """
        pass