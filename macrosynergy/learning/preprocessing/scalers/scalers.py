import pandas as pd

from base_panel_scaler import BasePanelScaler

from typing import Any, Union

class PanelMinMaxScaler(BasePanelScaler):
    """
    Scale and translate panel features to lie within the range [0,1].

    Notes
    -----
    This class is designed to replicate scikit-learn's MinMaxScaler() class, with the
    additional option to scale within cross-sections. Unlike the MinMaxScaler() class,
    dataframes are always returned, preserving the multi-indexing of the inputs.
    """

    def extract_statistics(self, X, feature):
        """
        Determine the minimum and maximum values of a feature in the input matrix.

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        feature : str
            The feature to extract statistics for.

        Returns
        -------
        statistics : list 
            List containing the minimum and maximum values of the feature.
        """
        return [X[feature].min(), X[feature].max()]
    
    def scale(self, X, feature, statistics):
        """
        Scale the 'feature' column in the design matrix 'X' based on the minimum and
        maximum values of the feature.

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        feature : str
            The feature to scale.
        statistics : list
            List containing the minimum and maximum values of the feature, in that order. 

        Returns
        -------
        X_transformed : pandas.Series
            The scaled feature.
        """
        return (X[feature] - statistics[0]) / (statistics[1] - statistics[0])

class PanelStandardScaler(BasePanelScaler):
    """
    Scale and translate panel features to have zero mean and unit variance.

    Parameters
    ----------
    type : str, default="panel"
        The panel dimension over which the scaling is applied. Options are 
        "panel" and "cross_section".
    with_mean : bool, default=True
        Whether to centre the data before scaling.
    with_std : bool, default=True
        Whether to scale the data to unit variance.

    Notes
    -----
    This class is designed to replicate scikit-learn's StandardScaler() class, with the
    additional option to scale within cross-sections. Unlike the StandardScaler() class,
    dataframes are always returned, preserving the multi-indexing of the inputs.
    """
    def __init__(self, type = "panel", with_mean = True, with_std = True):
        # Checks
        if not isinstance(with_mean, bool):
            raise TypeError("'with_mean' must be a boolean.")
        if not isinstance(with_std, bool):
            raise TypeError("'with_std' must be a boolean.")

        # Attributes
        self.with_mean = with_mean
        self.with_std = with_std

        super().__init__(type=type)

    def extract_statistics(self, X, feature):
        """
        Determine the mean and standard deviation of values of a feature in the input
        matrix.

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        feature : str
            The feature to extract statistics for.

        Returns
        -------
        statistics : list 
            List containing the mean and standard deviation of values of the feature.
        """
        return [X[feature].mean(), X[feature].std()]
    
    def scale(self, X, feature, statistics):
        """
        Scale the 'feature' column in the design matrix 'X' based on the mean and
        standard deviation values of the feature.

        Parameters
        ----------
        X : pandas.DataFrame
            The feature matrix.
        feature : str
            The feature to scale.
        statistics : list
            List containing the mean and standard deviation of values of the feature,
            in that order. 

        Returns
        -------
        X_transformed : pandas.Series
            The scaled feature.
        """
        if self.with_mean:
            if self.with_std:
                return (X[feature] - statistics[0]) / statistics[1]
            else:
                return X[feature] - statistics[0]
        else:
            if self.with_std:
                return X[feature] / statistics[1]
            else:
                return X[feature]