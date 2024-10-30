from macrosynergy.learning.preprocessing.scalers.base_panel_scaler import (
    BasePanelScaler,
)


class PanelMinMaxScaler(BasePanelScaler):
    """
    Scale and translate panel features to lie within the range [0,1].

    Notes
    -----
    This class is designed to replicate scikit-learn's `MinMaxScaler()` class, with the
    additional option to scale within cross-sections. Unlike the `MinMaxScaler()` class,
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

    def __init__(self, type="panel", with_mean=True, with_std=True):
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


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    from macrosynergy.management import (
        categories_df,
        make_qdf,
    )

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    """Example: Unbalanced panel """

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2000-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2000-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2000-01-01", "2020-12-31", -0.1, 2, 0.8, 0.3]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {
        "GBP": (
            pd.Timestamp(year=2009, month=1, day=1),
            pd.Timestamp(year=2012, month=6, day=30),
        ),
        "CAD": (
            pd.Timestamp(year=2015, month=1, day=1),
            pd.Timestamp(year=2100, month=1, day=1),
        ),
    }

    train = categories_df(
        df=dfd, xcats=xcats, cids=cids, val="value", blacklist=black, freq="M", lag=1
    ).dropna()
    train = train[
        train.index.get_level_values(1) >= pd.Timestamp(year=2005, month=8, day=1)
    ]

    X_train = train.drop(columns=["XR"])
    y_train = train["XR"]

    # Standard scaling over each cross-section
    scaler = PanelStandardScaler(type="cross_section")
    scaler.fit(X_train, y_train)
    print(scaler.transform(X_train))

    # MinMax scaling over the panel
    scaler = PanelMinMaxScaler()
    scaler.fit(X_train, y_train)
    print(scaler.transform(X_train))
