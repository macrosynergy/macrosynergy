import numpy as np
import pandas as pd

from collections import Counter, defaultdict
from abc import ABC


class BasePanelBootstrap(ABC):
    def __init__(
        self,
        bootstrap_method="panel",
        resample_ratio=1,
        max_features=None,
    ):
        """
        Construct bootstrap datasets over a panel.

        Parameters
        ----------
        bootstrap_method : str
            Method to bootstrap the data. Current options are "panel",
            "period", "cross", "cross_per_period" and "period_per_cross".
            Default is "panel".
        resample_ratio : numbers.Number
            Ratio of resampling units comprised in each bootstrap dataset.
            This is a fraction of the quantity of the panel component to be
            resampled. Default value is 1.
        max_features: str or numbers.Number, optional
            The number of features to consider in each bootstrap dataset.
            This can be used to increase the variation between bootstrap datasets.
            Default is None and currently not implemented.

        Notes
        -----
        The non-parametric bootstrap is a method to generate datasets that follow the same
        distribution as the original dataset, as best as possible given the observed data.
        Mathematically, a bootstrap dataset is equivalent to sampling from the empirical
        distribution of the original dataset. A bootstrap dataset is constructed by
        sampling the observed data with replacement.

        Bootstrapping can be used to estimate the distribution of a statistic, for
        instance the sampling distribution of a model parameter. As an example, a
        regression model can be fit to each bootstrap dataset to estimate the distribution
        of the model parameters. The standard deviation of these distributions,
        consequently, can be used to estimate the sampling variation of the model
        parameters.

        Bootstrapping can also be used to create "bagged" models. Bagging is a method
        aimed at reducing the variance of a machine learning model. It involves training
        multiple models on different bootstrap datasets and averaging the predictions
        of these models. Often, additional variation is also introduced to each bootstrap
        dataset - for instance by randomly sampling a subset of features for each dataset.
        """
        # Checks
        self._check_boot_params(
            bootstrap_method=bootstrap_method,
            resample_ratio=resample_ratio,
            max_features=max_features,
        )

        # Set attributes
        self.bootstrap_method = bootstrap_method
        self.resample_ratio = resample_ratio
        self.max_features = max_features

    def create_bootstrap_dataset(
        self,
        X,
        y,
    ):
        """
        Generate a bootstrap dataset based on a panel of features and a
        dependent variable.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix
        y : pd.DataFrame or pd.Series
            Dependent variable.

        Returns
        -------
        X_resampled : pd.DataFrame
            Bootstrap resampled feature matrix.
        y_resampled : pd.DataFrame or pd.Series
            Bootstrap resampled dependent variable.
        """
        # Store index information in numpy arrays
        index_array = np.array(X.index.tolist())
        cross_sections = index_array[:, 0]
        unique_cross_sections = np.unique(cross_sections)
        real_dates = index_array[:, 1]
        unique_real_dates = np.unique(real_dates)

        # Create a bootstrap dataset
        if self.bootstrap_method == "panel":
            X_resampled, y_resampled = self._panel_bootstrap(
                X=X,
                y=y,
            )

        elif self.bootstrap_method == "period":
            X_resampled, y_resampled = self._period_bootstrap(
                X=X,
                y=y,
                unique_real_dates=unique_real_dates,
            )

        elif self.bootstrap_method == "cross":
            X_resampled, y_resampled = self._cross_bootstrap(
                X=X,
                y=y,
                unique_cross_sections=unique_cross_sections,
            )

        elif self.bootstrap_method == "cross_per_period":
            X_resampled, y_resampled = self._cross_per_period_bootstrap(
                X=X,
                y=y,
                unique_cross_sections=unique_cross_sections,
            )

        elif self.bootstrap_method == "period_per_cross":
            X_resampled, y_resampled = self._period_per_cross_bootstrap(
                X=X,
                y=y,
                unique_real_dates=unique_real_dates,
            )

        return X_resampled, y_resampled

    def _panel_bootstrap(
        self,
        X,
        y,
    ):
        """
        Generate a bootstrap dataset by resampling the panel.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.
        y : pd.DataFrame or pd.Series
            Dependent variable.

        Returns
        -------
        X_resampled : pd.DataFrame
            Bootstrap resampled feature matrix.
        y_resampled : pd.DataFrame or pd.Series
            Bootstrap resampled dependent variable.
        """
        bootstrap_idx = np.random.choice(
            np.arange(X.shape[0]),
            size=int(np.ceil(self.resample_ratio * X.shape[0])),
            replace=True,
        )
        X_resampled = X.iloc[bootstrap_idx]
        y_resampled = y.iloc[bootstrap_idx]

        return X_resampled, y_resampled

    def _period_bootstrap(
        self,
        X,
        y,
        unique_real_dates,
    ):
        """
        Generate a bootstrap dataset by resampling periods in the panel.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.
        y : pd.DataFrame or pd.Series
            Dependent variable.
        unique_real_dates : np.ndarray of pd.Timestamp
            Unique dates in the panel.

        Returns
        -------
        X_resampled : pd.DataFrame
            Bootstrap resampled feature matrix.
        y_resampled : pd.DataFrame or pd.Series
            Bootstrap resampled dependent variable.
        """
        # Resample unique panel dates with replacement
        bootstrap_periods = np.random.choice(
            unique_real_dates,
            size=int(np.ceil(self.resample_ratio * len(unique_real_dates))),
            replace=True,
        )

        # Obtain a {count: [periods]} dictionary so that the new panel can be efficiently
        # constructed by looping through the counts instead of the periods.
        period_counts = dict(Counter(bootstrap_periods))
        count_to_periods = defaultdict(list)
        for period, count in period_counts.items():
            count_to_periods[count].append(period)

        # For each count, extract the periods that have that count and tile then by count
        # to create the new panel.
        X_resampled = np.empty((0, X.shape[1]))
        y_resampled = np.empty(0)
        index_resampled = []
        for count, periods in count_to_periods.items():
            X_resampled = np.vstack(
                [
                    X_resampled,
                    np.tile(
                        X[X.index.get_level_values(1).isin(periods)].values,
                        (count, 1),
                    ),
                ]
            )
            y_resampled = np.append(
                y_resampled,
                np.tile(y[y.index.get_level_values(1).isin(periods)].values, count),
            )
            index_resampled.extend(
                X[X.index.get_level_values(1).isin(periods)].index.tolist() * count
            )

        # reconstruct index
        index_resampled = pd.MultiIndex.from_tuples(
            index_resampled, names=["cid", "real_date"]
        )

        # Convert resampled datasets to pandas dataframes
        X_resampled = pd.DataFrame(
            data=X_resampled,
            index=index_resampled,
            columns=X.columns,
        )
        y_resampled = pd.Series(
            data=y_resampled,
            index=index_resampled,
            name=y.name,
        )

        return X_resampled, y_resampled

    def _cross_bootstrap(
        self,
        X,
        y,
        unique_cross_sections,
    ):
        """
        Generate a bootstrap dataset by resampling cross-sections in the panel.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.
        y : pd.DataFrame or pd.Series
            Dependent variable.
        unique_cross_sections : np.ndarray of str
            Unique cross-sections in the panel.

        Returns
        -------
        X_resampled : pd.DataFrame
            Bootstrap resampled feature matrix.
        y_resampled : pd.DataFrame or pd.Series
            Bootstrap resampled dependent variable.
        """
        # Resample unique panel cross-sections with replacement
        bootstrap_cross_sections = np.random.choice(
            unique_cross_sections,
            size=int(np.ceil(len(unique_cross_sections) * self.resample_ratio)),
            replace=True,
        )

        # Obtain a {count: [cross_sections]} dictionary so that the new panel can be efficiently
        # constructed by looping through the counts instead of the periods.
        cross_section_counts = dict(Counter(bootstrap_cross_sections))
        count_to_cross_sections = defaultdict(list)
        for cross_section, count in cross_section_counts.items():
            count_to_cross_sections[count].append(cross_section)

        # For each count, tile the observations within the cross-sections with that count
        X_resampled = np.empty((0, X.shape[1]))
        y_resampled = np.empty(0)
        index_resampled = []
        for count, cross_sections in count_to_cross_sections.items():
            X_resampled = np.vstack(
                [
                    X_resampled,
                    np.tile(
                        X[X.index.get_level_values(0).isin(cross_sections)].values,
                        (count, 1),
                    ),
                ]
            )
            y_resampled = np.append(
                y_resampled,
                np.tile(
                    y[y.index.get_level_values(0).isin(cross_sections)].values, count
                ),
            )
            index_resampled.extend(
                X[X.index.get_level_values(0).isin(cross_sections)].index.tolist()
                * count
            )

        # reconstruct index
        index_resampled = pd.MultiIndex.from_tuples(
            index_resampled, names=["cid", "real_date"]
        )

        # Convert resampled datasets to pandas dataframes
        X_resampled = pd.DataFrame(
            data=X_resampled,
            index=index_resampled,
            columns=X.columns,
        )
        y_resampled = pd.Series(
            data=y_resampled,
            index=index_resampled,
            name=y.name,
        )

        return X_resampled, y_resampled

    def _cross_per_period_bootstrap(
        self,
        X,
        y,
        unique_cross_sections,
    ):
        """
        Generate a bootstrap dataset by resampling cross-sections within each
        period in the panel.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.
        y : pd.DataFrame or pd.Series
            Dependent variable.
        unique_cross_sections : np.ndarray of str
            Unique cross-sections in the panel.

        Returns
        -------
        X_resampled : pd.DataFrame
            Bootstrap resampled feature matrix.
        y_resampled : pd.DataFrame or pd.Series
            Bootstrap resampled dependent variable.
        """
        n_resample = int(np.ceil(len(unique_cross_sections) * self.resample_ratio))
        X_resampled = X.groupby(level=1).sample(replace=True, n=n_resample)
        y_resampled = y.loc[X_resampled.index]

        return X_resampled, y_resampled

    def _period_per_cross_bootstrap(
        self,
        X,
        y,
        unique_real_dates,
    ):
        """
        Generate a bootstrap dataset by resampling periods within each
        cross-section in the panel.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.
        y : pd.DataFrame or pd.Series
            Dependent variable.
        unique_real_dates : np.ndarray of pd.Timestamp
            Unique dates in the panel.

        Returns
        -------
        X_resampled : pd.DataFrame
            Bootstrap resampled feature matrix.
        y_resampled : pd.DataFrame or pd.Series
            Bootstrap resampled dependent variable.
        """
        n_resample = int(np.ceil(len(unique_real_dates) * self.resample_ratio))
        X_resampled = X.groupby(level=0).sample(replace=True, n=n_resample)
        y_resampled = y.loc[X_resampled.index]

        return X_resampled, y_resampled

    def _check_boot_params(
        self,
        bootstrap_method,
        resample_ratio,
        max_features,
    ):
        """
        Bootstrap class initialization checks.

        Parameters
        ----------
        bootstrap_method : str
            Method to bootstrap the data. Current options are "panel",
            "period", "cross", "cross_per_period" and "period_per_cross".
            Default is "panel".
        resample_ratio : numbers.Number
            Ratio of resampling units comprised in each bootstrap dataset.
            This is a fraction of the quantity of the panel component to be
            resampled. Default value is 1.
        max_features: str or numbers.Number, optional
            The number of features to consider in each bootstrap dataset.
            This can be used to increase the variation between bootstrap datasets.
            Default is None and currently not implemented.
        """
        # bootstrap_method
        if not isinstance(bootstrap_method, str):
            raise TypeError(
                f"bootstrap_method must be a string. Got {type(bootstrap_method)}."
            )
        if bootstrap_method not in [
            "panel",
            "period",
            "cross",
            "cross_per_period",
            "period_per_cross",
        ]:
            raise ValueError(
                f"bootstrap_method must be one of 'panel', 'period', 'cross',"
                " 'cross_per_period', 'period_per_cross'. Got {bootstrap_method}."
            )

        # resample_ratio
        if not isinstance(resample_ratio, (int, float)):
            raise TypeError(
                f"resample_ratio must be an integer or a float. Got {type(resample_ratio)}."
            )
        if resample_ratio <= 0:
            raise ValueError("resample_ratio must be greater than 0.")
        if resample_ratio > 1:
            raise ValueError("resample_ratio must be less than or equal to 1.")

        # max_features
        if max_features is not None:
            raise NotImplementedError("max_features is not implemented yet.")


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_qdf
    import macrosynergy.management as msm

    # Simulate an unbalanced panel, multiindexed by cross-section and date
    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2000-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0, 1, 0, 3]
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-12-31", 0, 1, 0, 0]
    df_xcats.loc["GROWTH"] = ["2000-01-01", "2020-12-31", 0, 1, -0.9, 0]
    df_xcats.loc["INFL"] = ["2000-01-01", "2020-12-31", 0, 1, 0.8, 0]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {"GBP": ["2009-01-01", "2012-06-30"], "CAD": ["2018-01-01", "2100-01-01"]}
    dfd = msm.reduce_df(df=dfd, cids=cids, xcats=xcats, blacklist=black)

    dfd = dfd.pivot(index=["cid", "real_date"], columns="xcat", values="value")
    X = dfd.drop(columns=["XR"])
    y = dfd["XR"]

    bootstrap_methods = [
        "panel",
        "period",
        "cross",
        "cross_per_period",
        "period_per_cross",
    ]

    for method in bootstrap_methods:
        # Initialize the BasePanelBootstrap class
        bpb = BasePanelBootstrap(
            bootstrap_method=method,
            resample_ratio=0.8,
        )
        print(bpb.create_bootstrap_dataset(X, y))
