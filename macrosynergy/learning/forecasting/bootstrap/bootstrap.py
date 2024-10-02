import numpy as np
import pandas as pd
import datetime

from collections import Counter, defaultdict
from abc import ABC

from typing import Union, Optional

class BasePanelBootstrap(ABC):
    def __init__(
        self,
        bootstrap_method: str = "panel",
        resample_ratio: Union[float, int] = 1,
        max_features: Optional[Union[str, int, float]] = None, # TODO
    ):
        """
        Base class to construct bootstrap datasets over a panel. 

        :param <str> bootstrap_method: Method to bootstrap the data. Current options are
            "panel", "period", "cross", "cross_per_period" and "period_per_cross".
            Default is "panel".
        :param <Union[float, int]> resample_ratio: The ratio of resampling units comprised
            in each bootstrap dataset. This is a fraction of the quantity of the panel
            component to be resampled. Default value is 1.
        :param <Optional[Union[str, int, float]]> max_features: The number of features to
            consider in each bootstrap dataset. This can be used to increase the 
            variation of the bootstrap datasets. Default is None and currently not
            implemented.
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
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
    ):
        """
        Method to generate a bootstrap dataset based on a panel of features and a target.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
            associated with each sample in X.

        :return Tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series]]: A tuple of the
            resampled feature matrix and target vector.
        """
        # Checks
        self._check_create_bootstrap_dataset_params(
            X,
            y,
        )

        # Store index information in numpy arrays
        index_array = np.array(X.index.tolist())
        cross_sections = index_array[:, 0]
        unique_cross_sections = np.unique(cross_sections)
        real_dates = index_array[:, 1]
        unique_real_dates = np.unique(real_dates)

        # Create a bootstrap dataset 
        if self.bootstrap_method == "panel":
            X_resampled, y_resampled = self._panel_bootstrap(
                X = X,
                y = y,
            )

        elif self.bootstrap_method == "period":
             X_resampled, y_resampled = self._period_bootstrap(
                X = X,
                y = y,
                unique_real_dates = unique_real_dates,
            )

        elif self.bootstrap_method == "cross":
            X_resampled, y_resampled = self._cross_bootstrap(
                X = X,
                y = y,
                unique_cross_sections = unique_cross_sections,
            )

        elif self.bootstrap_method == "cross_per_period":
            X_resampled, y_resampled = self._cross_per_period_bootstrap(
                X = X,
                y = y,
                unique_cross_sections = unique_cross_sections,
            )

        elif self.bootstrap_method == "period_per_cross":
            X_resampled, y_resampled = self._period_per_cross_bootstrap(
                X = X,
                y = y,
                unique_real_dates = unique_real_dates,
            )

        return X_resampled, y_resampled

    def _panel_bootstrap(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],      
    ):
        """
        Method to generate a bootstrap dataset by resampling the entire panel.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
            associated with each sample in X.

        :return Tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series]]: A tuple of the
            resampled feature matrix and target vector.
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
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        unique_real_dates: np.ndarray,
    ):
        """
        Method to generate a bootstrap dataset by resampling periods in the panel.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
            associated with each sample in X.
        :param <np.ndarray[pd.Timestamp]> unique_real_dates: Numpy array of the unique
            dates in the panel.

        :return Tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series]]: A tuple of the
            resampled feature matrix and target vector.
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
                np.tile(
                    y[y.index.get_level_values(1).isin(periods)].values, count
                ),
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
            data = X_resampled,
            index = index_resampled,
            columns = X.columns,
        )
        y_resampled = pd.Series(
            data = y_resampled,
            index = index_resampled,
            name = y.name,
        )

        return X_resampled, y_resampled
    
    def _cross_bootstrap(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        unique_cross_sections: np.ndarray,
    ):
        """
        Method to generate a bootstrap dataset by resampling cross-sections in the panel.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
            associated with each sample in X.
        :param <np.ndarray[pd.Timestamp]> unique_real_dates: Numpy array of the unique
            dates in the panel.

        :return Tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series]]: A tuple of the
            resampled feature matrix and target vector.
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
                X[X.index.get_level_values(0).isin(cross_sections)].index.tolist() * count
            )

        # reconstruct index
        index_resampled = pd.MultiIndex.from_tuples(
            index_resampled, names=["cid", "real_date"]
        )

        # Convert resampled datasets to pandas dataframes
        X_resampled = pd.DataFrame(
            data = X_resampled,
            index = index_resampled,
            columns = X.columns,
        )
        y_resampled = pd.Series(
            data = y_resampled,
            index = index_resampled,
            name = y.name,
        )

        return X_resampled, y_resampled
    
    def _cross_per_period_bootstrap(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        unique_cross_sections: np.ndarray,      
    ):
        """
        Method to generate a bootstrap dataset by resampling cross-sections within each
        period in the panel.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
            associated with each sample in X.
        :param <np.ndarray[str]> unique_cross_sections: Numpy array of the unique
            cross-sections in the panel.

        :return Tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series]]: A tuple of the
            resampled feature matrix and target vector.
        """
        n_resample = int(np.ceil(len(unique_cross_sections) * self.resample_ratio))
        X_resampled = X.groupby(level=1).sample(replace=True, n=n_resample)
        y_resampled = y.loc[X_resampled.index]

        return X_resampled, y_resampled
    
    def _period_per_cross_bootstrap(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        unique_real_dates: np.ndarray,      
    ):
        """
        Method to generate a bootstrap dataset by resampling periods within each
        cross-section in the panel.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
            associated with each sample in X.
        :param <np.ndarray[pd.Timestamp]> unique_real_dates: Numpy array of the unique
            dates in the panel.

        :return Tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series]]: A tuple of the
            resampled feature matrix and target vector.
        """
        n_resample = int(np.ceil(len(unique_real_dates) * self.resample_ratio))
        X_resampled = X.groupby(level=0).sample(replace=True, n=n_resample)
        y_resampled = y.loc[X_resampled.index]

        return X_resampled, y_resampled

    def _check_create_bootstrap_dataset_params(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],        
    ):
        """
        Method to check the validity of the input parameters for create_bootstrap_dataset.

        :param <pd.DataFrame> X: Pandas dataframe of input features.
        :param <Union[pd.DataFrame, pd.Series]> y: Pandas series or dataframe of targets
        """
        # Checks
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input feature matrix must be a pandas dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas dataframe."
            )
        if not (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)):
            raise TypeError(
                "Target vector must be a pandas series or dataframe. "
                "If used as part of an sklearn pipeline, ensure that previous steps "
                "return a pandas series or dataframe."
            )
        if isinstance(y, pd.DataFrame) and y.shape[1] != 1:
            raise ValueError(
                "The target dataframe must have only one column. If used as part of "
                "an sklearn pipeline, ensure that previous steps return a pandas "
                "series or dataframe."
            )

        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must be multi-indexed.")
        if not isinstance(y.index, pd.MultiIndex):
            raise ValueError("y must be multi-indexed.")
        if not isinstance(X.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of X must be datetime.date.")
        if not isinstance(y.index.get_level_values(1)[0], datetime.date):
            raise TypeError("The inner index of y must be datetime.date.")
        if not X.index.equals(y.index):
            raise ValueError(
                "The indices of the input dataframe X and the output dataframe y don't "
                "match."
            )
        
    def _check_boot_params(
        self,
        bootstrap_method: str,
        resample_ratio: Union[float, int],
        max_features: Optional[Union[str, int, float]],
    ):
        """
        Private method to check the class initialization parameters.

        :param <str> bootstrap_method: Method to bootstrap the data. Current options are
            "panel", "period", "cross", "cross_per_period" and "period_per_cross".
        :param <Union[float, int]> resample_ratio: The ratio of resampling units comprised
            in each bootstrap dataset. This is a fraction of the quantity of the panel
            component to be resampled.
        :param <Optional[Union[str, int, float]]> max_features: The number of features to
            consider in each bootstrap dataset. This can be used to increase the 
            variation of the bootstrap datasets.

        :return None
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