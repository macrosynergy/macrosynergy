"""
Class to calculate sequentially-optimized forecasts based on panels of features. 
TODO: find a nicer way of writing this docstring
"""

import numpy as np
import pandas as pd

from macrosynergy.learning import ExpandingIncrementPanelSplit
from .base_panel_learner import BasePanelLearner

class SignalOptimizer(BasePanelLearner):
    """
    Class for sequential optimization of return forecasts based on panels of quantamental
    features.

    Parameters
    ----------
    TODO 

    Notes
    -----
    TODO
    """
    def __init__(
        self,
        df,
        xcats,
        cids = None,
        start = None,
        end = None,
        blacklist = None,
        freq = "M",
        lag = 1,
        xcat_aggs = ["last", "sum"],
    ):
        super().__init__(
            df = df,
            xcats = xcats,
            cids = cids,
            start = start,
            end = end,
            blacklist = blacklist,
            freq = freq,
            lag = lag,
            xcat_aggs = xcat_aggs,
        )

        # Create forecast dataframe index 
        min_date = min(self.unique_date_levels)
        max_date = max(self.unique_date_levels)
        forecast_date_levels = pd.date_range(start=min_date, end=max_date, freq="B")
        self.forecast_idxs = pd.MultiIndex.from_product(
            [self.unique_xs_levels, forecast_date_levels], names=["cid", "real_date"]
        )

        # Create initial dataframes to store relevant quantities from the learning process
        self.preds = pd.DataFrame(columns=["cid", "real_date", "xcat", "value"])
        self.ftr_coefficients = pd.DataFrame(
            columns=["real_date", "name"] + list(self.X.columns)
        )
        self.intercepts = pd.DataFrame(columns=["real_date", "name", "intercepts"]) # TODO: look at merging into ftr_coefficients
        self.selected_ftrs = pd.DataFrame(
            columns=["real_date", "name"] + list(self.X.columns)
        )

    def calculate_predictions(
        self,
        name,
        models,
        hyperparameters,
        scorers, # TODO: make optional if no hyperparameters present. Then can skip hparam search
        inner_splitters,
        search_type = "grid",
        cv_summary = "mean",
        min_cids = 4,
        min_periods = 12 * 3,
        test_size = 1,
        max_periods = None,
        split_dictionary = None,
        n_iter = None,
        n_jobs_outer = -1,
        n_jobs_inner = 1,
    ):
        """
        Determine forecasts and store relevant quantities over time. 
        """
        # First create pandas series to store the forecasts
        forecasts_df = pd.DataFrame(
            index = self.forecast_idxs,
            columns = [name],
            data = np.nan,
            dtype = "float32"
        )

        # Set up outer splitter
        outer_splitter = ExpandingIncrementPanelSplit(
            train_intervals = test_size,
            test_size = test_size,
            min_cids = min_cids,
            min_periods = min_periods,
            max_periods = max_periods,
        )

        results = self.run(
            name = name,
            outer_splitter = outer_splitter,
            inner_splitters = inner_splitters,
            models = models,
            hyperparameters = hyperparameters,
            scorers = scorers,
            search_type = search_type,
            cv_summary = cv_summary,
            split_dictionary = split_dictionary,
            n_iter = n_iter,
            n_jobs_outer = n_jobs_outer,
            n_jobs_inner = n_jobs_inner, 
        )

        # Collect results from the worker
        # quantamental_data, model_data, other_data
        prediction_data = []
        modelchoice_data = []
        ftrcoeff_data = []
        intercept_data = []
        ftr_selection_data = []

        for quantamental_data, model_data, other_data in results:
            prediction_data.append(quantamental_data["predictions"])
            modelchoice_data.append(model_data["model_choice"])
            ftrcoeff_data.append(other_data["ftr_coefficients"])
            intercept_data.append(other_data["intercepts"])
            ftr_selection_data.append(other_data["selected_ftrs"])

        # Create quantamental dataframe of forecasts
        for idx, forecasts in prediction_data:
            forecasts_df.loc[idx, name] = forecasts

        forecasts_df = forecasts_df.groupby(level = 0).ffill()

        if self.blacklist is not None:
            for cross_section, periods in self.blacklist.items():
                cross_section_key = cross_section.split("_")[0]
                if cross_section_key in self.unique_xs_levels:
                    forecasts_df.loc[
                        (cross_section_key, slice(periods[0], periods[1])), :
                    ] = np.nan

        forecasts_df_long = pd.melt(
            frame = forecasts_df.reset_index(),
            id_vars = ["cid", "real_date"],
            value_vars = "xcat",
        )
        self.preds = pd.concat((self.preds, forecasts_df_long), axis=0).astype(
            {
                "cid": "object",
                "real_date": "datetime64[ns]",
                "xcat": "object",
                "value": np.float32,
            }
        )

        # Store model selection data
        model_df_long = pd.DataFrame(
            columns = self.chosen_models.columns,
            data = modelchoice_data,
        )
        self.chosen_models = pd.concat(
            (
                self.chosen_models,
                model_df_long,
            ),
            axis=0,
        ).astype(
            {
                "real_date": "datetime64[ns]",
                "name": "object",
                "model_type": "object",
                "hparams": "object",
                "n_splits_used": "int",
            }
        )

        # Store feature coefficients
        coef_df_long = pd.DataFrame(
            columns=self.ftr_coefficients.columns, data=ftrcoeff_data
        )
        ftr_coef_types = {col: "float32" for col in self.X.columns}
        ftr_coef_types["real_date"] = "datetime64[ns]"
        ftr_coef_types["name"] = "object"
        self.ftr_coefficients = pd.concat(
            (
                self.ftr_coefficients,
                coef_df_long,
            ),
            axis=0,
        ).astype(ftr_coef_types)

        # Store intercept
        intercept_df_long = pd.DataFrame(
            columns=self.intercepts.columns, data=intercept_data
        )
        self.intercepts = pd.concat(
            (
                self.intercepts,
                intercept_df_long,
            ),
            axis=0,
        ).astype(
            {
                "real_date": "datetime64[ns]",
                "name": "object",
                "intercepts": "float32",
            }
        )

        # Store selected features
        ftr_select_df_long = pd.DataFrame(
            columns=self.selected_ftrs.columns, data=ftr_selection_data
        )
        ftr_selection_types = {col: "int" for col in self.X.columns}
        ftr_selection_types["real_date"] = "datetime64[ns]"
        ftr_selection_types["name"] = "object"

        self.selected_ftrs = pd.concat(
            (
                self.selected_ftrs,
                ftr_select_df_long,
            ),
            axis=0,
        ).astype(ftr_selection_types)

    

