"""
Class to estimate market betas and calculate out-of-sample hedged returns based on
sequential learning. 
"""

import numpy as np
import pandas as pd

from macrosynergy.management import categories_df, reduce_df, update_df
from macrosynergy.learning import ExpandingFrequencyPanelSplit

from .base_panel_learner import BasePanelLearner

class BetaEstimator(BasePanelLearner):
    """
    Class for sequential beta estimation by learning optimal regression coefficients.

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
        benchmark_return,
        cids = None,
        start = None,
        end = None, 
    ):
        # Checks 
        # TODO

        # Create pseudo-panel
        dfx = pd.DataFrame(columns=["real_date", "cid", "xcat", "value"])
        benchmark_cid, benchmark_xcat = benchmark_return.split("_", 1)

        for cid in cids:
            # Extract cross-section contract returns
            dfa = reduce_df(
                df=df,
                xcats=xcats if isinstance(xcats, list) else [xcats],
                cids=[cid],
            )
            # Extract benchmark returns
            dfb = reduce_df(
                df=df,
                xcats=[benchmark_xcat],
                cids=[benchmark_cid],
            )

            # Combine contract and benchmark returns and rename cross-section identifier
            # in order to match the benchmark return with each cross section in a pseudo
            # panel
            df_cid = pd.concat([dfa, dfb], axis=0)
            df_cid["cid"] = f"{cid}v{benchmark_cid}"

            dfx = update_df(dfx, df_cid)

        super().__init__(
            df = dfx,
            xcats = xcats + [benchmark_xcat] if isinstance(xcats, list) else [xcats, benchmark_xcat],
            cids = dfx["cid"].unique(),
            start = start,
            end = end,
            blacklist = None,
            freq = "D",
            lag = 0,
        )

        # Create initial dataframes to store estimated betas and OOS hedged returns
        self.betas = pd.DataFrame(columns=["cid", "real_date", "xcat", "value"])
        self.hedged_returns = pd.DataFrame(
            columns=["cid", "real_date", "xcat", "value"]
        )

    def estimate_beta(
        self,
        beta_xcat,
        hedged_return_xcat,
        models,
        hyperparameters,
        scorers,
        inner_splitters,
        search_type = "grid",
        cv_summary = "mean",
        min_cids = 4,
        min_periods = 12 * 3,
        est_freq = "D",
        max_periods = None,
        split_dictionary = None,
        n_iter = None,
        n_jobs_outer = -1,
        n_jobs_inner = 1,
    ):
        # Checks
        # TODO

        # Create pandas dataframes to store betas and hedged returns
        stored_betas = pd.DataFrame(
            index=self.forecast_idxs, columns=[beta_xcat], data=np.nan, dtype=np.float32
        )
        stored_hedged_returns = pd.DataFrame(
            index=self.forecast_idxs, columns=[hedged_return_xcat], data=np.nan, dtype=np.float32
        )

        # Set up outer splitter
        outer_splitter = ExpandingFrequencyPanelSplit(
            expansion_freq = est_freq,
            test_freq = est_freq,
            min_cids = min_cids,
            min_periods = min_periods,
        )

        # Run pipeline
        results = self.run(
            name = beta_xcat,
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
        beta_data = []
        hedged_return_data = []
        modelchoice_data = []

        for quantamental_data, model_data, other_data in results:
            beta_data.append(quantamental_data["betas"])
            hedged_return_data.append(quantamental_data["hedged_returns"])
            modelchoice_data.append(model_data["model_choice"])

        # Create quantamental dataframes of betas and hedged returns
        for cid, real_date, xcat, value in beta_data:
            stored_betas.loc[(cid, real_date), xcat] = value
        for cid, real_date, xcat, value in hedged_return_data:
            stored_hedged_returns.loc[(cid, real_date), xcat] = value

        stored_betas = stored_betas.groupby(level=0).ffill().dropna()
        stored_hedged_returns = stored_hedged_returns.dropna()
        stored_betas_long = pd.melt(
            frame=stored_betas.reset_index(),
            id_vars=["cid", "real_date"],
            var_name="xcat",
            value_name="value",
        )
        stored_hrets_long = pd.melt(
            frame=stored_hedged_returns.reset_index(),
            id_vars=["cid", "real_date"],
            var_name="xcat",
            value_name="value",
        )

        self.betas = pd.concat((self.betas, stored_betas_long), axis=0).astype(
            {
                "cid": "object",
                "real_date": "datetime64[ns]",
                "xcat": "object",
                "value": np.float64,
            }
        )
        self.hedged_returns = pd.concat(
            (self.hedged_returns, stored_hrets_long), axis=0
        ).astype(
            {
                "cid": "object",
                "real_date": "datetime64[ns]",
                "xcat": "object",
                "value": np.float64,
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
