"""
Class to estimate market betas and calculate out-of-sample hedged returns based on
sequential learning. 
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor

from macrosynergy.learning import ExpandingFrequencyPanelSplit
from macrosynergy.learning.sequential import BasePanelLearner
from macrosynergy.management import categories_df, reduce_df, update_df


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
        cids=None,
        start=None,
        end=None,
    ):
        # Checks
        # TODO: Refactor these checks.
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if not set(["cid", "xcat", "real_date", "value"]).issubset(df.columns):
            raise ValueError(
                "df must have columns 'cid', 'xcat', 'real_date' and 'value'."
            )

        # cids checks
        if cids is not None:
            if not isinstance(cids, list):
                raise TypeError("cids must be a list.")
            if not all(isinstance(cid, str) for cid in cids):
                raise TypeError("All elements in cids must be strings.")
            for cid in cids:
                if cid not in df["cid"].unique():
                    raise ValueError(f"{cid} not in the dataframe.")

        if not isinstance(benchmark_return, str):
            raise TypeError("benchmark_return must be a string.")

        # Create pseudo-panel
        dfx = pd.DataFrame(columns=["real_date", "cid", "xcat", "value"])

        self.benchmark_return = benchmark_return
        self.benchmark_cid, self.benchmark_xcat = benchmark_return.split("_", 1)

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
                xcats=[self.benchmark_xcat],
                cids=[self.benchmark_cid],
            )

            # Combine contract and benchmark returns and rename cross-section identifier
            # in order to match the benchmark return with each cross section in a pseudo
            # panel
            df_cid = pd.concat([dfa, dfb], axis=0)
            df_cid["cid"] = f"{cid}v{self.benchmark_cid}"

            dfx = update_df(dfx, df_cid)

        super().__init__(
            df=dfx,
            xcats=(
                [self.benchmark_xcat] + xcats
                if isinstance(xcats, list)
                else [self.benchmark_xcat, xcats]
            ),
            cids=list(dfx["cid"].unique()),
            start=start,
            end=end,
            blacklist=None,
            freq="D",
            lag=0,
        )

        # Create forecast dataframe index
        min_date = min(self.unique_date_levels)
        max_date = max(self.unique_date_levels)
        forecast_date_levels = pd.date_range(start=min_date, end=max_date, freq="B")
        self.forecast_idxs = pd.MultiIndex.from_product(
            [
                [cid.split("v")[0] for cid in self.unique_xs_levels],
                forecast_date_levels,
            ],
            names=["cid", "real_date"],
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
        search_type="grid",
        cv_summary="mean",
        min_cids=4,
        min_periods=12 * 3,
        est_freq="D",
        max_periods=None,
        split_functions=None,
        n_iter=None,
        n_jobs_outer=-1,
        n_jobs_inner=1,
    ):
        # Checks
        # TODO

        # Create pandas dataframes to store betas and hedged returns
        stored_betas = pd.DataFrame(
            index=self.forecast_idxs, columns=[beta_xcat], data=np.nan, dtype=np.float32
        )
        stored_hedged_returns = pd.DataFrame(
            index=self.forecast_idxs,
            columns=[hedged_return_xcat],
            data=np.nan,
            dtype=np.float32,
        )

        # Set up outer splitter
        outer_splitter = ExpandingFrequencyPanelSplit(
            expansion_freq=est_freq,
            test_freq=est_freq,
            min_cids=min_cids,
            min_periods=min_periods,
        )

        # Run pipeline
        results = self.run(
            name=beta_xcat,
            outer_splitter=outer_splitter,
            inner_splitters=inner_splitters,
            models=models,
            hyperparameters=hyperparameters,
            scorers=scorers,
            search_type=search_type,
            cv_summary=cv_summary,
            split_functions=split_functions,
            n_iter=n_iter,
            n_jobs_outer=n_jobs_outer,
            n_jobs_inner=n_jobs_inner,
        )

        # Collect results from the worker
        beta_data = []
        hedged_return_data = []
        model_choice_data = []

        for split_result in results:
            beta_data.extend(split_result["betas"])
            hedged_return_data.extend(split_result["hedged_returns"])
            model_choice_data.append(split_result["model_choice"])

        # Create quantamental dataframes of betas and hedged returns
        for cid, real_date, xcat, value in beta_data:
            stored_betas.loc[(cid, real_date), xcat] = value
        for cid, real_date, xcat, value in hedged_return_data:
            stored_hedged_returns.loc[(cid, real_date), hedged_return_xcat] = value

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

        self.betas = pd.concat(
            (self.betas if self.betas.size != 0 else None, stored_betas_long), axis=0
        ).astype(
            {
                "cid": "object",
                "real_date": "datetime64[ns]",
                "xcat": "object",
                "value": np.float64,
            }
        )
        self.hedged_returns = pd.concat(
            (
                self.hedged_returns if self.hedged_returns.size != 0 else None,
                stored_hrets_long,
            ),
            axis=0,
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
            columns=self.chosen_models.columns,
            data=model_choice_data,
        )
        self.chosen_models = pd.concat(
            (
                self.chosen_models if self.chosen_models.size != 0 else None,
                model_df_long,
            ),
            axis=0,
        ).astype(
            {
                "real_date": "datetime64[ns]",
                "name": "object",
                "model_type": "object",
                "hparams": "object",
                "n_splits_used": "object",
            }
        )

    def store_split_data(
        self,
        pipeline_name,
        optimal_model,
        optimal_model_name,
        optimal_model_score,
        optimal_model_params,
        inner_splitters_adj,
        X_train,
        y_train,
        X_test,
        y_test,
        timestamp,
        adjusted_test_index,
    ):
        if isinstance(optimal_model, VotingRegressor):
            estimators = optimal_model.estimators_
            coefs_list = [est.coefs_ for est in estimators]
            sum_dict = defaultdict(lambda: [0, 0])

            for coefs in coefs_list:
                for key, value in coefs.items():
                    sum_dict[key][0] += value
                    sum_dict[key][1] += 1

            betas = {key: sum / count for key, (sum, count) in sum_dict.items()}
        else:
            betas = optimal_model.coefs_

        betas_list = [
            [
                cid.split("v")[0],
                X_train.index.get_level_values(1).max(),
                pipeline_name,
                beta,
            ]
            for cid, beta in betas.items()
        ]

        # Now calculate the induced hedged returns
        betas_series = pd.Series(betas)
        XB = X_test.mul(betas_series, level=0, axis=0)
        hedged_returns = y_test.values.reshape(-1, 1) - XB.values.reshape(-1, 1)
        hedged_returns_data = [
            [idx[0].split("v")[0], idx[1]] + [pipeline_name] + [hedged_returns[i]]
            for i, (idx, _) in enumerate(y_test.items())
        ]
        return {"betas": betas_list, "hedged_returns": hedged_returns_data}


if __name__ == "__main__":
    from macrosynergy.learning import (
        ExpandingKFoldPanelSplit,
        LinearRegressionSystem,
        neg_mean_abs_corr,
    )
    from macrosynergy.management.simulate import make_qdf

    # Simulate a panel dataset of benchmark and contract returns
    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["BENCH_XR", "CONTRACT_XR"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2002-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2003-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2000-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["BENCH_XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CONTRACT_XR"] = ["2001-01-01", "2020-12-31", 0.1, 1, 0, 0.3]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    # Initialize the BetaEstimator object
    # Use for the benchmark return: USD_BENCH_XR.
    be = BetaEstimator(
        df=dfd,
        xcats="CONTRACT_XR",
        benchmark_return="USD_BENCH_XR",
        cids=cids,
    )

    models = {
        "LR": LinearRegressionSystem(min_xs_samples=21 * 3),
    }
    hparam_grid = {"LR": {"fit_intercept": [True, False], "positive": [True, False]}}

    scorer = {"scorer": neg_mean_abs_corr}

    be.estimate_beta(
        beta_xcat="BETA_NSA",
        hedged_return_xcat="HEDGED_RETURN_NSA",
        models=models,
        hyperparameters=hparam_grid,
        scorers=scorer,
        inner_splitters={"expandingkfold": ExpandingKFoldPanelSplit(n_splits=5)},
        search_type="grid",
        cv_summary="median",
        min_cids=1,
        min_periods=21 * 12,
        est_freq="Q",
        n_jobs_outer=-1,
        n_jobs_inner=1,
    )

    be.models_heatmap(name="BETA_NSA")

    # evaluation_df = be.evaluate_hedged_returns(
    #     correlation_types=["pearson", "spearman", "kendall"],
    #     freqs=["W", "M", "Q"],
    # )

    # print(evaluation_df)
