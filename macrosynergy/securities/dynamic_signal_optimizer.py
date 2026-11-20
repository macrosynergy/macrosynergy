"""Sequential signal optimization over explicit constituent regimes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from joblib import Parallel, cpu_count, delayed, parallel_config
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)
from sklearn.base import BaseEstimator
from tqdm.auto import tqdm

from macrosynergy.learning.sequential.signal_optimizer import SignalOptimizer
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import concat_categorical, reduce_df

_RegimeId = Union[str, int]
_RegimeInput = Dict[_RegimeId, Dict[str, Any]]


class DynamicSignalOptimizer:
    """
    Sequential signal optimization over constituent regimes.

    Wraps a separate SignalOptimizer for each supplied regime. Features and
    cross-sections remain fixed.

    Parameters
    ----------
    df : pd.DataFrame
        Daily quantamental dataframe in JPMaQS format, with columns `cid`,
        `xcat`, `real_date` and `value`. Must contain all features.
        Candidate targets without records in a regime
        are reported and excluded from that run.
    features : list of str
        Ordered feature categories used in every regime. Must contain at least
        one category, with no duplicates or overlap with any target category.
        Candidate targets are derived from `regimes`.
    regimes : dict
        Required, nonempty mapping from string or integer regime IDs to
        dictionaries with `start`, `end`, `end_return` and `targets`. Dates
        are ISO date strings or dates; `targets` is an ordered list of target
        category names in `df`.
        IDs are retained and regimes are processed by start date.
        Output intervals must not overlap. An empty target list skips a regime.

        `start` and `end` define the output interval, inclusive and exclusive
        respectively, after rolling weekends back to the previous business day.
        `end_return` caps input observations inclusively. The history check uses
        target records through `end`, before this second cutoff. Global `start`
        and `end` still restrict the available input history.
    cids : list of str, optional
        Cross-sections considered in every vintage. Default is None, which
        considers all cross-sections in `df`.
    start : str, optional
        Start date for considered data in ISO 8601 format. Default is None,
        which uses the earliest available date.
    end : str, optional
        End date for considered data in ISO 8601 format. Default is None,
        which uses the latest available date.
    blacklist : dict, optional
        Cross-sections mapped to tuples of start and end dates to exclude,
        in ISO 8601 format. Passed to each SignalOptimizer. Default is None.
    freq : str, optional
        Analysis frequency passed to each SignalOptimizer. Default is "M"
        for monthly analysis.
    lag : int, optional
        Number of analysis periods to lag the features relative to the
        targets. Default is 1.
    xcat_aggs : list of str, optional
        Two aggregation functions, for features and targets respectively.
        Default is None, which uses ["last", "sum"].
    generate_labels : callable, optional
        Function applied to the response data by each SignalOptimizer to
        produce alternative targets or classification labels. Default is None.
    drop_nas : bool or str, optional
        Missing-data strategy passed to each SignalOptimizer. True drops rows
        with any missing values; "X" or "y" drops rows with missing features
        or targets respectively. False keeps remaining missing values. Rows
        with all features missing are always dropped. Default is True.

    Attributes
    ----------
    features : list of str
        Feature categories in their supplied order.
    targets : list of str
        Unique candidate targets, in order of first appearance across the
        chronologically sorted regimes.
    optimizers : dict
        Process names mapped to dictionaries of fitted SignalOptimizer objects,
        keyed by the original regime IDs. Skipped regimes have no child entry.
        Children retain their input data, training matrices and full diagnostics.
    regime_results : dict
        Process names mapped to regime reports. Use `get_regimes(name)` to
        retrieve a copy, including missing targets and reasons for skipping.
    regimes : pd.DataFrame
        Report from the most recent successful calculation. Empty beforehand.

    Notes
    -----
    Each regime uses its constituents' available history, including observations
    before their admission. Targets with no rows through the regime's `end` are
    excluded. Following the notebook, the initial history check counts distinct
    calendar months across the selected target records, including rows with NaN
    values. It does not require that history length for each target separately.
    The threshold is `min_periods` from `calculate_predictions`, even when the
    analysis frequency is not monthly.

    Each child receives `xcats=features + available_targets` and
    `n_targets=len(available_targets)`. The target count can change between regimes;
    callers do not supply a separate candidate list or target count.

    Both features and returns are then capped at the earlier of `end` and
    `end_return`. The child derives its learning dates after aggregation,
    lagging and missing-data handling. The latest learning date on or before
    the supplied regime start becomes its calculation `start_date`. Regimes
    with insufficient history or no subsequent prediction period are skipped.
    Other child errors propagate. Every supplied regime resets this clock,
    even when adjacent regimes have identical target membership.

    Predictions and diagnostics are retained within the business-day-adjusted
    output interval. Diagnostic records are sliced by their own dates, without
    carrying earlier records forward. Getters return copies of combined results;
    child plots and full diagnostics are available via `optimizers[name][regime_id]`.
    Keeping children also retains their data and full results in memory.

    Examples
    --------
    Given `df` with features F1 and F2 and targets Y and Z, using history from
    2018 onwards, add Z for the second quarter of 2019:

    >>> from sklearn.linear_model import LinearRegression
    >>> optimizer = DynamicSignalOptimizer(
    ...     df, features=["F1", "F2"],
    ...     regimes={
    ...         "Q1": {"start": "2018-12-31", "end": "2019-03-31",
    ...                "end_return": "2019-03-29", "targets": ["Y"]},
    ...         "Q2": {"start": "2019-03-31", "end": "2019-06-30",
    ...                "end_return": "2019-06-28", "targets": ["Y", "Z"]},
    ...     },
    ... )
    >>> optimizer.calculate_predictions(
    ...     "signal", models={"linear": LinearRegression()},
    ...     min_cids=1, min_periods=6, test_size=3, show_tqdm=False,
    ... )
    >>> signals = optimizer.get_optimized_signals("signal")
    >>> report = optimizer.get_regimes("signal")
    """

    _RESULT_TABLES = (
        "preds",
        "chosen_models",
        "feature_importances",
        "intercepts",
        "selected_ftrs",
        "ftr_corr",
    )

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        regimes: _RegimeInput,
        cids: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        blacklist: Optional[Dict[str, Any]] = None,
        freq: str = "M",
        lag: int = 1,
        xcat_aggs: Optional[List[str]] = None,
        generate_labels: Optional[Callable] = None,
        drop_nas: Union[bool, str] = True,
    ) -> None:
        self.config = _InputConfig(features=features, cids=cids, regimes=regimes)
        self._optimizer_options = {
            "start": start,
            "end": end,
            "blacklist": blacklist,
            "freq": freq,
            "lag": lag,
            "xcat_aggs": ["last", "sum"] if xcat_aggs is None else xcat_aggs,
            "generate_labels": generate_labels,
            "drop_nas": drop_nas,
        }
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if "value" not in df.columns:
            raise ValueError("df must contain a value column.")
        self.df = QuantamentalDataFrame(df)
        self.features = self.config.features
        self.targets = self.config.candidate_targets
        self.cids = self.config.cids
        for column, requested in (("xcat", self.features), ("cid", self.cids)):
            if requested is not None:
                missing = set(requested) - set(self.df[column])
                if missing:
                    raise ValueError(f"Unknown {column} values: {sorted(missing)}")
        data_window = reduce_df(
            self.df,
            xcats=self.features + self.targets,
            cids=self.cids,
            start=start,
            end=end,
            blacklist=blacklist,
        )
        if data_window.empty:
            raise ValueError(
                "No data remains after selecting categories, cids and dates."
            )
        for table, frame in _empty_result_tables(self.config.features).items():
            setattr(self, table, frame)
        self.store_correlations = False
        self.optimizers: Dict[str, Dict[_RegimeId, SignalOptimizer]] = {}
        self.regime_results: Dict[str, pd.DataFrame] = {}
        self._process_outputs: Dict[str, Set[str]] = {}
        self.regimes = pd.DataFrame()

    def calculate_predictions(
        self,
        name: str,
        models: Dict[str, BaseEstimator],
        *,
        n_jobs_vintages: int = 1,
        show_tqdm: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Calculate sequential predictions for each active vintage and store results.

        Parameters
        ----------
        name : str
            Name of the signal optimization process. A successful rerun replaces
            this process's stored results and children, preserving other processes.
        models : dict
            Model names mapped to scikit-learn compatible estimators. Models are
            deep-copied for each vintage. They must support the number of targets
            active in that vintage.
        n_jobs_vintages : int, optional
            Number of vintages to run concurrently. Default is 1. Use -1 for
            all cores available according to `joblib.cpu_count()`. Positive
            values cannot exceed that count. Time-period fitting and cross-
            validation within each vintage are fixed at one job each.
        show_tqdm : bool, optional
            Whether to display one progress bar for all supplied regimes,
            including skipped regimes.
            Results advance the bar as they are collected in schedule order;
            child progress bars are disabled. Default is True.
        **kwargs
            Options passed to `SignalOptimizer.calculate_predictions` for
            every vintage, with their native defaults. These include
            `hyperparameters`, `scorers`, `inner_splitters`, `min_cids`,
            `min_periods`, `test_size`, `max_periods`,
            `multi_target_fill`, `store_additional_data` and
            `store_correlations`. See that method for the complete options.
            `n_jobs_outer` and `n_jobs_inner` must not be supplied.
            `start_date` is derived by the wrapper and must be omitted.
            `min_periods` defaults to 36 and must be a positive integer. It is
            also the minimum number of distinct calendar months of target data
            required before applying `end_return`.

        Returns
        -------
        None
            Results are stored on the instance and accessed through its getters.

        Raises
        ------
        pydantic.ValidationError
            If the worker count, progress-bar option or min_periods is invalid.
        TypeError
            If `n_jobs_outer`, `n_jobs_inner` or `start_date` is supplied.
        ValueError
            If process and output names collide with those of another stored process.

        Notes
        -----
        Each regime derives its own `start_date` from the child's
        processed learning dates. Other options apply to every child, including
        `test_size` for retraining intervals and `max_periods` for rolling windows.
        If all regimes are skipped, the process stores empty result tables and
        a report explaining each skipped regime.

        Results are published only after all vintages have fitted and their
        output tables have been combined successfully. A failure preserves
        previously stored results. Exceptions from child optimizers propagate
        to the caller.
        """
        if {"n_jobs_outer", "n_jobs_inner"}.intersection(kwargs):
            raise TypeError(
                "n_jobs_outer and n_jobs_inner are fixed at 1; "
                "remove them and use n_jobs_vintages instead."
            )
        if "start_date" in kwargs:
            raise TypeError("start_date is derived from each regime; remove it.")
        execution_options = {
            "n_jobs_vintages": n_jobs_vintages,
            "show_tqdm": show_tqdm,
        }
        if "min_periods" in kwargs:
            execution_options["min_periods"] = kwargs["min_periods"]
        execution = _ExecutionConfig(**execution_options)
        run_options = {**kwargs, "min_periods": execution.min_periods}
        results = self._fit_periods(
            name=name, models=models, run_options=run_options, execution=execution
        )
        tables = self._collect_results(results, name)
        self._store_results(
            name,
            tables,
            results,
            store_correlations=kwargs.get("store_correlations", False),
        )

    def get_regimes(self, name: Optional[str] = None) -> pd.DataFrame:
        """
        Return the planned regimes or the outcome of a completed process.

        Parameters
        ----------
        name : str, optional
            Process name. Default is None, which returns the plan without fitting.

        Returns
        -------
        pd.DataFrame
            Regimes indexed by their original IDs, sorted by start date. Plan
            columns are `start`, `end`, `end_return` and `targets`. A completed
            process also includes `available_targets`, `missing_targets`,
            `history_months`, `first_prediction`, `status` and `skip_reason`.
            Dates in the plan are the supplied dates, before business-day rollback.

        Raises
        ------
        ValueError
            If the requested process has not completed successfully.
        """
        if name is not None:
            if name not in self.regime_results:
                raise ValueError(f"No regime results stored for {name!r}.")
            return self.regime_results[name].copy()
        records = [
            dict(
                regime_id=key,
                start=pd.Timestamp(regime.start),
                end=pd.Timestamp(regime.end),
                end_return=pd.Timestamp(regime.end_return),
                targets=tuple(regime.targets),
            )
            for key, regime in self.config.regimes.items()
        ]
        return pd.DataFrame(records).set_index("regime_id")

    def get_optimized_signals(
        self, name: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Return combined daily predictions for one or more processes or targets.

        Parameters
        ----------
        name : str or list of str, optional
            Process names or individual output category names. A process name
            selects all its target outputs. Default is None, which selects all
            stored predictions.

        Returns
        -------
        pd.DataFrame
            Copy of the predictions in JPMaQS format, with columns `cid`,
            `xcat`, `real_date` and `value`. Output categories are the
            process name when there is one candidate target, otherwise
            `<target>_<name>`, including vintages in which only one candidate
            target is active.
            Only predictions dated within each vintage are retained.

        Raises
        ------
        TypeError
            If name is not a string or list of strings.
        ValueError
            If a requested output has no stored predictions.
        """
        if name is not None:
            names = [name] if isinstance(name, str) else name
            if not isinstance(names, list) or not all(
                isinstance(item, str) for item in names
            ):
                raise TypeError("name must be a string or a list of strings.")
            name = [
                output
                for item in names
                for output in sorted(self._process_outputs.get(item, [item]))
            ]
        signals = self._get_result("preds", name, column="xcat")
        return QuantamentalDataFrame(
            signals, categorical=self.df.InitializedAsCategorical
        ).to_original_dtypes()

    def get_optimal_models(
        self, name: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Return the model choices for one or more processes across vintages.

        Parameters
        ----------
        name : str or list of str, optional
            Process names to select. Default is None, which selects all stored
            records. Target output names are not process names.

        Returns
        -------
        pd.DataFrame
            Copy of model choices at retained retraining dates, with columns
            `real_date`, `name`, `model_type`, `score`, `hparams`,
            `additional_data` and `n_splits_used`. Additional data contains
            attributes requested through `store_additional_data`.
        """
        return self._get_result("chosen_models", name)

    def get_selected_features(
        self, name: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Return feature-selection indicators across vintages.

        Parameters
        ----------
        name : str or list of str, optional
            Process names to select. Default is None, which selects all stored
            records.

        Returns
        -------
        pd.DataFrame
            Copy of the indicators at retained retraining dates, with columns
            `real_date`, `name` and one column per input feature. A value
            of 1 indicates a feature identified as an input to the final
            estimator; 0 indicates it was not identified as such.
        """
        return self._get_result("selected_ftrs", name)

    def get_feature_importances(
        self, name: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Return feature importances or coefficients across vintages.

        Parameters
        ----------
        name : str or list of str, optional
            Process names to select. Default is None, which selects all stored
            records.

        Returns
        -------
        pd.DataFrame
            Copy of importances at retained retraining dates, with columns
            `real_date`, `name` and one column per input feature. Named
            selections are sorted by date.

        Notes
        -----
        Availability depends on the selected estimator exposing
        `feature_importances_` or `coef_` and on feature-name recovery
        through any pipeline. Unavailable values are NaN. The child averages
        coefficients across targets when their shape matches its target count;
        these are not separate per-target coefficient tables.
        """
        frame = self._get_result("feature_importances", name)
        return frame if name is None else frame.sort_values("real_date")

    def get_intercepts(
        self, name: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Return estimator intercepts across vintages.

        Parameters
        ----------
        name : str or list of str, optional
            Process names to select. Default is None, which selects all stored
            records.

        Returns
        -------
        pd.DataFrame
            Copy of intercepts at retained retraining dates, with columns
            `real_date`, `name` and `intercepts`. Named selections are
            sorted by date. Unavailable intercepts are NaN. The child averages
            per-target intercepts when their length matches its target count.
        """
        frame = self._get_result("intercepts", name)
        return frame if name is None else frame.sort_values("real_date")

    def get_feature_correlations(
        self, name: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Return correlations between input and transformed features across vintages.

        Parameters
        ----------
        name : str or list of str, optional
            Process names to select. Default is None, which selects all stored
            records.

        Returns
        -------
        pd.DataFrame
            Copy of correlations at retained retraining dates, with columns
            `real_date`, `name`, `predictor_input`, `pipeline_input` and
            `pearson`. An unfiltered request returns an empty table if no
            correlations have been stored.

        Notes
        -----
        Requires `store_correlations=True` during calculation and models
        supported by SignalOptimizer's pipeline correlation calculation.
        Selecting a process with no stored correlation records raises ValueError.
        """
        return self._get_result("ftr_corr", name)

    def _fit_periods(
        self,
        name: str,
        models: Dict[str, BaseEstimator],
        run_options: Dict[str, Any],
        execution: _ExecutionConfig,
    ) -> Dict[_RegimeId, _RegimeResult]:
        """Collect fitted and skipped regimes in date order, with one progress bar."""
        # QDF's original-dtype flag is not preserved by process serialization.
        worker_df = pd.DataFrame(self.df)
        regimes = self.config.regimes
        outcomes = {}
        with parallel_config(backend="loky", inner_max_num_threads=1):
            with Parallel(
                n_jobs=execution.n_jobs_vintages, return_as="generator"
            ) as parallel, tqdm(
                total=len(regimes),
                desc="Vintages",
                disable=not execution.show_tqdm,
            ) as progress:
                results = parallel(
                    delayed(_fit_vintage)(
                        df=worker_df,
                        features=self.features,
                        cids=self.cids,
                        regime=regime,
                        init_options=self._optimizer_options,
                        name=name,
                        models=models,
                        run_options=run_options,
                    )
                    for regime in regimes.values()
                )
                for key, result in zip(regimes, results):
                    if result.optimizer is not None:
                        result.optimizer.df.InitializedAsCategorical = (
                            self.df.InitializedAsCategorical
                        )
                    outcomes[key] = result
                    progress.update(1)
        return outcomes

    def _collect_results(
        self, results: Dict[_RegimeId, _RegimeResult], name: str
    ) -> Dict[str, pd.DataFrame]:
        """Retain each regime's outputs without changing its child diagnostics."""
        pieces = {table: [] for table in self._RESULT_TABLES}
        bday = pd.offsets.BDay()
        for key, result in results.items():
            if result.optimizer is None:
                continue
            regime = self.config.regimes[key]
            for table in self._RESULT_TABLES:
                frame = getattr(result.optimizer, table)
                within = frame["real_date"].ge(
                    bday.rollback(pd.Timestamp(regime.start))
                )
                within &= frame["real_date"].lt(bday.rollback(pd.Timestamp(regime.end)))
                frame = frame.loc[within].copy()
                if (
                    table == "preds"
                    and len(self.targets) > 1
                    and len(result.targets) == 1
                ):
                    frame["xcat"] = frame["xcat"].cat.rename_categories(
                        {name: f"{result.targets[0]}_{name}"}
                    )
                pieces[table].append(frame)

        tables = _empty_result_tables(self.config.features)
        for table, frames in pieces.items():
            if not frames:
                continue
            # Align category sets so concat never expands labels into object arrays.
            for column in frames[0].select_dtypes("category"):
                categories = pd.Index(
                    [
                        category
                        for frame in frames
                        for category in frame[column].cat.categories
                    ]
                ).unique()
                dtype = pd.CategoricalDtype(categories)
                for frame in frames:
                    frame[column] = frame[column].astype(dtype)
            tables[table] = pd.concat(frames, ignore_index=True)
        return tables

    def _store_results(
        self,
        name: str,
        tables: Dict[str, pd.DataFrame],
        results: Dict[_RegimeId, _RegimeResult],
        *,
        store_correlations: bool,
    ) -> None:
        """Stage all replacements before publishing any changes to stored results."""
        output_names = set(tables["preds"]["xcat"])
        for other_name, other_outputs in self._process_outputs.items():
            if other_name == name:
                continue
            if (
                name in other_outputs
                or other_name in output_names
                or output_names.intersection(other_outputs)
            ):
                raise ValueError(
                    f"Process or output category names collide with {other_name!r}."
                )

        previous_outputs = self._process_outputs.get(name, set())
        replacements = {}
        for table, incoming in tables.items():
            existing = getattr(self, table)
            belongs_to_process = (
                existing["xcat"].isin(previous_outputs)
                if table == "preds"
                else existing["name"].eq(name)
            )
            replacements[table] = concat_categorical(
                existing.loc[~belongs_to_process], incoming
            )
        report = self.get_regimes()
        report["available_targets"] = [result.targets for result in results.values()]
        report["missing_targets"] = [
            result.missing_targets for result in results.values()
        ]
        report["history_months"] = [
            result.history_months for result in results.values()
        ]
        report["first_prediction"] = pd.to_datetime(
            [result.first_prediction for result in results.values()]
        )
        report["status"] = [
            "fitted" if result.optimizer is not None else "skipped"
            for result in results.values()
        ]
        report["skip_reason"] = [result.skip_reason for result in results.values()]
        optimizers = {
            key: result.optimizer
            for key, result in results.items()
            if result.optimizer is not None
        }

        for table, frame in replacements.items():
            setattr(self, table, frame)
        self.optimizers[name] = optimizers
        self._process_outputs[name] = output_names
        self.store_correlations = store_correlations
        self.regimes = report
        self.regime_results[name] = report

    def _get_result(
        self, table: str, name: Optional[Union[str, List[str]]], *, column: str = "name"
    ) -> pd.DataFrame:
        """Select named results and return a dataframe copy."""
        frame = getattr(self, table)
        if name is None:
            return frame.copy()
        names = [name] if isinstance(name, str) else name
        if not isinstance(names, list) or not all(
            isinstance(item, str) for item in names
        ):
            raise TypeError("name must be a string or a list of strings.")
        missing = set(names) - set(frame[column])
        if missing:
            raise ValueError(f"No {table} results stored for {sorted(missing)}.")
        return frame.loc[frame[column].isin(names)].copy()


class _RegimeConfig(BaseModel):
    """Validate one explicit output interval and its input cutoff."""

    model_config = ConfigDict(extra="forbid")

    start: date
    end: date
    end_return: date
    targets: List[StrictStr]

    @model_validator(mode="after")
    def check_interval(self) -> _RegimeConfig:
        if self.start >= self.end:
            raise ValueError("Regime start must precede end.")
        if len(self.targets) != len(set(self.targets)):
            raise ValueError("Regime targets must not contain duplicates.")
        return self


class _InputConfig(BaseModel):
    """Validate candidates and retain separate, ordered regimes with their IDs."""

    model_config = ConfigDict(extra="forbid")

    features: List[StrictStr] = Field(min_length=1)
    cids: Optional[List[StrictStr]] = None
    regimes: Dict[Union[StrictStr, StrictInt], _RegimeConfig] = Field(min_length=1)

    @property
    def candidate_targets(self) -> List[str]:
        return list(
            dict.fromkeys(
                target for regime in self.regimes.values() for target in regime.targets
            )
        )

    @model_validator(mode="after")
    def check_members(self) -> _InputConfig:
        if len(set(self.features)) != len(self.features):
            raise ValueError("features must not contain duplicates.")
        self.regimes = dict(
            sorted(self.regimes.items(), key=lambda item: item[1].start)
        )
        previous_end = None
        for regime in self.regimes.values():
            if previous_end is not None and regime.start < previous_end:
                raise ValueError("Regime output intervals must not overlap.")
            previous_end = regime.end
        overlap = set(self.features).intersection(self.candidate_targets)
        if overlap:
            raise ValueError(
                f"Features and targets must not overlap: {sorted(overlap)}"
            )
        return self


@dataclass
class _RegimeResult:
    """A fitted child or an explicit reason for skipping its regime."""

    targets: Tuple[str, ...]
    missing_targets: Tuple[str, ...] = ()
    history_months: Optional[int] = None
    first_prediction: Optional[pd.Timestamp] = None
    skip_reason: Optional[str] = None
    optimizer: Optional[SignalOptimizer] = None


class _ExecutionConfig(BaseModel):
    """Validate vintage workers and resolve -1 before starting any jobs."""

    n_jobs_vintages: int = Field(default=1, strict=True)
    show_tqdm: StrictBool = True
    min_periods: int = Field(default=36, strict=True, ge=1)

    @field_validator("n_jobs_vintages")
    @classmethod
    def check_jobs(cls, value: int) -> int:
        if value != -1 and value < 1:
            raise ValueError("Worker counts must be positive integers or -1.")
        cores = cpu_count()
        if value == -1:
            return cores
        if value > cores:
            raise ValueError(
                f"n_jobs_vintages={value}, exceeding {cores} available cores."
            )
        return value


def _fit_vintage(
    *,
    df: pd.DataFrame,
    features: List[str],
    regime: _RegimeConfig,
    cids: Optional[List[str]],
    init_options: Dict[str, Any],
    name: str,
    models: Dict[str, BaseEstimator],
    run_options: Dict[str, Any],
) -> _RegimeResult:
    """Apply the notebook's preparation, history check and prediction clock."""
    targets = regime.targets
    result = _RegimeResult(targets=tuple(targets))
    options = init_options.copy()
    run_options = deepcopy(run_options)
    end = pd.Timestamp(regime.end)
    if options.get("end") is not None:
        end = min(end, pd.Timestamp(options["end"]))
    df = reduce_df(
        df,
        xcats=features + targets,
        cids=cids,
        start=options.get("start"),
        end=end.strftime("%Y-%m-%d"),
        blacklist=options.get("blacklist"),
    )
    returns = df.loc[df["xcat"].isin(targets)]
    available = set(returns["xcat"])
    result.targets = tuple(target for target in targets if target in available)
    result.missing_targets = tuple(
        target for target in targets if target not in available
    )
    # Match the notebook: count union months before applying end_return.
    result.history_months = returns["real_date"].dt.to_period("M").nunique()
    if not result.targets:
        result.skip_reason = "no_targets"
        return result
    if result.history_months < run_options["min_periods"]:
        result.skip_reason = "insufficient_history"
        return result
    cutoff = min(end, pd.Timestamp(regime.end_return))
    df = df.loc[df["real_date"].le(cutoff)]
    if df.empty:
        result.skip_reason = "no_input_data"
        return result
    options["end"] = cutoff.strftime("%Y-%m-%d")

    optimizer = SignalOptimizer(
        df,
        features + list(result.targets),
        cids=cids,
        n_targets=len(result.targets),
        **options,
    )
    dates = optimizer.y.index.get_level_values(1).unique().sort_values()
    eligible = dates[dates <= pd.Timestamp(regime.start)]
    if eligible.empty or eligible[-1] >= dates.max():
        result.skip_reason = "no_prediction_periods"
        return result
    result.first_prediction = eligible[-1]
    run_options["start_date"] = result.first_prediction.strftime("%Y-%m-%d")
    optimizer.calculate_predictions(
        name,
        deepcopy(models),
        n_jobs_outer=1,
        n_jobs_inner=1,
        show_tqdm=False,
        **run_options,
    )
    result.optimizer = optimizer
    return result


def _empty_result_tables(features: List[str]) -> Dict[str, pd.DataFrame]:
    """Initialize output schemas without constructing a learner or training panel."""
    dated_process = {"real_date": "datetime64[ns]", "name": "category"}
    schemas = {
        "preds": {
            "real_date": "datetime64[ns]",
            "cid": "category",
            "xcat": "category",
            "value": "float32",
        },
        "chosen_models": {
            **dated_process,
            "model_type": "category",
            "score": "float32",
            "hparams": "object",
            "additional_data": "object",
            "n_splits_used": "object",
        },
        "feature_importances": {**dated_process, **{f: "float32" for f in features}},
        "intercepts": {**dated_process, "intercepts": "float32"},
        "selected_ftrs": {**dated_process, **{f: "int64" for f in features}},
        "ftr_corr": {
            **dated_process,
            "predictor_input": "category",
            "pipeline_input": "category",
            "pearson": "float64",
        },
    }
    return {
        name: pd.DataFrame(
            {col: pd.Series(dtype=dtype) for col, dtype in schema.items()}
        )
        for name, schema in schemas.items()
    }


if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression

    from macrosynergy.management.simulate import make_qdf

    cids = ["GBP", "USD"]
    xcats = ["CRY", "GROWTH", "XR1", "XR2"]
    df_cids = pd.DataFrame(
        [["2018-01-01", "2020-12-31", 0, 1]] * len(cids),
        index=cids,
        columns=["earliest", "latest", "mean_add", "sd_mult"],
    )
    df_xcats = pd.DataFrame(
        [
            ["2018-01-01", "2020-12-31", 0, 1, 0.9, 1],
            ["2018-01-01", "2020-12-31", 0, 1, 0.8, 1],
            ["2018-01-01", "2020-12-31", 0, 1, 0.2, 0.5],
            ["2018-01-01", "2020-12-31", 0, 1, 0.3, 0.5],
        ],
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )
    df = make_qdf(df_cids, df_xcats, back_ar=0.75, seed=42)

    # Use earlier observations for training; each quarter has its own fit clock.
    optimizer = DynamicSignalOptimizer(
        df,
        features=["CRY", "GROWTH"],
        cids=cids,
        freq="M",
        regimes={
            "2019-Q1": {
                "start": "2018-12-31",
                "end": "2019-03-31",
                "end_return": "2019-03-29",
                "targets": ["XR1"],
            },
            "2019-Q2": {
                "start": "2019-03-31",
                "end": "2019-06-30",
                "end_return": "2019-06-28",
                "targets": ["XR1", "XR2"],
            },
            "2019-Q3": {
                "start": "2019-06-30",
                "end": "2019-09-30",
                "end_return": "2019-09-27",
                "targets": ["XR2"],
            },
        },
    )
    optimizer.calculate_predictions(
        "linear",
        models={"linear": LinearRegression()},
        min_cids=2,
        min_periods=6,
        test_size=3,
        n_jobs_vintages=1,  # Increase to run vintages in parallel.
    )

    print(optimizer.get_regimes("linear").to_string())
    signals = optimizer.get_optimized_signals("linear")
    print(
        signals.groupby("xcat", observed=True)
        .agg(
            first_signal=("real_date", "min"),
            last_signal=("real_date", "max"),
            rows=("value", "size"),
        )
        .to_string()
    )
