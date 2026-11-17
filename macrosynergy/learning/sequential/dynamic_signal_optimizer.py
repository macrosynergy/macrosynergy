"""Sequential signal optimization with dated target selections."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from joblib import Parallel, cpu_count, delayed, parallel_config
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)
from sklearn.base import BaseEstimator
from tqdm.auto import tqdm

from macrosynergy.learning.sequential.signal_optimizer import SignalOptimizer
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import concat_categorical, reduce_df

_ScheduleInput = Union[Dict[Union[str, date], List[str]], pd.DataFrame]


class _ScheduleConfig(BaseModel):
    """Validate target membership and normalize schedules into date order."""

    model_config = ConfigDict(extra="forbid")

    xcats: List[StrictStr] = Field(min_length=2)
    n_targets: int = Field(default=1, strict=True, ge=1)
    cids: Optional[List[StrictStr]] = None
    target_schedule: Optional[Dict[date, List[StrictStr]]] = None

    @property
    def features(self) -> List[str]:
        return self.xcats[: -self.n_targets]

    @property
    def targets(self) -> List[str]:
        return self.xcats[-self.n_targets :]

    @field_validator("target_schedule", mode="before")
    @classmethod
    def read_schedule(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, pd.DataFrame):
            if not value.index.is_unique or not value.columns.is_unique:
                raise ValueError("Schedule dates and columns must be unique.")
            if value.isna().any().any() or not all(
                pd.api.types.is_bool_dtype(dtype) for dtype in value.dtypes
            ):
                raise ValueError("Schedule frames must contain only booleans.")
            value = {key: list(row.index[row]) for key, row in value.iterrows()}
        if not isinstance(value, dict) or not value:
            raise ValueError("A schedule must be a nonempty date-to-list mapping.")
        if not all(isinstance(key, (str, date, datetime)) for key in value):
            raise ValueError("Schedule keys must be dates or ISO date strings.")
        return value

    @field_validator("target_schedule")
    @classmethod
    def sort_schedule(
        cls, value: Optional[Dict[date, List[str]]]
    ) -> Optional[Dict[date, List[str]]]:
        return None if value is None else dict(sorted(value.items()))

    @model_validator(mode="after")
    def check_members(self) -> _ScheduleConfig:
        if self.n_targets >= len(self.xcats):
            raise ValueError("n_targets must leave at least one feature.")
        if len(set(self.xcats)) != len(self.xcats):
            raise ValueError("xcats must not contain duplicates.")
        candidates = set(self.targets)
        for members in (self.target_schedule or {}).values():
            unknown = set(members) - candidates
            if unknown:
                raise ValueError(f"Unknown targets: {sorted(unknown)}")
        return self


@dataclass(frozen=True)
class _TargetPeriod:
    """A fixed target selection over [start, end); None means no upper bound."""

    start: pd.Timestamp
    end: Optional[pd.Timestamp]
    targets: Tuple[str, ...]


class _ExecutionConfig(BaseModel):
    """Validate vintage workers and resolve -1 before starting any jobs."""

    n_jobs_vintages: int = Field(default=1, strict=True)
    show_tqdm: StrictBool = True

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
    xcats: List[str],
    cids: Optional[List[str]],
    n_targets: int,
    init_options: Dict[str, Any],
    name: str,
    models: Dict[str, BaseEstimator],
    run_options: Dict[str, Any],
) -> SignalOptimizer:
    """Fit one vintage without serializing the parent wrapper or sharing models."""
    optimizer = SignalOptimizer(
        df, xcats, cids=cids, n_targets=n_targets, **init_options
    )
    optimizer.calculate_predictions(
        name,
        deepcopy(models),
        n_jobs_outer=1,
        n_jobs_inner=1,
        show_tqdm=False,
        **deepcopy(run_options),
    )
    return optimizer


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


class DynamicSignalOptimizer:
    """Wrap SignalOptimizer for changing targets, with fixed features and cids.

    ``xcats`` contains features followed by ``n_targets`` candidate targets.
    ``target_schedule`` is a date-to-target-list mapping or a boolean DataFrame
    indexed by date. Dates are inclusive signal-date boundaries, without
    rounding. An empty list suspends output until the next nonempty selection.
    Before the first supplied date there is no output; omitting the schedule
    selects all targets throughout.

    Each period runs an ordinary SignalOptimizer over the supplied history,
    then keeps outputs within that period. Calculation options keep their
    native meaning. Schedule changes do not force a retrain; this simple MVP
    can repeat fits outside a period's output interval.

    Getters return the combined result tables. ``optimizers[name]`` holds the
    ordinary optimizers for per-vintage plots and training diagnostics. The
    wrapper does not construct a training panel or expose a learner's run method.
    Other constructor arguments are passed through to each SignalOptimizer.
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
        xcats: List[str],
        cids: Optional[List[str]] = None,
        n_targets: int = 1,
        start: Optional[str] = None,
        end: Optional[str] = None,
        blacklist: Optional[Dict[str, Any]] = None,
        freq: str = "M",
        lag: int = 1,
        xcat_aggs: Optional[List[str]] = None,
        generate_labels: Optional[Callable] = None,
        drop_nas: Union[bool, str] = True,
        target_schedule: Optional[_ScheduleInput] = None,
    ) -> None:
        self.config = _ScheduleConfig(
            xcats=xcats, n_targets=n_targets, cids=cids, target_schedule=target_schedule
        )
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
        self.xcats = self.config.xcats
        self.n_targets = self.config.n_targets
        self.cids = self.config.cids
        for column, requested in (("xcat", self.xcats), ("cid", self.cids)):
            if requested is not None:
                missing = set(requested) - set(self.df[column])
                if missing:
                    raise ValueError(f"Unknown {column} values: {sorted(missing)}")
        data_window = reduce_df(
            self.df,
            xcats=self.xcats,
            cids=self.cids,
            start=start,
            end=end,
            blacklist=blacklist,
        )
        if data_window.empty:
            raise ValueError(
                "No data remains after selecting categories, cids and dates."
            )
        self._data_start = data_window["real_date"].min()
        self._data_end = data_window["real_date"].max()
        for table, frame in _empty_result_tables(self.config.features).items():
            setattr(self, table, frame)
        self.store_correlations = False
        self.optimizers: Dict[str, List[SignalOptimizer]] = {}
        self._process_outputs: Dict[str, Set[str]] = {}
        self.regimes = pd.DataFrame()

    def _periods(self) -> List[_TargetPeriod]:
        """Clip schedule changes to the data window and merge identical selections."""
        candidates = self.config.targets
        schedule = self.config.target_schedule
        first, last = self._data_start, self._data_end
        if schedule is None:
            return [_TargetPeriod(first, None, tuple(candidates))]

        # Entries before the data window determine its opening selection.
        selections = {first: ()}
        for effective, members in schedule.items():
            start = pd.Timestamp(effective)
            if start > last:
                break
            selected = set(members)
            selections[max(first, start)] = tuple(
                target for target in candidates if target in selected
            )

        changes = []
        for start, targets in selections.items():
            if not changes or targets != changes[-1][1]:
                changes.append((start, targets))
        ends = [start for start, _ in changes[1:]] + [None]
        return [
            _TargetPeriod(start, end, targets)
            for (start, targets), end in zip(changes, ends)
        ]

    def get_regimes(self) -> pd.DataFrame:
        """Return target selections within the input window, before preprocessing.

        Starts are inclusive and ends exclusive. Aggregation, lagging and
        training-history requirements are applied later by the child optimizers.
        """
        return pd.DataFrame([asdict(period) for period in self._periods()])

    def calculate_predictions(
        self,
        name: str,
        models: Dict[str, BaseEstimator],
        *,
        n_jobs_vintages: int = 1,
        show_tqdm: bool = True,
        **kwargs: Any,
    ) -> None:
        """Fit vintages and replace the named process's results after success.

        ``n_jobs_vintages`` controls concurrent target periods (default 1), up to
        joblib.cpu_count(). Use -1 for all available logical cores. Time-period
        fits and cross-validation are always serial within each vintage.

        ``show_tqdm`` controls the single vintage bar; child bars are disabled.
        Other options retain SignalOptimizer.calculate_predictions semantics.
        """
        if {"n_jobs_outer", "n_jobs_inner"}.intersection(kwargs):
            raise TypeError(
                "n_jobs_outer and n_jobs_inner are fixed at 1; "
                "remove them and use n_jobs_vintages instead."
            )
        execution = _ExecutionConfig(
            n_jobs_vintages=n_jobs_vintages, show_tqdm=show_tqdm
        )
        periods = [period for period in self._periods() if period.targets]
        if not periods:
            raise ValueError("No active targets. Check target_schedule.")

        optimizers = self._fit_periods(
            periods=periods,
            name=name,
            models=models,
            run_options=kwargs,
            execution=execution,
        )
        tables = self._collect_results(periods, optimizers, name)
        self._store_results(
            name,
            tables,
            optimizers,
            store_correlations=kwargs.get("store_correlations", False),
        )

    def _fit_periods(
        self,
        periods: List[_TargetPeriod],
        name: str,
        models: Dict[str, BaseEstimator],
        run_options: Dict[str, Any],
        execution: _ExecutionConfig,
    ) -> List[SignalOptimizer]:
        """Collect worker results in schedule order, with one progress display."""
        # QDF's original-dtype flag is not preserved by process serialization.
        worker_df = pd.DataFrame(self.df)
        optimizers = []
        # Limit native thread pools as well as joblib's explicit worker counts.
        with parallel_config(backend="loky", inner_max_num_threads=1):
            with Parallel(
                n_jobs=execution.n_jobs_vintages, return_as="generator"
            ) as parallel, tqdm(
                total=len(periods),
                desc="Vintages",
                unit="vintage",
                disable=not execution.show_tqdm,
            ) as progress:
                results = parallel(
                    delayed(_fit_vintage)(
                        df=worker_df,
                        xcats=self.config.features + list(period.targets),
                        cids=self.cids,
                        n_targets=len(period.targets),
                        init_options=self._optimizer_options,
                        name=name,
                        models=models,
                        run_options=run_options,
                    )
                    for period in periods
                )
                for optimizer in results:
                    optimizer.df.InitializedAsCategorical = (
                        self.df.InitializedAsCategorical
                    )
                    optimizers.append(optimizer)
                    progress.update(1)
        return optimizers

    def _collect_results(
        self,
        periods: List[_TargetPeriod],
        optimizers: List[SignalOptimizer],
        name: str,
    ) -> Dict[str, pd.DataFrame]:
        """Slice each vintage's outputs without changing its stored diagnostics."""
        pieces = {table: [] for table in self._RESULT_TABLES}
        for period, optimizer in zip(periods, optimizers):
            for table in self._RESULT_TABLES:
                frame = getattr(optimizer, table)
                within_period = frame["real_date"].ge(period.start)
                if period.end is not None:
                    within_period &= frame["real_date"].lt(period.end)
                frame = frame.loc[within_period].copy()
                if table == "preds" and self.n_targets > 1 and len(period.targets) == 1:
                    frame["xcat"] = f"{period.targets[0]}_{name}"
                pieces[table].append(frame)
        return {
            table: pd.concat(frames, ignore_index=True)
            for table, frames in pieces.items()
        }

    def _store_results(
        self,
        name: str,
        tables: Dict[str, pd.DataFrame],
        optimizers: List[SignalOptimizer],
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
            categorical_columns = existing.select_dtypes("category").columns
            incoming = incoming.astype(
                {column: "category" for column in categorical_columns}
            )
            replacements[table] = concat_categorical(
                existing.loc[~belongs_to_process], incoming
            )
        regimes = self.get_regimes()

        for table, frame in replacements.items():
            setattr(self, table, frame)
        self.optimizers[name] = optimizers
        self._process_outputs[name] = output_names
        self.store_correlations = store_correlations
        self.regimes = regimes

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

    def get_optimized_signals(
        self, name: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """Get combined signals by process name or individual target output name."""
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
        """Return combined model choices and stored model attributes."""
        return self._get_result("chosen_models", name)

    def get_selected_features(
        self, name: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """Return combined feature-selection indicators."""
        return self._get_result("selected_ftrs", name)

    def get_feature_importances(
        self, name: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """Return combined feature importances, sorted by date for named processes."""
        frame = self._get_result("feature_importances", name)
        return frame if name is None else frame.sort_values("real_date")

    def get_intercepts(
        self, name: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """Return combined intercepts, sorted by date for named processes."""
        frame = self._get_result("intercepts", name)
        return frame if name is None else frame.sort_values("real_date")

    def get_feature_correlations(
        self, name: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """Return combined input/transformed-feature correlations when recorded."""
        return self._get_result("ftr_corr", name)
