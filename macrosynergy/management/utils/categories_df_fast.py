"""
A vectorised alternative to `categories_df`: reduce, pivot, downsample, aggregate, lag,
order the columns, drop the all-NaN rows.
"""

import operator
import warnings
from typing import Any, Dict, FrozenSet, Iterable, List, NamedTuple, Optional
from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils.core import _map_to_business_day_frequency

# `categories_df`'s own order, and the order the assertion message shows the caller.
CATEGORIES_DF_METRICS = ["value", "grading", "mop_lag", "eop_lag"]

NS_PER_DAY = 86_400_000_000_000

# `_dfw_row_positions` scans a `seen` mask over the whole row-id space while the
# space stays this small, and factorizes the observations otherwise.
MAX_ROW_IDS_PER_OBS = 24
MAX_ROW_IDS = 32_000_000

DEFAULT_ARGS: Dict[str, Any] = dict(
    xcats=None,
    cids=None,
    val="value",
    start=None,
    end=None,
    blacklist=None,
    years=None,
    freq="M",
    lag=0,
    fwin=1,
    xcat_aggs=["mean", "mean"],
)


class DroppedObs(NamedTuple):
    """The observations the union's cross sections dropped, in whole-frame codes."""

    cid_of_obs: np.ndarray
    xcat_of_obs: np.ndarray
    date_ns_of_obs: np.ndarray


class WideFrame(NamedTuple):
    """
    The dense ``(cid, real_date) x xcat`` `dfw`, built over the union of a batch's
    requests. Every field is settled at construction; a request never mutates it. A row
    is one row of `dfw`, i.e. one ``(cid, real_date)`` pair.

    Attributes
    ----------
    is_qdf : bool
        whether ``type(df) is QuantamentalDataFrame``, which is what selects between
        `reduce_df`'s two bodies. Deliberately narrower than an `isinstance` test.
    dfw_values : Dict[str, np.ndarray]
        per metric, an ``(n_xcats, n_rows)`` array of cells. A metric with no float
        column of its own is absent, and its requests fall back on `pivot`.
    periods : Dict[str, Tuple[np.ndarray, pd.DatetimeIndex]]
        per requested frequency, the period each date falls in and the end-of-period
        dates that label them.
    observed_cells : np.ndarray
        ``(n_xcats, n_rows)``, the cells `pivot` would emit.
    cid_of_row : np.ndarray
        per row, the position of its cross section on the cid axis.
    date_of_row : np.ndarray
        per row, the position of its date on the date axis.
    date_ns_of_row : np.ndarray
        per row, its date as int64 nanoseconds.
    cids : List[str]
        the cross sections on the cid axis, sorted.
    pos_of_cid : Dict[str, int]
        cross section to its cid axis position.
    pos_of_xcat : Dict[str, int]
        category to its column position in `dfw`.
    dfw_columns : pd.Index
        `dfw`'s column axis, one entry per category on the xcat axis.
    cid_index_level : pd.Index
        the outer level of the ``(cid, real_date)`` index, as object dtype.
    obs_dropped_by_cids : Optional[DroppedObs]
        the rows the union's cross sections dropped. Kept only while a category derived
        from them could still survive a QuantamentalDataFrame request.
    xcats_only_outside_cids : FrozenSet[str]
        the categories that appear nowhere inside the union's cross sections.
    code_of_cid : Dict[str, int]
        cross section to its code over the whole frame, the coding
        `obs_dropped_by_cids` speaks.
    code_of_xcat : Dict[str, int]
        category to its code over the whole frame.
    """

    is_qdf: bool
    dfw_values: Dict[str, np.ndarray]
    periods: Dict[str, Tuple[np.ndarray, pd.DatetimeIndex]]
    # the cells `pivot` emits, not `~isnan(values)`: an all-NaN daily row still forms a
    # group, and that group participates in the positional shift - H11
    observed_cells: np.ndarray
    cid_of_row: np.ndarray
    date_of_row: np.ndarray
    date_ns_of_row: np.ndarray
    cids: List[str]
    pos_of_cid: Dict[str, int]
    pos_of_xcat: Dict[str, int]
    dfw_columns: pd.Index
    cid_index_level: pd.Index
    # the QuantamentalDataFrame body derives the categories before the cid filter
    # (methods.py:364-370, vs the plain body at :385) - H1
    obs_dropped_by_cids: Optional[DroppedObs]
    xcats_only_outside_cids: FrozenSet[str]
    code_of_cid: Dict[str, int]
    code_of_xcat: Dict[str, int]


def _fill_default_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Fill one request's defaults, failing the way ``categories_df(df, **args)`` does."""

    unknown = sorted(set(args) - set(DEFAULT_ARGS))
    if unknown:
        raise TypeError(
            f"categories_df() got an unexpected keyword argument {unknown[0]!r}"
        )
    if "xcats" not in args:
        raise TypeError(
            "categories_df() missing 1 required positional argument: 'xcats'"
        )
    full_args = dict(DEFAULT_ARGS)
    full_args.update(args)
    # both `reduce_df` bodies promote a bare string, so nothing below has to
    if isinstance(full_args["cids"], str):
        full_args["cids"] = [full_args["cids"]]
    return full_args


def _check_categories_df_args(df: pd.DataFrame, args: Dict[str, Any]) -> None:
    """What `categories_df` checks before it reads a row; the order is observable."""

    args["_bday_freq"] = _map_to_business_day_frequency(args["freq"])

    xcats, xcat_aggs, years = args["xcats"], args["xcat_aggs"], args["years"]
    assert isinstance(xcats, list), f"<list> expected and not {type(xcats)}."
    assert all([isinstance(c, str) for c in xcats]), "List of categories expected."

    aggs_error = "List of strings, outlining the aggregation methods, expected."
    assert isinstance(xcat_aggs, list), aggs_error
    assert all([isinstance(a, str) for a in xcat_aggs]), aggs_error
    aggs_len = (
        "Only two aggregation methods required. The first will be used for all "
        "explanatory category(s)."
    )
    assert len(xcat_aggs) == 2, aggs_len

    assert not (years is not None) & (
        args["lag"] != 0
    ), "Lags cannot be applied to year groups."
    if years is not None:
        assert isinstance(args["start"], str), "Year aggregation requires a start date."

        no_xcats = (
            "If the data is aggregated over a multi-year timeframe, only two "
            "categories are permitted."
        )
        assert len(xcats) == 2, no_xcats

    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("Argument `df` must be a standardised Quantamental DataFrame.")


def _check_val_column(df: pd.DataFrame, val: str) -> None:
    """The value column check, which `categories_df` makes after `reduce_df` warns."""

    val_error = (
        "The column of interest must be one of the defined JPMaQS metrics, "
        f"{CATEGORIES_DF_METRICS}, but received {val}."
    )
    assert val in CATEGORIES_DF_METRICS, val_error
    avbl_cols = list(df.columns)
    assert val in avbl_cols, (
        f"The passed column name, {val}, must be present in the "
        f"received DataFrame. DataFrame contains {avbl_cols}."
    )


def _check_reduced_xcats_and_cids(
    xcats: List[str], cids: List[str], args: Dict[str, Any]
) -> None:
    """
    Raise or warn on what survived the reduction, as `categories_df` does.

    Parameters
    ----------
    xcats : List[str]
        the categories that survived the reduction, in requested order.
    cids : List[str]
        the cross sections that survived the reduction, sorted.
    args : Dict[str, Any]
        one request's arguments; `xcats` and `cids` are read as the requested lists.

    Raises
    ------
    ValueError
        if fewer than two categories, or no cross section, survived.

    Notes
    -----
    Both warning texts are byte-identical to `df_utils.py:1039-1041` and `:1049-1051`,
    and the cross-section one is skipped when no `cids` were requested - H5.
    """

    input_xcats, input_cids = args["xcats"], args["cids"]

    if len(xcats) < 2:
        # the trailing space is in the shipped message (df_utils.py:1036) - do not trim
        raise ValueError("The DataFrame must contain at least two categories. ")
    elif set(xcats) != set(input_xcats):
        missing_xcats = list(set(input_xcats) - set(xcats))
        warnings.warn(
            f"The following categories are missing from the DataFrame: {missing_xcats}"
        )

    if len(cids) < 1:
        # likewise trailing (df_utils.py:1044)
        raise ValueError(
            "The DataFrame must contain at least one valid cross section. "
        )
    elif input_cids and set(cids) != set(input_cids):
        missing_cids = list(set(input_cids) - set(cids))
        warnings.warn(
            f"The following cross sections are missing from the DataFrame: {missing_cids}"
        )


def _check_dfw_columns(xcats: List[str], dfw_columns: Iterable[str]) -> None:
    """
    Reject a surviving category the reshape gave no column of its own.

    A category derived before the cid filter can survive the reduction while `pivot`
    never makes it a column (methods.py:364-370); the shipped body then raises this
    `KeyError` from `df_utils.py:1099` - H1.
    """

    missing = [x for x in xcats if x not in dfw_columns]
    if missing:
        raise KeyError(f"Column not found: {missing[0]}")


def _check_blacklist(blacklist: Any) -> None:
    """`apply_blacklist`'s type guards, which only the QuantamentalDataFrame body runs."""

    if not isinstance(blacklist, dict):
        raise TypeError("`blacklist` must be a dictionary.")
    if not all([isinstance(k, str) for k in blacklist.keys()]):
        raise TypeError("Keys of `blacklist` must be strings.")
    if not all([isinstance(v, Iterable) for v in blacklist.values()]):
        raise TypeError("Values of `blacklist` must be iterables.")
    if not all(
        [isinstance(vv, (str, pd.Timestamp)) for v in blacklist.values() for vv in v]
    ) or any([len(v) != 2 for v in blacklist.values()]):
        raise TypeError(
            "Values of `blacklist` must be lists of start & end dates "
            "(str or pd.Timestamp)."
        )


def _reduce_df(
    df: pd.DataFrame, args: Dict[str, Any], is_qdf: bool
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    ``reduce_df(..., out_all=True)``, in whichever of its two bodies applies.

    The two genuinely disagree on the surviving categories, because the
    QuantamentalDataFrame body derives them before the cid filter (methods.py:364-370)
    and the plain body after it (:385) - H1.
    """

    xcats, cids = args["xcats"], args["cids"]
    start, end, blacklist = args["start"], args["end"], args["blacklist"]

    if is_qdf:
        if start:
            df = df[df["real_date"] >= pd.to_datetime(start)]
        if end:
            df = df[df["real_date"] <= pd.to_datetime(end)]
        if blacklist is not None:
            _check_blacklist(blacklist=blacklist)
            for key, value in blacklist.items():
                df = df[
                    ~(
                        (df["cid"] == key[:3])
                        & (df["real_date"] >= value[0])
                        & (df["real_date"] <= value[1])
                    )
                ]
            df = df.reset_index(drop=True)
        xcats_in_df = df["xcat"].unique()
        xcats = [xcat for xcat in xcats if xcat in xcats_in_df]
        df = df[df["xcat"].isin(xcats)]
        cids_in_df = df["cid"].unique()
        cids = (
            sorted(cids_in_df) if cids is None else [c for c in cids if c in cids_in_df]
        )
        df = df[df["cid"].isin(cids)].reset_index(drop=True)
        if all(df[c].dtype.name == "category" for c in ("cid", "xcat")):
            df = df.copy()
            df["cid"] = df["cid"].cat.remove_unused_categories().astype("category")
            df["xcat"] = df["xcat"].cat.remove_unused_categories().astype("category")
        return df.drop_duplicates().reset_index(drop=True), xcats, sorted(cids)

    df = df[df["xcat"].isin(xcats)]
    if cids is not None:
        df = df[df["cid"].isin(cids)]
    if start:
        df = df[df["real_date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["real_date"] <= pd.to_datetime(end)]
    if blacklist is not None:
        for key, value in blacklist.items():
            df = df[
                ~(
                    (df["cid"] == key[:3])
                    & (df["real_date"] >= pd.to_datetime(value[0]))
                    & (df["real_date"] <= pd.to_datetime(value[1]))
                )
            ]
    xcats_in_df = df["xcat"].unique()
    xcats = [xcat for xcat in xcats if xcat in xcats_in_df]
    cids_in_df = df["cid"].unique()
    cids = sorted(cids_in_df) if cids is None else [c for c in cids if c in cids_in_df]
    df = df[df["cid"].isin(cids)]
    return df.drop_duplicates(), xcats, sorted(cids)


def _blacklist_bound(value: Any, is_qdf: bool) -> int:
    """One blacklist bound as int64 nanoseconds, raising the way its body would."""

    # `apply_blacklist` compares the raw bound against a datetime64 column, so a bad
    # one is a TypeError; the plain body parses it first and raises DateParseError - H6.
    try:
        return pd.Timestamp(value).value
    except ValueError:
        if not is_qdf:
            raise
        raise TypeError(
            f"Invalid comparison between dtype=datetime64[ns] and {type(value).__name__}"
        ) from None


def _blacklist_ranges(
    args: Dict[str, Any], is_qdf: bool
) -> Tuple[Tuple[str, int, int], ...]:
    """Every blacklist entry as a ``(cid, low, high)`` triple of nanosecond bounds."""

    # An absent cross section is kept: both bodies evaluate every bound, so a
    # malformed one raises even where it matches no row.
    if args["blacklist"] is None:
        return ()
    if is_qdf:
        _check_blacklist(blacklist=args["blacklist"])
    return tuple(
        (
            key[:3],
            _blacklist_bound(value=value[0], is_qdf=is_qdf),
            _blacklist_bound(value=value[1], is_qdf=is_qdf),
        )
        for key, value in args["blacklist"].items()
    )


def _date_mask(
    args: Dict[str, Any],
    blacklist_ranges: Tuple[Tuple[str, int, int], ...],
    cid_codes: np.ndarray,
    date_ns: np.ndarray,
    code_of_cid: Dict[str, int],
) -> np.ndarray:
    """`start`, `end` and `blacklist` as a mask over any ``(cid, real_date)`` coding."""

    mask = np.ones(len(date_ns), dtype=bool)
    if args["start"]:
        mask &= date_ns >= pd.to_datetime(args["start"]).value
    if args["end"]:
        mask &= date_ns <= pd.to_datetime(args["end"]).value
    for cid, low, high in blacklist_ranges:
        code = code_of_cid.get(cid)
        if code is not None:
            mask &= ~((cid_codes == code) & (date_ns >= low) & (date_ns <= high))
    return mask


def _label_mask(
    labels: Iterable[str], code_of_label: Dict[str, int], n_codes: int
) -> np.ndarray:
    """A mask over label codes, one entry longer so a ``-1`` code gathers False."""

    mask = np.zeros(n_codes + 1, dtype=bool)
    for label in labels:
        code = code_of_label.get(label)
        if code is not None:
            mask[code] = True
    return mask


def _reduce_dfw_rows(
    dfw: "WideFrame", args: Dict[str, Any]
) -> Tuple[np.ndarray, List[str], List[str]]:
    """`reduce_df` as a mask over the reshape's rows, plus the lists it derives."""

    blacklist_ranges = _blacklist_ranges(args=args, is_qdf=dfw.is_qdf)
    date_mask = _date_mask(
        args=args,
        blacklist_ranges=blacklist_ranges,
        cid_codes=dfw.cid_of_row,
        date_ns=dfw.date_ns_of_row,
        code_of_cid=dfw.pos_of_cid,
    )
    # A row mask, not a period one. The forward window rolls the dependent series
    # without grouping by cid (df_utils.py:1103-1105), so cross sections bleed - H8.
    cid_mask = (
        None
        if args["cids"] is None
        else _label_mask(
            labels=args["cids"], code_of_label=dfw.pos_of_cid, n_codes=len(dfw.cids)
        )[dfw.cid_of_row]
    )
    # the QuantamentalDataFrame body derives the categories before the cid filter
    # (methods.py:364-370, vs the plain body at :385) - H1
    for_xcats = date_mask if dfw.is_qdf or cid_mask is None else (date_mask & cid_mask)
    # Off the observed cells, not the values: an all-NaN period is still a group, and
    # `dropna` removing its row does not stop `shift` sourcing from it - H11.
    xcat_survives = (dfw.observed_cells & for_xcats).any(axis=1)

    xcats_outside = frozenset()
    if dfw.obs_dropped_by_cids is not None:
        dropped = dfw.obs_dropped_by_cids
        kept = _date_mask(
            args=args,
            blacklist_ranges=blacklist_ranges,
            cid_codes=dropped.cid_of_obs,
            date_ns=dropped.date_ns_of_obs,
            code_of_cid=dfw.code_of_cid,
        )
        seen = np.zeros(len(dfw.code_of_xcat), dtype=bool)
        seen[dropped.xcat_of_obs[kept]] = True
        xcats_outside = frozenset(
            x for x in dfw.xcats_only_outside_cids if seen[dfw.code_of_xcat[x]]
        )

    xcats = [
        x
        for x in args["xcats"]
        if (x in dfw.pos_of_xcat and xcat_survives[dfw.pos_of_xcat[x]])
        or x in xcats_outside
    ]
    no_rows = np.zeros(dfw.observed_cells.shape[1], dtype=bool)
    if len(xcats) < 2:
        return no_rows, xcats, []

    xcat_cols = [
        dfw.pos_of_xcat[x] for x in dict.fromkeys(xcats) if x in dfw.pos_of_xcat
    ]
    rows = (
        (for_xcats & dfw.observed_cells[xcat_cols].any(axis=0))
        if xcat_cols
        else no_rows
    )
    if dfw.is_qdf and cid_mask is not None:
        rows = rows & cid_mask

    cid_survives = np.zeros(len(dfw.cids), dtype=bool)
    cid_survives[dfw.cid_of_row[rows]] = True
    cids = (
        sorted(c for i, c in enumerate(dfw.cids) if cid_survives[i])
        if args["cids"] is None
        else [
            c
            for c in args["cids"]
            if c in dfw.pos_of_cid and cid_survives[dfw.pos_of_cid[c]]
        ]
    )
    return rows, xcats, cids


def _pivot_dfw(reduced: pd.DataFrame, val: str) -> pd.DataFrame:
    """The ``(cid, real_date) x xcat`` frame, the way `categories_df` builds it."""

    return reduced.pivot(index=("cid", "real_date"), columns="xcat", values=val)


def _label_codes(col: pd.Series) -> Tuple[np.ndarray, List[str]]:
    """Integer codes for a label column, and its labels in the order `pivot` sorts to."""

    if isinstance(col.dtype, pd.CategoricalDtype):
        return col.cat.codes.to_numpy(), list(col.dtype.categories)
    codes, uniques = pd.factorize(col.to_numpy(), sort=True)
    return codes, list(uniques)


def _axis_positions(
    codes: np.ndarray, n_codes: int, pos_dtype: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """The codes one axis of the reshape uses, and the axis position of every code."""

    seen = np.zeros(n_codes, dtype=bool)
    seen[codes] = True
    axis = np.flatnonzero(seen)
    pos_of_code = np.zeros(n_codes, dtype=pos_dtype)
    pos_of_code[axis] = np.arange(len(axis))
    return axis, pos_of_code


def _dfw_row_positions(
    row_id: np.ndarray, n_row_ids: int, n_obs: int, pos_dtype: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """The row ids that occur, ascending, and the `dfw` row of every observation."""

    # linear in the id space here, linear in the observations below; both are exact
    if n_row_ids <= MAX_ROW_IDS_PER_OBS * n_obs and n_row_ids <= MAX_ROW_IDS:
        seen = np.zeros(n_row_ids, dtype=bool)
        seen[row_id] = True
        dfw_rows = np.flatnonzero(seen)
        if len(dfw_rows) == n_row_ids:
            return dfw_rows, row_id
        row_of_id = np.zeros(n_row_ids, dtype=pos_dtype)
        row_of_id[dfw_rows] = np.arange(len(dfw_rows))
        return dfw_rows, row_of_id[row_id]

    codes, uniques = pd.factorize(row_id)
    order = np.argsort(uniques)
    row_of_code = np.empty(len(uniques), dtype=pos_dtype)
    row_of_code[order] = np.arange(len(uniques))
    return uniques[order], row_of_code[codes]


def _dfw_metric_values(
    df: pd.DataFrame,
    metrics: Iterable[str],
    cell_of_obs: np.ndarray,
    obs_idx: np.ndarray,
    shape: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    """A dense ``(n_xcats, n_rows)`` array of cells per metric with a float column."""

    dfw_values = {}
    for metric in metrics:
        col = df[metric].to_numpy() if metric in df.columns else None
        # a NaN-observed_cells scatter cannot stand up an integer column; `pivot` serves those
        if col is None or col.dtype.kind != "f":
            continue
        cells = np.full(shape[0] * shape[1], np.nan, dtype=col.dtype)
        cells[cell_of_obs] = col.take(obs_idx)
        dfw_values[metric] = cells.reshape(shape)
    return dfw_values


def _can_build_dfw(df: pd.DataFrame) -> bool:
    """Whether a dense reshape can stand in for `pivot` on this frame at all - §5."""

    if str(df["real_date"].dtype) != "datetime64[ns]":
        return False  # a row id is int64 nanoseconds // a day
    dtype = df["cid"].dtype
    if type(df) is not QuantamentalDataFrame and isinstance(dtype, pd.CategoricalDtype):
        # `pivot` honours a categorical level's order only while every category is
        # used, and the plain `reduce_df` body leaves unused ones behind - §2, §5
        categories = list(dtype.categories)
        return categories == sorted(categories)
    return True


def _period_index(
    dates: pd.DatetimeIndex, freq: str
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """The downsample period each date falls in, and the end-of-period dates."""

    # Empty periods are numbered too, and the labels come from pandas' own resampler
    # so that `freq="B"` snaps weekend dates the way `categories_df` does - H9.
    counts = pd.Series(np.zeros(len(dates)), index=dates).resample(freq).size()
    period_of_date = np.repeat(np.arange(len(counts)), counts.to_numpy())
    return period_of_date, counts.index


def _build_dfw(
    df: pd.DataFrame, args_list: Sequence[Dict[str, Any]], is_qdf: bool
) -> Optional[WideFrame]:
    """
    Build the one reshape a batch shares, or None where only `pivot` can serve it.

    Parameters
    ----------
    df : pd.DataFrame
        the standardised JPMaQS DataFrame the whole batch is answered from.
    args_list : Sequence[Dict[str, Any]]
        one filled argument dict per frequency request in the batch.
    is_qdf : bool
        whether ``type(df) is QuantamentalDataFrame``.

    Returns
    -------
    Optional[WideFrame]
        the shared reshape, or None where the frame is one the dense route cannot
        reproduce exactly and the caller must fall back on `pivot` - §5.
    """

    cid_codes, df_cids = _label_codes(col=df["cid"])
    xcat_codes, df_xcats = _label_codes(col=df["xcat"])
    date_ns = df["real_date"].to_numpy().view("int64")
    code_of_cid = {c: i for i, c in enumerate(df_cids)}
    code_of_xcat = {x: i for i, x in enumerate(df_xcats)}

    # The batch's union. Dates and blacklists stay per request: they vary, and hoisting
    # one would drop a row another request needs.
    union_xcats = list(dict.fromkeys(x for a in args_list for x in a["xcats"]))
    union_cids = (
        None
        if any(a["cids"] is None for a in args_list)
        else list(dict.fromkeys(c for a in args_list for c in a["cids"]))
    )
    in_union = _label_mask(
        labels=union_xcats, code_of_label=code_of_xcat, n_codes=len(df_xcats)
    )[xcat_codes]
    obs_idx = np.flatnonzero(in_union)
    cid_of_obs = cid_codes.take(obs_idx)
    obs_dropped_by_cids = None
    if union_cids is not None:
        in_cids = _label_mask(
            labels=union_cids, code_of_label=code_of_cid, n_codes=len(df_cids)
        )[cid_of_obs]
        if is_qdf and not in_cids.all():
            dropped_idx = obs_idx[~in_cids]
            obs_dropped_by_cids = DroppedObs(
                cid_of_obs=cid_of_obs[~in_cids],
                xcat_of_obs=xcat_codes.take(dropped_idx),
                date_ns_of_obs=date_ns.take(dropped_idx),
            )
        obs_idx, cid_of_obs = obs_idx[in_cids], cid_of_obs[in_cids]
    xcat_of_obs, date_ns_of_obs = xcat_codes.take(obs_idx), date_ns.take(obs_idx)

    if len(obs_idx) == 0:
        return None
    if (cid_of_obs < 0).any():
        return None  # a NaN cross section, which the reduction raises on
    date_of_obs = date_ns_of_obs // NS_PER_DAY
    if (date_of_obs * NS_PER_DAY != date_ns_of_obs).any():
        return None  # an intraday timestamp moves every start, end and blacklist bound

    first_day = int(date_of_obs.min())
    dates_spanned = int(date_of_obs.max()) - first_day + 1
    date_of_obs -= first_day  # in place; the floor-divide already gave a fresh array
    n_obs = len(obs_idx)
    # Bounded above rather than measured, because the axes below need the width to
    # allocate; no realistic frame is near the int32 ceiling either way.
    pos_dtype = (
        np.int32
        if max(
            min(len(df_cids), n_obs) * min(dates_spanned, n_obs),
            min(len(df_xcats), n_obs) * n_obs,
        )
        < 2**31
        else np.int64
    )
    cid_axis, pos_of_cid_code = _axis_positions(
        codes=cid_of_obs, n_codes=len(df_cids), pos_dtype=pos_dtype
    )
    xcat_axis, pos_of_xcat_code = _axis_positions(
        codes=xcat_of_obs, n_codes=len(df_xcats), pos_dtype=pos_dtype
    )
    date_axis, pos_of_date_code = _axis_positions(
        codes=date_of_obs, n_codes=dates_spanned, pos_dtype=pos_dtype
    )
    n_cids, n_xcats, n_dates = len(cid_axis), len(xcat_axis), len(date_axis)

    row_id = pos_of_cid_code[cid_of_obs]  # the only allocation; the rest is in place
    row_id *= n_dates
    row_id += pos_of_date_code[date_of_obs]
    dfw_rows, row_of_obs = _dfw_row_positions(
        row_id=row_id, n_row_ids=n_cids * n_dates, n_obs=n_obs, pos_dtype=pos_dtype
    )
    n_rows = len(dfw_rows)

    # Column-major over the ``(xcat, row)`` output: the orientation every reduction
    # below wants, and the transpose of the F-ordered values pandas' kernels want.
    cell_of_obs = pos_of_xcat_code[xcat_of_obs]
    cell_of_obs *= n_rows
    cell_of_obs += row_of_obs

    # Two rows sharing (cid, real_date, xcat) land on one cell, so a fill count short
    # of the observation count is the duplicate `pivot` raises on (df_utils.py:1072) - H2
    observed_cells = np.zeros(n_xcats * n_rows, dtype=bool)
    observed_cells[cell_of_obs] = True
    if np.count_nonzero(observed_cells) != len(cell_of_obs):
        return None

    pos_of_xcat = {df_xcats[i]: j for j, i in enumerate(xcat_axis)}
    dfw_dates = pd.DatetimeIndex((date_axis + first_day) * NS_PER_DAY)

    if obs_dropped_by_cids is not None:
        # only a category found nowhere inside the union can change a derived list
        xcats_only_outside_cids = frozenset(
            df_xcats[i]
            for i in np.unique(obs_dropped_by_cids.xcat_of_obs)
            if df_xcats[i] not in pos_of_xcat
        )
        if not xcats_only_outside_cids:
            obs_dropped_by_cids = None
    else:
        xcats_only_outside_cids = frozenset()

    date_of_row = (dfw_rows % n_dates).astype(np.int32)
    cids = [df_cids[i] for i in cid_axis]
    return WideFrame(
        is_qdf=is_qdf,
        dfw_values=_dfw_metric_values(
            df=df,
            metrics={a["val"] for a in args_list},
            cell_of_obs=cell_of_obs,
            obs_idx=obs_idx,
            shape=(n_xcats, n_rows),
        ),
        periods={
            a["_bday_freq"]: _period_index(dates=dfw_dates, freq=a["_bday_freq"])
            for a in args_list
        },
        observed_cells=observed_cells.reshape(n_xcats, n_rows),
        cid_of_row=(dfw_rows // n_dates).astype(np.int32),
        date_of_row=date_of_row,
        date_ns_of_row=(date_axis[date_of_row] + first_day) * NS_PER_DAY,
        cids=cids,
        pos_of_cid={c: i for i, c in enumerate(cids)},
        pos_of_xcat=pos_of_xcat,
        dfw_columns=pd.Index(list(pos_of_xcat), name="xcat", dtype=object),
        cid_index_level=pd.Index(cids, dtype=object),
        obs_dropped_by_cids=obs_dropped_by_cids,
        xcats_only_outside_cids=xcats_only_outside_cids,
        code_of_cid=code_of_cid,
        code_of_xcat=code_of_xcat,
    )


def _downsample_dfw(dfw: pd.DataFrame, freq: str) -> pd.core.groupby.DataFrameGroupBy:
    """`dfw` grouped by cross section and downsample period, ready to aggregate."""

    return dfw.groupby(
        [pd.Grouper(level="cid"), pd.Grouper(level="real_date", freq=freq)],
        observed=True,  # unused cid categories would explode the groupby - H12
    )


def _downsample_dfw_arrays(
    dfw: WideFrame, rows: np.ndarray, metric: str, freq: str
) -> Tuple[pd.core.groupby.DataFrameGroupBy, pd.MultiIndex]:
    """
    `_downsample_dfw` over the shared reshape's arrays rather than over a pivot.

    Parameters
    ----------
    dfw : WideFrame
        the reshape the batch shares.
    rows : np.ndarray
        boolean mask over `dfw`'s rows, one request's surviving observations.
    metric : str
        the value column this request asked for.
    freq : str
        the business-day frequency to downsample to.

    Returns
    -------
    Tuple[pd.core.groupby.DataFrameGroupBy, pd.MultiIndex]
        the grouped frame, and the ``(cid, real_date)`` index its aggregation takes.
    """

    period_of_date, eop_dates = dfw.periods[freq]
    n_periods = len(eop_dates)
    # a request keeping every row is the single-call case, where a gather is all overhead
    row_idx = None if rows.all() else np.flatnonzero(rows)
    if row_idx is None:
        cid_of_row, date_of_row = dfw.cid_of_row, dfw.date_of_row
        cells = dfw.dfw_values[metric]
    else:
        cid_of_row = dfw.cid_of_row.take(row_idx)
        date_of_row = dfw.date_of_row.take(row_idx)
        # gather along the contiguous axis, so `.T` stays F-ordered
        cells = dfw.dfw_values[metric].take(row_idx, axis=1)

    # One integer per (cid, period). The rows are sorted by (cid, date), so each
    # group is one run of it and the distinct keys are a run-length scan.
    period_key = cid_of_row.astype(np.int64) * n_periods
    period_key += period_of_date[date_of_row]
    period_starts = np.ones(len(period_key), dtype=bool)
    np.not_equal(period_key[1:], period_key[:-1], out=period_starts[1:])
    distinct_keys = period_key[period_starts]

    downsampled_index = pd.MultiIndex(
        levels=[dfw.cid_index_level, eop_dates],
        codes=[distinct_keys // n_periods, distinct_keys % n_periods],
        names=["cid", "real_date"],
        verify_integrity=False,
    )
    return (
        pd.DataFrame(cells.T, columns=dfw.dfw_columns, copy=False).groupby(
            np.cumsum(period_starts) - 1, sort=False
        ),
        downsampled_index,
    )


def _aggregate_xcats(
    grouped: pd.core.groupby.DataFrameGroupBy, xcats: List[str], agg_method: str
) -> pd.DataFrame:
    """Aggregate the given categories over each downsample period."""

    if agg_method == "sum":
        dfw_agg = grouped[xcats].sum(min_count=1)  # `min_count` keeps a NaN period NaN
    else:
        # every aggregation but "sum" is cast down (df_utils.py:920-923) - H3
        dfw_agg = grouped[xcats].agg(agg_method).astype(dtype=np.float32)
    if isinstance(dfw_agg, pd.Series):
        # "size" reduces across the columns rather than within each one; broadcast back
        dfw_agg = pd.DataFrame({x: dfw_agg for x in xcats})
    return dfw_agg


def _lag_explanatory(dfw_explanatory: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Shift the explanatory columns by `lag` periods, within each cross section."""

    # `df_utils.py:925` guards on `lag > 0`, so a non-positive lag is a no-op - H7
    if lag <= 0:
        return dfw_explanatory
    return dfw_explanatory.groupby(level=0, observed=True).shift(lag)


def _build_dfc(
    dfw_explanatory: pd.DataFrame, dep_col: pd.Series, xcats: List[str], fwin: int
) -> pd.DataFrame:
    """Order the columns so that the dependent category is the right-most one."""

    dep = xcats[-1]
    if fwin > 1:
        # not grouped by cid (df_utils.py:1103-1105), so the last periods of each
        # cross section are averaged with the first of the next - H8
        dep_col = dep_col.rolling(window=fwin).mean().shift(1 - fwin)
    dfw_explanatory.index.names = ["cid", "real_date"]
    # assigned last: a category used as both explanatory and dependent holds this
    dfw_explanatory[dep] = dep_col
    columns = xcats[:-1] + [dep]
    dfc = dfw_explanatory[columns]
    # the shipped body assigns columns one at a time into an empty frame
    # (df_utils.py:918-932), which leaves a plain unnamed object axis - H4
    dfc.columns = pd.Index(columns, dtype=object)
    return dfc


def _cast_cid_index_to_object(dfc: pd.DataFrame) -> pd.DataFrame:
    """Cast a categorical `cid` index level back to object."""

    if dfc.index.dtypes["cid"].name == "category":
        new_outer_index = dfc.index.levels[0].astype("object")
        new_index = pd.MultiIndex(
            levels=[new_outer_index, dfc.index.levels[1]],
            codes=dfc.index.codes,
            names=dfc.index.names,
        )
        dfc.index = new_index
    return dfc


def _categories_df_via_pivot(
    df: pd.DataFrame, args: Dict[str, Any], is_qdf: bool
) -> pd.DataFrame:
    """One frequency request through `pivot`; serves every frame, at every shape."""

    reduced, xcats, cids = _reduce_df(df=df, args=args, is_qdf=is_qdf)
    _check_reduced_xcats_and_cids(xcats=xcats, cids=cids, args=args)
    _check_val_column(df=df, val=args["val"])

    dfw = _pivot_dfw(reduced=reduced, val=args["val"])
    _check_dfw_columns(xcats=xcats, dfw_columns=dfw.columns)

    dfw = _downsample_dfw(dfw=dfw, freq=args["_bday_freq"])
    xcat_aggs = args["xcat_aggs"]
    dfw_explanatory = _aggregate_xcats(
        grouped=dfw, xcats=list(dict.fromkeys(xcats[:-1])), agg_method=xcat_aggs[0]
    )
    dep_col = _aggregate_xcats(grouped=dfw, xcats=[xcats[-1]], agg_method=xcat_aggs[1])[
        xcats[-1]
    ]
    dfw_explanatory = _lag_explanatory(dfw_explanatory=dfw_explanatory, lag=args["lag"])
    dfc: pd.DataFrame = _build_dfc(
        dfw_explanatory=dfw_explanatory,
        dep_col=dep_col,
        xcats=xcats,
        fwin=args["fwin"],
    )
    # `categories_df` is a support function: a category that is NaN for a period is
    # left for the caller to handle, and only an entirely empty row is dropped.
    return _cast_cid_index_to_object(dfc=dfc).dropna(axis=0, how="all")


def _categories_df_via_dfw(
    df: pd.DataFrame, dfw: WideFrame, args: Dict[str, Any]
) -> pd.DataFrame:
    """The same steps as `_categories_df_via_pivot`, off the reshape the batch shares."""

    rows, xcats, cids = _reduce_dfw_rows(dfw=dfw, args=args)
    _check_reduced_xcats_and_cids(xcats=xcats, cids=cids, args=args)
    _check_val_column(df=df, val=args["val"])
    xcat_survives = (dfw.observed_cells & rows).any(axis=1)
    _check_dfw_columns(
        xcats=xcats,
        dfw_columns={x for x, i in dfw.pos_of_xcat.items() if xcat_survives[i]},
    )

    grouped, downsampled_index = _downsample_dfw_arrays(
        dfw=dfw, rows=rows, metric=args["val"], freq=args["_bday_freq"]
    )
    xcat_aggs = args["xcat_aggs"]
    dfw_explanatory = _aggregate_xcats(
        grouped=grouped, xcats=list(dict.fromkeys(xcats[:-1])), agg_method=xcat_aggs[0]
    )
    dep_frame = _aggregate_xcats(
        grouped=grouped, xcats=[xcats[-1]], agg_method=xcat_aggs[1]
    )
    dfw_explanatory.index = dep_frame.index = downsampled_index
    dfw_explanatory = _lag_explanatory(dfw_explanatory=dfw_explanatory, lag=args["lag"])
    dfc: pd.DataFrame = _build_dfc(
        dfw_explanatory=dfw_explanatory,
        dep_col=dep_frame[xcats[-1]],
        xcats=xcats,
        fwin=args["fwin"],
    )
    return _cast_cid_index_to_object(dfc=dfc).dropna(axis=0, how="all")


def _year_group_labels(start_year: int, end_year: int, years: int) -> List[str]:
    """Year-group labels spanning `start_year` to `end_year`, the last open-ended."""

    grouping = int((end_year - start_year) / years)
    operator.index(years)  # a non-integer `years` raises from `range` first - H14
    list_y_groups, group_start_year = [], start_year
    for _ in range(grouping):
        list_y_groups.append(f"{group_start_year} - {group_start_year + (years - 1)}")
        group_start_year += years
    list_y_groups.append(f"{group_start_year} - now")
    return list_y_groups


def _categories_df_by_year_groups(
    df: pd.DataFrame, args: Dict[str, Any], is_qdf: bool
) -> pd.DataFrame:
    """
    One multi-year request: two categories aggregated over fixed year groups.

    A separate pipeline in the shipped function too (df_utils.py:1112-1142), and every
    difference shows: string period labels, float64 values, sorted columns named
    'xcat', and `numeric_only=True` making "size", "count" and "nunique" raise - H14.
    """

    reduced, xcats, cids = _reduce_df(df=df, args=args, is_qdf=is_qdf)
    _check_reduced_xcats_and_cids(xcats=xcats, cids=cids, args=args)
    _check_val_column(df=df, val=args["val"])

    val, years = args["val"], args["years"]
    start_year = pd.to_datetime(args["start"]).year
    list_y_groups = _year_group_labels(
        start_year=start_year, end_year=reduced["real_date"].max().year + 1, years=years
    )

    # `year % start_year` is `categories_df`'s own group lookup, kept on Python ints
    # so that a `start_year` of 0 raises rather than returning 0 - H14
    year_of_obs = reduced["real_date"].dt.year.to_numpy()
    distinct_years, inverse = np.unique(year_of_obs, return_inverse=True)
    group_of_year = np.array(
        [list_y_groups[int((int(y) % start_year) / years)] for y in distinct_years],
        dtype=object,
    )

    reduced = reduced.copy()
    reduced["custom_date"] = group_of_year[inverse]

    col_names = ["cid", "xcat", "real_date", val]
    df_output = []
    for xcat, agg_method in zip(xcats, args["xcat_aggs"]):
        dfx = (
            reduced[reduced["xcat"] == xcat]
            .groupby(["xcat", "cid", "custom_date"], observed=True)[[val]]
            # `numeric_only` is what makes "size", "count" and "nunique" raise here
            .aggregate(agg_method, numeric_only=True)
            .reset_index()
            .rename(columns={"custom_date": "real_date"})
        )
        df_output.append(dfx[col_names])

    dfc = pd.concat(df_output).pivot(
        index=("cid", "real_date"), columns="xcat", values=val
    )
    return _cast_cid_index_to_object(dfc=dfc).dropna(axis=0, how="all")


def _categories_df_many(
    df: pd.DataFrame,
    args_list: Sequence[Optional[Dict[str, Any]]],
) -> List[Any]:
    """One result per request, in order; a raising request yields its exception."""

    results: List[Any] = [None] * len(args_list)
    is_qdf = type(df) is QuantamentalDataFrame
    freq_idx: List[int] = []
    for i, args in enumerate(args_list):
        if args is None:
            continue
        try:
            _check_categories_df_args(df=df, args=args)
            if args["years"] is not None:
                results[i] = _categories_df_by_year_groups(
                    df=df, args=args, is_qdf=is_qdf
                )
            else:
                freq_idx.append(i)
        except Exception as e:  # the exception IS this request's result
            results[i] = e
    if not freq_idx:
        return results

    dfw = (
        _build_dfw(df=df, args_list=[args_list[i] for i in freq_idx], is_qdf=is_qdf)
        if _can_build_dfw(df=df)
        else None
    )
    for i in freq_idx:
        args = args_list[i]
        try:
            if dfw is not None and args["val"] in dfw.dfw_values:
                results[i] = _categories_df_via_dfw(df=df, dfw=dfw, args=args)
            else:
                results[i] = _categories_df_via_pivot(df=df, args=args, is_qdf=is_qdf)
        except Exception as e:
            results[i] = e
    return results


def categories_df_fast(
    df: pd.DataFrame,
    xcats: List[str],
    cids: List[str] = None,
    val: str = "value",
    start: str = None,
    end: str = None,
    blacklist: dict = None,
    years: int = None,
    freq: str = "M",
    lag: int = 0,
    fwin: int = 1,
    xcat_aggs: List[str] = ["mean", "mean"],
) -> pd.DataFrame:
    """
    Create a custom two-categories DataFrame with appropriate frequency and, if
    applicable, lags.

    Parameters
    ----------
    df : pd.DataFrame
        standardized JPMaQS DataFrame with the following necessary columns: 'cid',
        'xcat', 'real_date' and at least one column with values of interest.
    xcats : List[str]
        extended categories involved in the custom DataFrame. The last category in the
        list represents the dependent variable, and the (n - 1) preceding categories
        will be the explanatory variables(s).
    cids : List[str]
        cross-sections to be included. Default is all in the DataFrame.
    val : str
        name of column that contains the values of interest. Default is 'value'.
    start : str
        earliest date in ISO 8601 format. Default is None, i.e. earliest date in
        DataFrame is used.
    end : str
        latest date in ISO 8601 format. Default is None, i.e. latest date in DataFrame
        is used.
    blacklist : dict
        cross-sections with date ranges that should be excluded from the DataFrame. If
        one cross section has several blacklist periods append numbers to the cross
        section code.
    years : int
        number of years over which data are aggregated. Supersedes the "freq"
        parameter and does not allow lags, Default is None, i.e. no multi-year
        aggregation.
    freq : str
        letter denoting frequency at which the series are to be sampled. This must be
        one of 'D', 'W', 'M', 'Q', 'A'. Default is 'M'. Will always be the last
        business day of the respective frequency.
    lag : int
        lag (delay of arrival) of explanatory category(s) in periods as set by freq.
        Default is 0.
    fwin : int
        forward moving average window of first category. Default is 1, i.e no average.
    xcat_aggs : List[str]
        exactly two aggregation methods. Default is 'mean' for both. The same
        aggregation method, the first method in the parameter, will be used for all
        explanatory variables.

    Returns
    -------
    pd.DataFrame
        custom DataFrame indexed by ``(cid, real_date)``, with one column per category
        in the order of `xcats` and the dependent variable right-most.

    Notes
    -----
    An alternative to `categories_df`, returning the same result for the same arguments
    - values, dtypes, warnings and exceptions alike - by way of a reshape indexed by
    integer codes rather than `pivot`. Use `categories_df_fast_loop` when several
    requests share one DataFrame; they then share the reshape too.
    """

    args = {
        "xcats": xcats,
        "cids": cids,
        "val": val,
        "start": start,
        "end": end,
        "blacklist": blacklist,
        "years": years,
        "freq": freq,
        "lag": lag,
        "fwin": fwin,
        "xcat_aggs": xcat_aggs,
    }
    result = _categories_df_many(
        df=df,
        args_list=[_fill_default_args(args=args)],
    )[0]
    if isinstance(result, BaseException):
        raise result
    return result


def categories_df_fast_loop(
    df: pd.DataFrame, arg_batches: Iterable[Dict[str, Any]]
) -> List[Union[pd.DataFrame, Exception]]:
    """
    Run many `categories_df_fast` requests over one DataFrame, sharing the reshape.

    Parameters
    ----------
    df : pd.DataFrame
        standardized JPMaQS DataFrame with the following necessary columns: 'cid',
        'xcat', 'real_date' and at least one column with values of interest.
    arg_batches : Iterable[Dict[str, Any]]
        one dict of `categories_df_fast` keyword arguments per requested result.
        `xcats` is required; every other key takes its usual default, and an unknown
        key raises the `TypeError` the call itself would raise.

    Returns
    -------
    List[Union[pd.DataFrame, Exception]]
        one entry per request, in order. Each entry is what ``categories_df_fast(df,
        **args)`` returns, or - where that call would raise - the exception it would
        raise, returned rather than thrown.

    Notes
    -----
    Exceptions are returned positionally because a sweep routinely contains a request
    that legitimately fails - a target with no data for the requested cross sections,
    a category missing from this vintage - and throwing would discard the results
    already computed. Callers wanting the throwing behaviour test each entry with
    ``isinstance(r, Exception)``.
    """

    args_list: List[Optional[Dict[str, Any]]] = []
    results: List[Any] = []
    for args in arg_batches:
        try:
            args_list.append(_fill_default_args(args=args))
            results.append(None)
        except Exception as e:  # an unusable request is its own result
            args_list.append(None)
            results.append(e)
    for i, result in enumerate(_categories_df_many(df=df, args_list=args_list)):
        if args_list[i] is not None:
            results[i] = result
    return results
