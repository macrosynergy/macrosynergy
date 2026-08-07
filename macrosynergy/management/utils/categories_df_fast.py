"""
A vectorised alternative to the `categories_df` reshape.

`categories_df_fast` answers one request; `categories_df_fast_loop` answers a batch of
them off a single reshape. Both walk the same seven steps as `categories_df` - reduce,
reshape, downsample, lag the explanatory categories, aggregate the dependent one,
assemble, drop empty rows - and return the same frame for the same arguments.

The module is laid out by those steps, and the first two have two implementations each.
`_reduce_df` and `_pivot_dfw` are the shipped `reduce_df` and `pivot` flow
transcribed, and serve whatever the vectorised reshape cannot reproduce exactly.
`_build_dfw` scatters the frame into one dense ``(cid, real_date) x xcat`` `dfw`
indexed by integer codes, and `_reduce_dfw_rows` turns a request into a mask over that
`dfw`'s rows, so a batch shares the one reshape. Year groups are a separate pipeline in
the shipped function too, and keep their own runner here.

The parity contract - including the quirks reproduced deliberately - is in
CATEGORIES_DF_PARITY.md at the root of the repository.
"""

import operator
import warnings
from typing import Any, Dict, FrozenSet, Iterable, List, NamedTuple, Optional
from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils.core import _map_to_business_day_frequency

__all__ = ["categories_df_fast", "categories_df_fast_loop"]

# `categories_df`'s own order, and the order the assertion message shows the caller.
CATEGORIES_DF_METRICS = ["value", "grading", "mop_lag", "eop_lag"]

NS_PER_DAY = 86_400_000_000_000

# `_dfw_row_positions` scans a bitmap of the row-id space while it stays this small.
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


# ------------------------------------------------------------------- the arguments


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

    xcats, aggs, years = args["xcats"], args["xcat_aggs"], args["years"]
    assert isinstance(xcats, list), f"<list> expected and not {type(xcats)}."
    assert all([isinstance(c, str) for c in xcats]), "List of categories expected."

    aggs_error = "List of strings, outlining the aggregation methods, expected."
    assert isinstance(aggs, list), aggs_error
    assert all([isinstance(a, str) for a in aggs]), aggs_error
    assert len(aggs) == 2, (
        "Only two aggregation methods required. The first will be used for all "
        "explanatory category(s)."
    )

    assert not (years is not None) & (
        args["lag"] != 0
    ), "Lags cannot be applied to year groups."
    if years is not None:
        assert isinstance(args["start"], str), "Year aggregation requires a start date."
        assert len(xcats) == 2, (
            "If the data is aggregated over a multi-year timeframe, only two "
            "categories are permitted."
        )

    if not isinstance(df, QuantamentalDataFrame):
        raise TypeError("Argument `df` must be a standardised Quantamental DataFrame.")


def _check_val_column(df: pd.DataFrame, val: str) -> None:
    """The value column check, which `categories_df` makes after `reduce_df` warns."""

    assert val in CATEGORIES_DF_METRICS, (
        "The column of interest must be one of the defined JPMaQS metrics, "
        f"{CATEGORIES_DF_METRICS}, but received {val}."
    )
    avbl_cols = list(df.columns)
    assert val in avbl_cols, (
        f"The passed column name, {val}, must be present in the "
        f"received DataFrame. DataFrame contains {avbl_cols}."
    )


def _check_reduced_xcats_and_cids(
    out_xcats: List[str], out_cids: List[str], args: Dict[str, Any]
) -> None:
    """Raise or warn on what survived the reduction, as `categories_df` does - H5."""

    if len(out_xcats) < 2:
        raise ValueError("The DataFrame must contain at least two categories. ")
    elif set(out_xcats) != set(args["xcats"]):
        missing_xcats = list(set(args["xcats"]) - set(out_xcats))
        warnings.warn(
            f"The following categories are missing from the DataFrame: {missing_xcats}"
        )

    if len(out_cids) < 1:
        raise ValueError(
            "The DataFrame must contain at least one valid cross section. "
        )
    elif args["cids"] and set(out_cids) != set(args["cids"]):
        missing_cids = list(set(args["cids"]) - set(out_cids))
        warnings.warn(
            f"The following cross sections are missing from the DataFrame: {missing_cids}"
        )


def _check_dfw_columns(out_xcats: List[str], dfw_columns: Iterable[str]) -> None:
    """Reject a surviving category the reshape gave no column of its own - H1."""

    missing = [x for x in out_xcats if x not in dfw_columns]
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


# ------------------------------------------------------------------------- reduce


def _reduce_df(
    df: pd.DataFrame, args: Dict[str, Any], is_qdf: bool
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """``reduce_df(..., out_all=True)``, in whichever of its two bodies applies - H1."""

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
    # `apply_blacklist` compares the raw bound, the plain body parses it first - H6.
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
    bl_ranges: Tuple[Tuple[str, int, int], ...],
    cid_codes: np.ndarray,
    nanos: np.ndarray,
    code_of_cid: Dict[str, int],
) -> np.ndarray:
    """`start`, `end` and `blacklist` as a mask over any ``(cid, real_date)`` coding."""

    mask = np.ones(len(nanos), dtype=bool)
    if args["start"]:
        mask &= nanos >= pd.to_datetime(args["start"]).value
    if args["end"]:
        mask &= nanos <= pd.to_datetime(args["end"]).value
    for cid, low, high in bl_ranges:
        code = code_of_cid.get(cid)
        if code is not None:
            mask &= ~((cid_codes == code) & (nanos >= low) & (nanos <= high))
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
    wide: "WideFrame", args: Dict[str, Any]
) -> Tuple[np.ndarray, List[str], List[str]]:
    """`reduce_df` as a mask over the reshape's rows, plus the lists it derives."""

    bl_ranges = _blacklist_ranges(args=args, is_qdf=wide.is_qdf)
    by_date = _date_mask(
        args=args,
        bl_ranges=bl_ranges,
        cid_codes=wide.cid_of_row,
        nanos=wide.nanos_of_row,
        code_of_cid=wide.cid_pos,
    )
    # a row mask, not a period one: the forward window is not grouped by cid - H8
    by_cid = (
        None
        if args["cids"] is None
        else _label_mask(
            labels=args["cids"], code_of_label=wide.cid_pos, n_codes=len(wide.cids)
        )[wide.cid_of_row]
    )
    # the QuantamentalDataFrame body derives the categories before the cid filter - H1
    for_xcats = by_date if wide.is_qdf or by_cid is None else (by_date & by_cid)
    # off the filled cells, not the values: an all-NaN period still moves `shift` - §2
    xcat_survives = (wide.filled & for_xcats).any(axis=1)

    xcats_outside = frozenset()
    if wide.obs_outside_cids is not None:
        cid_of_obs, xcat_of_obs, nanos_of_obs = wide.obs_outside_cids
        kept = _date_mask(
            args=args,
            bl_ranges=bl_ranges,
            cid_codes=cid_of_obs,
            nanos=nanos_of_obs,
            code_of_cid=wide.df_cid_code,
        )
        seen = np.zeros(len(wide.df_xcat_code), dtype=bool)
        seen[xcat_of_obs[kept]] = True
        xcats_outside = frozenset(
            x for x in wide.xcats_outside_cids if seen[wide.df_xcat_code[x]]
        )

    out_xcats = [
        x
        for x in args["xcats"]
        if (x in wide.xcat_pos and xcat_survives[wide.xcat_pos[x]])
        or x in xcats_outside
    ]
    no_rows = np.zeros(wide.filled.shape[1], dtype=bool)
    if len(out_xcats) < 2:
        return no_rows, out_xcats, []

    xcat_cols = [
        wide.xcat_pos[x] for x in dict.fromkeys(out_xcats) if x in wide.xcat_pos
    ]
    rows = (for_xcats & wide.filled[xcat_cols].any(axis=0)) if xcat_cols else no_rows
    if wide.is_qdf and by_cid is not None:
        rows = rows & by_cid

    cid_survives = np.zeros(len(wide.cids), dtype=bool)
    cid_survives[wide.cid_of_row[rows]] = True
    out_cids = (
        sorted(c for i, c in enumerate(wide.cids) if cid_survives[i])
        if args["cids"] is None
        else [
            c
            for c in args["cids"]
            if c in wide.cid_pos and cid_survives[wide.cid_pos[c]]
        ]
    )
    return rows, out_xcats, out_cids


# ------------------------------------------------------------------------ reshape


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
    """The row ids that occur, ascending, and the wide row of every observation."""

    # linear in the id space here, linear in the observations below; both are exact
    if n_row_ids <= MAX_ROW_IDS_PER_OBS * n_obs and n_row_ids <= MAX_ROW_IDS:
        seen = np.zeros(n_row_ids, dtype=bool)
        seen[row_id] = True
        wide_rows = np.flatnonzero(seen)
        if len(wide_rows) == n_row_ids:
            return wide_rows, row_id
        row_of_id = np.zeros(n_row_ids, dtype=pos_dtype)
        row_of_id[wide_rows] = np.arange(len(wide_rows))
        return wide_rows, row_of_id[row_id]

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

    value_arrs = {}
    for metric in metrics:
        col = df[metric].to_numpy() if metric in df.columns else None
        # a NaN-filled scatter cannot stand up an integer column; `pivot` serves those
        if col is None or col.dtype.kind != "f":
            continue
        cells = np.full(shape[0] * shape[1], np.nan, dtype=col.dtype)
        cells[cell_of_obs] = col.take(obs_idx)
        value_arrs[metric] = cells.reshape(shape)
    return value_arrs


def _can_build_dfw(df: pd.DataFrame) -> bool:
    if str(df["real_date"].dtype) != "datetime64[ns]":
        return False  # a row id is int64 nanoseconds // a day
    dtype = df["cid"].dtype
    if type(df) is not QuantamentalDataFrame and isinstance(dtype, pd.CategoricalDtype):
        # the plain `reduce_df` body leaves unused categories behind - §2, §5
        categories = list(dtype.categories)
        return categories == sorted(categories)
    return True


class WideFrame(NamedTuple):
    """The dense ``(cid, real_date) x xcat`` `dfw`, built over a batch's union.

    Every field is settled at construction; a request never mutates the reshape.
    """

    df: pd.DataFrame
    is_qdf: bool
    # (n_xcats, n_rows) cells per requested metric; one without a float column is absent
    value_arrs: Dict[str, np.ndarray]
    # per requested frequency: (period_of_date, period_dates)
    periods: Dict[str, Tuple[np.ndarray, pd.DatetimeIndex]]
    # (n_xcats, n_rows): the cells `pivot` would emit - not `~isnan(values)`, H11
    filled: np.ndarray
    # per row of the reshape: cid axis position, date axis position, and the date
    cid_of_row: np.ndarray
    date_of_row: np.ndarray
    nanos_of_row: np.ndarray
    cids: List[str]
    cid_pos: Dict[str, int]
    xcat_pos: Dict[str, int]
    columns: pd.Index
    cid_level: pd.Index
    # rows the union's cross sections dropped, kept only while a category derived from
    # them could still survive a QuantamentalDataFrame request - H1
    obs_outside_cids: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    xcats_outside_cids: FrozenSet[str]
    # label -> code over the whole frame, the coding `obs_outside_cids` speaks
    df_cid_code: Dict[str, int]
    df_xcat_code: Dict[str, int]


def _build_dfw(
    df: pd.DataFrame, arg_batches: Sequence[Dict[str, Any]], is_qdf: bool
) -> Optional[WideFrame]:
    """The one reshape a batch shares, or None where only `pivot` can serve it - §5."""

    cid_codes, df_cids = _label_codes(col=df["cid"])
    xcat_codes, df_xcats = _label_codes(col=df["xcat"])
    nanos = df["real_date"].to_numpy().view("int64")
    df_cid_code = {c: i for i, c in enumerate(df_cids)}
    df_xcat_code = {x: i for i, x in enumerate(df_xcats)}

    # The batch's union. Dates and blacklists stay per request: they vary, and hoisting
    # one would drop a row another request needs.
    batch_xcats = list(dict.fromkeys(x for a in arg_batches for x in a["xcats"]))
    batch_cids = (
        None
        if any(a["cids"] is None for a in arg_batches)
        else list(dict.fromkeys(c for a in arg_batches for c in a["cids"]))
    )
    in_batch = _label_mask(
        labels=batch_xcats, code_of_label=df_xcat_code, n_codes=len(df_xcats)
    )[xcat_codes]
    obs_idx = np.flatnonzero(in_batch)
    cid_of_obs = cid_codes.take(obs_idx)
    obs_outside_cids = None
    if batch_cids is not None:
        in_cids = _label_mask(
            labels=batch_cids, code_of_label=df_cid_code, n_codes=len(df_cids)
        )[cid_of_obs]
        if is_qdf and not in_cids.all():
            dropped_idx = obs_idx[~in_cids]
            obs_outside_cids = (
                cid_of_obs[~in_cids],
                xcat_codes.take(dropped_idx),
                nanos.take(dropped_idx),
            )
        obs_idx, cid_of_obs = obs_idx[in_cids], cid_of_obs[in_cids]
    xcat_of_obs, nanos_of_obs = xcat_codes.take(obs_idx), nanos.take(obs_idx)

    if len(obs_idx) == 0:
        return None
    if (cid_of_obs < 0).any():
        return None  # a NaN cross section, which the reduction raises on
    date_of_obs = nanos_of_obs // NS_PER_DAY
    if (date_of_obs * NS_PER_DAY != nanos_of_obs).any():
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
    cid_axis, cid_pos_of_code = _axis_positions(
        codes=cid_of_obs, n_codes=len(df_cids), pos_dtype=pos_dtype
    )
    xcat_axis, xcat_pos_of_code = _axis_positions(
        codes=xcat_of_obs, n_codes=len(df_xcats), pos_dtype=pos_dtype
    )
    date_axis, date_pos_of_code = _axis_positions(
        codes=date_of_obs, n_codes=dates_spanned, pos_dtype=pos_dtype
    )
    n_cids, n_xcats, n_dates = len(cid_axis), len(xcat_axis), len(date_axis)

    row_id = cid_pos_of_code[cid_of_obs]  # the only allocation; the rest is in place
    row_id *= n_dates
    row_id += date_pos_of_code[date_of_obs]
    wide_rows, row_of_obs = _dfw_row_positions(
        row_id=row_id, n_row_ids=n_cids * n_dates, n_obs=n_obs, pos_dtype=pos_dtype
    )
    n_rows = len(wide_rows)

    # Column-major over the ``(xcat, row)`` output: the orientation every reduction
    # below wants, and the transpose of the F-ordered values pandas' kernels want.
    cell_of_obs = xcat_pos_of_code[xcat_of_obs]
    cell_of_obs *= n_rows
    cell_of_obs += row_of_obs

    # duplicate (cid, real_date, xcat) keys collide on one cell - H2
    filled = np.zeros(n_xcats * n_rows, dtype=bool)
    filled[cell_of_obs] = True
    if np.count_nonzero(filled) != len(cell_of_obs):
        return None

    xcat_pos = {df_xcats[i]: j for j, i in enumerate(xcat_axis)}
    wide_dates = pd.DatetimeIndex((date_axis + first_day) * NS_PER_DAY)

    if obs_outside_cids is not None:
        # only a category found nowhere inside the union can change a derived list
        xcats_outside_cids = frozenset(
            df_xcats[i]
            for i in np.unique(obs_outside_cids[1])
            if df_xcats[i] not in xcat_pos
        )
        if not xcats_outside_cids:
            obs_outside_cids = None
    else:
        xcats_outside_cids = frozenset()

    date_of_row = (wide_rows % n_dates).astype(np.int32)
    cids = [df_cids[i] for i in cid_axis]
    return WideFrame(
        df=df,
        is_qdf=is_qdf,
        value_arrs=_dfw_metric_values(
            df=df,
            metrics={a["val"] for a in arg_batches},
            cell_of_obs=cell_of_obs,
            obs_idx=obs_idx,
            shape=(n_xcats, n_rows),
        ),
        periods={
            a["_bday_freq"]: _period_index(dates=wide_dates, freq=a["_bday_freq"])
            for a in arg_batches
        },
        filled=filled.reshape(n_xcats, n_rows),
        cid_of_row=(wide_rows // n_dates).astype(np.int32),
        date_of_row=date_of_row,
        nanos_of_row=(date_axis[date_of_row] + first_day) * NS_PER_DAY,
        cids=cids,
        cid_pos={c: i for i, c in enumerate(cids)},
        xcat_pos=xcat_pos,
        columns=pd.Index(list(xcat_pos), name="xcat", dtype=object),
        cid_level=pd.Index(cids, dtype=object),
        obs_outside_cids=obs_outside_cids,
        xcats_outside_cids=xcats_outside_cids,
        df_cid_code=df_cid_code,
        df_xcat_code=df_xcat_code,
    )


# --------------------------------------------------------------------- downsample


def _period_index(
    dates: pd.DatetimeIndex, freq: str
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    # Empty periods are numbered too, and the labels come from pandas' own resampler
    # so that `freq="B"` snaps weekend dates the way `categories_df` does - H9.
    counts = pd.Series(np.zeros(len(dates)), index=dates).resample(freq).size()
    period_of_date = np.repeat(np.arange(len(counts)), counts.to_numpy())
    return period_of_date, counts.index


def _downsample_dfw(dfw: pd.DataFrame, freq: str) -> pd.core.groupby.DataFrameGroupBy:
    return dfw.groupby(
        [pd.Grouper(level="cid"), pd.Grouper(level="real_date", freq=freq)],
        observed=True,  # H12
    )


def _downsample_dfw_arrays(
    wide: WideFrame, rows: np.ndarray, metric: str, freq: str
) -> Tuple[pd.core.groupby.DataFrameGroupBy, pd.MultiIndex]:
    period_of_date, period_dates = wide.periods[freq]
    n_periods = len(period_dates)
    # a request keeping every row is the single-call case, where a gather is all overhead
    row_idx = None if rows.all() else np.flatnonzero(rows)
    if row_idx is None:
        cid_of_row, date_of_row = wide.cid_of_row, wide.date_of_row
        cells = wide.value_arrs[metric]
    else:
        cid_of_row = wide.cid_of_row.take(row_idx)
        date_of_row = wide.date_of_row.take(row_idx)
        # gather along the contiguous axis, so `.T` stays F-ordered
        cells = wide.value_arrs[metric].take(row_idx, axis=1)

    # One integer per (cid, period). The rows are sorted by (cid, date), so each
    # group is one run of it and the distinct keys are a run-length scan.
    period_key = cid_of_row.astype(np.int64) * n_periods
    period_key += period_of_date[date_of_row]
    period_starts = np.ones(len(period_key), dtype=bool)
    np.not_equal(period_key[1:], period_key[:-1], out=period_starts[1:])
    distinct_keys = period_key[period_starts]

    period_index = pd.MultiIndex(
        levels=[wide.cid_level, period_dates],
        codes=[distinct_keys // n_periods, distinct_keys % n_periods],
        names=["cid", "real_date"],
        verify_integrity=False,
    )
    dfw = pd.DataFrame(cells.T, columns=wide.columns, copy=False)
    return dfw.groupby(np.cumsum(period_starts) - 1, sort=False), period_index


# ----------------------------------------- aggregate, lag, assemble, drop empty rows


def _aggregate_xcats(
    grouped: pd.core.groupby.DataFrameGroupBy, xcats: List[str], agg_method: str
) -> pd.DataFrame:
    if agg_method == "sum":
        dfw_agg = grouped[xcats].sum(min_count=1)  # `min_count` keeps a NaN period NaN
    else:
        dfw_agg = grouped[xcats].agg(agg_method).astype(dtype=np.float32)  # H3
    if isinstance(dfw_agg, pd.Series):
        # "size" reduces across the columns rather than within each one; broadcast back
        dfw_agg = pd.DataFrame({x: dfw_agg for x in xcats})
    return dfw_agg


def _lag_explanatory(dfw_explanatory: pd.DataFrame, lag: int) -> pd.DataFrame:
    # a non-positive lag is a no-op in the shipped function too - H7
    if lag <= 0:
        return dfw_explanatory
    return dfw_explanatory.groupby(level=0, observed=True).shift(lag)


def _build_dfc(
    dfw_explanatory: pd.DataFrame, dep_col: pd.Series, out_xcats: List[str], fwin: int
) -> pd.DataFrame:
    dep = out_xcats[-1]
    if fwin > 1:
        # the forward window is not grouped by cid, so cross sections bleed - H8
        dep_col = dep_col.rolling(window=fwin).mean().shift(1 - fwin)
    dfw_explanatory.index.names = ["cid", "real_date"]
    # assigned last: a category used as both explanatory and dependent holds this
    dfw_explanatory[dep] = dep_col
    columns = out_xcats[:-1] + [dep]
    dfc = dfw_explanatory[columns]
    dfc.columns = pd.Index(columns, dtype=object)  # a plain unnamed object axis - H4
    return dfc


def _cast_cid_index_to_object(dfc: pd.DataFrame) -> pd.DataFrame:
    """Cast a categorical `cid` index level back to object, then drop all-NaN rows."""

    if dfc.index.dtypes["cid"].name == "category":
        dfc.index = pd.MultiIndex(
            levels=[dfc.index.levels[0].astype("object"), dfc.index.levels[1]],
            codes=dfc.index.codes,
            names=dfc.index.names,
        )
    return dfc.dropna(axis=0, how="all")


# ------------------------------------------------------------ one request, end to end


def _categories_df_via_pivot(
    df: pd.DataFrame, args: Dict[str, Any], is_qdf: bool
) -> pd.DataFrame:
    """One frequency request through `pivot`; serves every frame, at every shape."""

    reduced, out_xcats, out_cids = _reduce_df(df=df, args=args, is_qdf=is_qdf)
    _check_reduced_xcats_and_cids(out_xcats=out_xcats, out_cids=out_cids, args=args)
    _check_val_column(df=df, val=args["val"])

    dfw = _pivot_dfw(reduced=reduced, val=args["val"])
    _check_dfw_columns(out_xcats=out_xcats, dfw_columns=dfw.columns)

    grouped = _downsample_dfw(dfw=dfw, freq=args["_bday_freq"])
    aggs = args["xcat_aggs"]
    dfw_explanatory = _aggregate_xcats(
        grouped=grouped, xcats=list(dict.fromkeys(out_xcats[:-1])), agg_method=aggs[0]
    )
    dep_col = _aggregate_xcats(
        grouped=grouped, xcats=[out_xcats[-1]], agg_method=aggs[1]
    )[out_xcats[-1]]
    return _cast_cid_index_to_object(
        dfc=_build_dfc(
            dfw_explanatory=_lag_explanatory(
                dfw_explanatory=dfw_explanatory, lag=args["lag"]
            ),
            dep_col=dep_col,
            out_xcats=out_xcats,
            fwin=args["fwin"],
        )
    )


def _categories_df_via_dfw(wide: WideFrame, args: Dict[str, Any]) -> pd.DataFrame:
    """The same steps as `_categories_df_via_pivot`, off the reshape the batch shares."""

    rows, out_xcats, out_cids = _reduce_dfw_rows(wide=wide, args=args)
    _check_reduced_xcats_and_cids(out_xcats=out_xcats, out_cids=out_cids, args=args)
    _check_val_column(df=wide.df, val=args["val"])
    xcat_survives = (wide.filled & rows).any(axis=1)
    _check_dfw_columns(
        out_xcats=out_xcats,
        dfw_columns={x for x, i in wide.xcat_pos.items() if xcat_survives[i]},
    )

    grouped, period_index = _downsample_dfw_arrays(
        wide=wide, rows=rows, metric=args["val"], freq=args["_bday_freq"]
    )
    aggs = args["xcat_aggs"]
    dfw_explanatory = _aggregate_xcats(
        grouped=grouped, xcats=list(dict.fromkeys(out_xcats[:-1])), agg_method=aggs[0]
    )
    dep_frame = _aggregate_xcats(
        grouped=grouped, xcats=[out_xcats[-1]], agg_method=aggs[1]
    )
    dfw_explanatory.index = dep_frame.index = period_index
    return _cast_cid_index_to_object(
        dfc=_build_dfc(
            dfw_explanatory=_lag_explanatory(
                dfw_explanatory=dfw_explanatory, lag=args["lag"]
            ),
            dep_col=dep_frame[out_xcats[-1]],
            out_xcats=out_xcats,
            fwin=args["fwin"],
        )
    )


def _year_group_labels(start_year: int, end_year: int, years: int) -> List[str]:
    """Year-group labels spanning `start_year` to `end_year`, the last open-ended."""

    n_groups = int((end_year - start_year) / years)
    operator.index(years)  # `categories_df` lists each group with `range` - H14
    labels, group_start_year = [], start_year
    for _ in range(n_groups):
        labels.append(f"{group_start_year} - {group_start_year + (years - 1)}")
        group_start_year += years
    labels.append(f"{group_start_year} - now")
    return labels


def _categories_df_by_year_groups(
    df: pd.DataFrame, args: Dict[str, Any], is_qdf: bool
) -> pd.DataFrame:
    """One multi-year request: two categories aggregated over fixed year groups - H14."""

    reduced, out_xcats, out_cids = _reduce_df(df=df, args=args, is_qdf=is_qdf)
    _check_reduced_xcats_and_cids(out_xcats=out_xcats, out_cids=out_cids, args=args)
    _check_val_column(df=df, val=args["val"])

    val, years = args["val"], args["years"]
    start_year = pd.to_datetime(args["start"]).year
    labels = _year_group_labels(
        start_year=start_year, end_year=reduced["real_date"].max().year + 1, years=years
    )

    # `year % start_year` is `categories_df`'s own group lookup, kept on Python ints
    # so that a `start_year` of 0 raises rather than returning 0 - H14
    year_of_obs = reduced["real_date"].dt.year.to_numpy()
    distinct_years, inverse = np.unique(year_of_obs, return_inverse=True)
    group_of_year = np.array(
        [labels[int((int(y) % start_year) / years)] for y in distinct_years],
        dtype=object,
    )

    reduced = reduced.copy()
    reduced["custom_date"] = group_of_year[inverse]

    parts = []
    for xcat, agg_method in zip(out_xcats, args["xcat_aggs"]):
        part = (
            reduced[reduced["xcat"] == xcat]
            .groupby(["xcat", "cid", "custom_date"], observed=True)[[val]]
            # `numeric_only` is what makes "size", "count" and "nunique" raise here
            .aggregate(agg_method, numeric_only=True)
            .reset_index()
            .rename(columns={"custom_date": "real_date"})
        )
        parts.append(part[["cid", "xcat", "real_date", val]])

    dfc = pd.concat(parts).pivot(index=("cid", "real_date"), columns="xcat", values=val)
    return _cast_cid_index_to_object(dfc=dfc)


# ----------------------------------------------------------------- the entry points


def _categories_df_many(
    df: pd.DataFrame,
    arg_batches: Sequence[Optional[Dict[str, Any]]],
) -> List[Any]:
    """One result per request, in order; a raising request yields its exception."""

    results: List[Any] = [None] * len(arg_batches)
    is_qdf = type(df) is QuantamentalDataFrame
    freq_idx: List[int] = []
    for i, args in enumerate(arg_batches):
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
        except Exception as e:  # noqa: BLE001 - the exception IS this request's result
            results[i] = e
    if not freq_idx:
        return results

    wide = (
        _build_dfw(df=df, arg_batches=[arg_batches[i] for i in freq_idx], is_qdf=is_qdf)
        if _can_build_dfw(df=df)
        else None
    )
    for i in freq_idx:
        args = arg_batches[i]
        try:
            if wide is not None and args["val"] in wide.value_arrs:
                results[i] = _categories_df_via_dfw(wide=wide, args=args)
            else:
                results[i] = _categories_df_via_pivot(df=df, args=args, is_qdf=is_qdf)
        except Exception as e:  # noqa: BLE001
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

    _args = {
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
        arg_batches=[_fill_default_args(args=_args)],
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

    full_args: List[Optional[Dict[str, Any]]] = []
    results: List[Any] = []
    for args in arg_batches:
        try:
            full_args.append(_fill_default_args(args=args))
            results.append(None)
        except Exception as e:  # noqa: BLE001 - an unusable request is its own result
            full_args.append(None)
            results.append(e)
    for i, result in enumerate(_categories_df_many(df=df, arg_batches=full_args)):
        if full_args[i] is not None:
            results[i] = result
    return results
