"""
Benchmarks for `macrosynergy.signal.signal_return_relations`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List

import pytest

from macrosynergy.signal.signal_return_relations import SignalReturnRelations
from tests.perf.harness.measurement import describe_call, measure
from tests.perf.harness.panel_sizes import PANEL_SIZES

pytestmark = [pytest.mark.perf, pytest.mark.perf_group("signal_return_relations")]

SIGNAL_XCAT_COUNT = 2
RETURN_XCAT_COUNT = 1


def signal_return_sizes(*only_tiers: str) -> List[Any]:
    """
    Long-format sizes whose xcats carry signal and return roles.

    Parameters
    ----------
    *only_tiers : str
        Tiers the benchmark is limited to. Omit to allow every selected tier.

    Returns
    -------
    List[Any]
        One `pytest.param` per measured tier, each holding a size whose tickers pair
        every cid with the signal and return categories.
    """
    role_xcats = [f"SIG{i:02d}" for i in range(SIGNAL_XCAT_COUNT)]
    role_xcats += [f"XR{i:02d}" for i in range(RETURN_XCAT_COUNT)]
    parameters = []
    for parameter in PANEL_SIZES.qdf_sizes(*only_tiers):
        panel_size = parameter.values[0]
        if panel_size is None:
            parameters.append(parameter)
            continue
        tickers = tuple(
            f"{cid}_{xcat}" for cid in panel_size.cids for xcat in role_xcats
        )
        parameters.append(
            pytest.param(
                replace(panel_size, cid_count=None, xcat_count=None, tickers=tickers),
                id=panel_size.tier,
                marks=parameter.marks,
            )
        )
    return parameters


@pytest.mark.parametrize("panel_size", signal_return_sizes("tiny", "small"), ids=str)
def test_single_statistic_table(benchmark, panel_size):
    """Tabulate panel map p-values of two signals against one return, monthly."""
    df = panel_size.as_qdf()
    setup = dict(
        rets=[f"XR{i:02d}" for i in range(RETURN_XCAT_COUNT)],
        sigs=[f"SIG{i:02d}" for i in range(SIGNAL_XCAT_COUNT)],
        cids=panel_size.cids,
        freqs=["M"],
        ms_panel_test=True,
    )
    relations = SignalReturnRelations(df, **setup)
    benchmark.extra_info["setup"] = {
        "SignalReturnRelations": describe_call(SignalReturnRelations, (df,), setup)
    }
    measure(
        benchmark,
        relations.single_statistic_table,
        stat="map_pval",
        type="panel",
        show_heatmap=False,
    )
