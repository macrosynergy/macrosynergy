from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from macrosynergy.management.simulate.simulate_quantamental_data import make_qdf
from macrosynergy.panel.category_relations import CategoryRelations

def multiple_reg_scatter(cat_rels, ncol=0, nrow=0, figsize=(20, 15), title="", xlabel="", ylabel = ""):
    """
    Visualize the results of a multiple regression analysis across categories.

    :param <List[CategoryRelations]> cat_rels: list of CategoryRelations objects.

    :return: None
    """
    if ncol == 0:
        ncol = len(cat_rels)
    if nrow == 0:
        nrow = 1

    fig, axes = plt.subplots(
        nrows=nrow, 
        ncols=ncol, 
        figsize=figsize, 
        sharex=True, 
        sharey=True
    )
    fig.suptitle(title)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)

    for i, cat_rel in enumerate(cat_rels):
        row = i // nrow
        col = i % ncol
        if not isinstance(axes, np.ndarray):
            ax = axes
        else:
            ax = axes[i] if (ncol == 1 or nrow == 1) else axes[row, col]
        cat_rel.reg_scatter(
            labels=False,
            xlab="",
            ylab="",
            coef_box="upper right",
            prob_est="pool",
            fit_reg=True,
            reg_robust=False,
            single_chart=True,
            ax=ax,
        )
    plt.show()

if __name__ == "__main__":
    cids = ["AUD", "CAD", "GBP", "NZD", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2000-01-01", "2020-12-31", 0.1, 1]
    df_cids.loc["CAD"] = ["2001-01-01", "2020-11-30", 0, 1]
    df_cids.loc["BRL"] = ["2001-01-01", "2020-11-30", -0.1, 2]
    df_cids.loc["GBP"] = ["2002-01-01", "2020-11-30", 0, 2]
    df_cids.loc["NZD"] = ["2002-01-01", "2020-09-30", -0.1, 2]
    df_cids.loc["USD"] = ["2003-01-01", "2020-12-31", -0.1, 2]

    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]
    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-10-30", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2001-01-01", "2020-10-30", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {"AUD": ["2000-01-01", "2003-12-31"], "GBP": ["2018-01-01", "2100-01-01"]}

    # All AUD GROWTH locations.
    filt1 = (dfd["xcat"] == "GROWTH") & (dfd["cid"] == "AUD")
    filt2 = (dfd["xcat"] == "INFL") & (dfd["cid"] == "NZD")

    # Reduced DataFrame.
    dfdx = dfd[~(filt1 | filt2)].copy()
    dfdx["ERA"]: str = "before 2007"
    dfdx.loc[dfdx["real_date"].dt.year > 2007, "ERA"] = "from 2010"

    cidx = ["AUD", "CAD", "GBP", "USD"]

    cr1 = CategoryRelations(
        dfdx,
        xcats=["CRY", "XR"],
        freq="M",
        lag=1,
        cids=cidx,
        xcat_aggs=["mean", "sum"],
        start="2001-01-01",
        blacklist=black,
        years=None,
    )

    cr2 = CategoryRelations(
        dfdx,
        xcats=["CRY", "GROWTH"],
        # xcat1_chg="diff",
        freq="M",
        lag=1,
        cids=cidx,
        xcat_aggs=["mean", "sum"],
        start="2001-01-01",
        blacklist=black,
        years=None,
    )

    cr3 = CategoryRelations(
        dfdx,
        xcats=["CRY", "INFL"],
        # xcat1_chg="diff",
        freq="M",
        lag=1,
        cids=cidx,
        xcat_aggs=["mean", "sum"],
        start="2001-01-01",
        blacklist=black,
        years=None,
    )

    multiple_reg_scatter([cr1, cr2, cr3], title="Growth trend and subsequent sectoral equity returns.", xlabel="Real technical growth trend", ylabel="Excess Return")