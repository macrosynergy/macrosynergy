from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from macrosynergy.management.simulate.simulate_quantamental_data import make_qdf
from macrosynergy.panel.category_relations import CategoryRelations
import textwrap
import seaborn as sns


def multiple_reg_scatter(
    cat_rels,
    ncol=0,
    nrow=0,
    figsize=(20, 15),
    title="",
    title_xadj=0.5,
    title_yadj=0.99,
    title_fontsize=20,
    xlab="",
    ylab="",
    fit_reg=True,
    reg_ci=95,
    reg_order=1,
    reg_robust=False,
    coef_box=None,
    coef_box_font_size=12,
    prob_est="pool",
    separator=None,
    single_chart=False,
    subplot_titles=None,
    color_cids=False,
):
    """
    Visualize the results of a multiple regression analysis across categories.

    :param <List[CategoryRelations]> cat_rels: list of CategoryRelations objects.
    :param <int> ncol: number of columns in the grid. Default is 0, which will be set to
        the length of cat_rels.
    :param <int> nrow: number of rows in the grid. Default is 0, which will be set to 1.
    :param <Tuple[float]> figsize: size of the figure. Default is (20, 15).
    :param <str> title: title of the figure. Default is an empty string.
    :param <str> xlab: label of the x-axis. Default is an empty string.
    :param <str> ylab: label of the y-axis. Default is an empty string.
    :param <bool> fit_reg: if True (default) a linear regression line is fitted to the
        data.
    :param <int> reg_ci: confidence interval for the regression line. Default is 95.
    :param <int> reg_order: order of the regression line. Default is 1.
    :param <bool> reg_robust: if True (default is False) robust standard errors are used.
    :param <str> coef_box: if not None, a box with the coefficients of the regression is
        displayed. Default is None.
    :param <int> coef_box_font_size: font size of the coefficients box. Default is 12. If
        set to 0 it automatically sets the fontsize according to matplotlib.
    :param <str> prob_est: method to estimate the probability. Default is 'pool'.
    :param <int> separator: allows categorizing the scatter analysis by
            integer. This is done by setting it to a year [2010, for instance] which will
            subsequently split the time-period into the sample before (not including) that
            year and from (including) that year.
    :param <bool> single_chart: if True (default is False) all the data is plotted in a
        single chart. If False, a grid of charts is created.
    :param <List[str]> subplot_titles: list of titles for each subplot. Default is None.
    """
    sns.set_theme(style="whitegrid")
    if ncol == 0:
        ncol = len(cat_rels)
    if nrow == 0:
        nrow = 1
    if subplot_titles is not None:
        if len(subplot_titles) != len(cat_rels):
            raise ValueError(
                "The length of subplot_titles must be equal to the length of cat_rels."
            )

    if separator is not None:
        if separator == "cids":
            raise ValueError(
                "Separator 'cids' is not permitted in multiple_reg_scatter. To get a plot across multiple cids, please specify separator as cids inside reg_scatter."
            )
    single_scatter = color_cids
    separator = "cids" if color_cids else separator
    fig, axes = plt.subplots(
        nrows=nrow, ncols=ncol, figsize=figsize, sharex=True, sharey=True
    )
    fig.suptitle(title, x=title_xadj, y=title_yadj, fontsize=title_fontsize)
    fig.supxlabel(xlab)
    fig.supylabel(ylab)

    for i, cat_rel in enumerate(cat_rels):
        row = i // ncol
        col = i % ncol
        if not isinstance(axes, np.ndarray):
            ax = axes
            ax.set_facecolor("white")
        else:
            ax = axes[i] if (ncol == 1 or nrow == 1) else axes[row, col]
            ax.set_facecolor("white")
        if subplot_titles is not None:
            subplot_title = subplot_titles[i]
        else:
            if cat_rel.years is None:
                dates = (
                    cat_rel.df.index.get_level_values("real_date")
                    .to_series()
                    .dt.strftime("%Y-%m-%d")
                )
                subplot_title = (
                    f"{cat_rel.xcats[0]} and {cat_rel.xcats[1]} "
                    f"from {dates.min()} to {dates.max()}"
                )
            else:
                subplot_title = f"{cat_rel.xcats[0]} and {cat_rel.xcats[1]}"

        width = (figsize[0] // ncol) * 6

        wrapped_title = "\n".join(textwrap.wrap(subplot_title, width=width))
        cat_rel.reg_scatter(
            title=wrapped_title,
            labels=False,
            xlab="",
            ylab="",
            fit_reg=fit_reg,
            reg_ci=reg_ci,
            reg_order=reg_order,
            reg_robust=reg_robust,
            coef_box=coef_box,
            coef_box_font_size=coef_box_font_size,
            prob_est=prob_est,
            single_chart=single_chart,
            separator=separator,
            ax=ax,
            single_scatter=single_scatter,
        )

    plt.subplots_adjust(top=title_yadj - 0.01)
    plt.tight_layout()
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
    df_xcats.loc["CRY"] = [
        "2000-01-01",
        "2020-10-30",
        1,
        2,
        0.95,
        1,
    ]
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

    # cr1.reg_scatter(
    #     labels=False,
    #     single_scatter=True,
    #     title="Carry and Return",
    #     xlab="Carry",
    #     ylab="Return",
    #     prob_est="map",
    #     separator="cids"
    # )

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

    cr4 = CategoryRelations(
        dfdx,
        xcats=["CRY", "INFL"],
        # xcat1_chg="diff",
        freq="Q",
        lag=1,
        cids=cidx,
        xcat_aggs=["mean", "sum"],
        start="2001-01-01",
        blacklist=black,
        years=None,
    )

    cr5 = CategoryRelations(
        dfdx,
        xcats=["CRY", "INFL"],
        # xcat1_chg="diff",
        freq="Q",
        lag=1,
        cids=cidx,
        xcat_aggs=["mean", "sum"],
        start="2001-01-01",
        blacklist=black,
        years=None,
    )

    cr6 = CategoryRelations(
        dfdx,
        xcats=["CRY", "INFL"],
        # xcat1_chg="diff",
        freq="Q",
        lag=1,
        cids=cidx,
        xcat_aggs=["mean", "sum"],
        start="2001-01-01",
        blacklist=black,
        years=None,
    )

    multiple_reg_scatter(
        [cr1, cr2, cr3, cr4, cr5, cr6],
        title="Growth trend and subsequent sectoral equity returns.",
        xlab="Real technical growth trend",
        ylab="Excess Return",
        ncol=3,
        nrow=2,
        coef_box="upper right",
        color_cids=True,
    )

    multiple_reg_scatter(
        [cr1, cr2, cr3, cr4, cr5, cr6],
        title="Growth trend and subsequent sectoral equity returns.",
        xlab="Real technical growth trend",
        ylab="Excess Return",
        ncol=6,
        nrow=2,
    )
