import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
from macrosynergy.management import business_day_dif


def view_panel_dates(
    df: pd.DataFrame,
    size: Tuple[float, float] = None,
    use_last_businessday: bool = True,
    header: str = None,
    row_order: List[str] = None,
):
    """
    Visualize panel dates with color codes.

    Parameters
    ----------
    df : ~pandas.DataFrame
        A standardized Quantamental DataFrame with dates as index and series as columns.
    size : Tuple[float, float]
        tuple of floats with width/length of displayed heatmap.
    use_last_businessday : bool
        boolean indicating whether or not to use the last business day before today as
        the end date. Default is True.
    header : str
        A string to be used as the title of the heatmap. If None, a default header will be used
        based on the data type of the DataFrame.
    row_order : List[str]
        A list of strings specifying the order of rows in the heatmap. These rows
        correspond to the columns of the input DataFrame. If None, the default order
        used by Seaborn will be applied.
    """

    # DataFrame of official timestamps.
    if all(df.dtypes == object):
        df = df.apply(pd.to_datetime)
        # All series, in principle, should be populated to the last active release date
        # in the DataFrame.

        if use_last_businessday:
            maxdate: pd.Timestamp = (
                pd.Timestamp.today() - pd.tseries.offsets.BusinessDay()
            )
        else:
            maxdate: pd.Timestamp = df.max().max()

        df = business_day_dif(df=df, maxdate=maxdate)

        df = df.astype(float)
        # Ideally the data type should be int, but Pandas cannot represent NaN as int.
        # -- https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html#support-for-integer-na
        if header is None:
            header = f"Missing days up to {maxdate.strftime('%Y-%m-%d')}"

    else:
        if header is None:
            header = "Start years of quantamental indicators."

    if size is None:
        size = (max(df.shape[0] / 2, 18), max(1, df.shape[1] / 2))

    df = df.T

    if row_order is None:
        row_order = df.index.tolist()

    if isinstance(df.index, pd.CategoricalIndex):
        missing = set(row_order) - set(df.index.categories)
        if missing:
            df = df.reindex(row_order, fill_value=pd.NA)
        df.index = pd.CategoricalIndex(df.index, categories=row_order, ordered=True)
        df = df.sort_index()
    else:
        missing = set(row_order) - set(df.index)
        if missing:
            df = df.reindex(row_order, fill_value=pd.NA)
        df = df.loc[row_order]

    sns.set(rc={"figure.figsize": size})
    sns.heatmap(
        df,
        cmap="YlOrBr",
        center=df.stack().mean(),
        annot=True,
        fmt=".0f",
        linewidth=1,
        cbar=False,
    )
    plt.xlabel("")
    plt.ylabel("")
    plt.title(header, fontsize=18)
    plt.show()
