import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from macrosynergy.management import business_day_dif
from macrosynergy.management.types import QuantamentalDataFrame


def view_panel_dates(
    df: pd.DataFrame,
    size: Tuple[float, float] = None,
    use_last_businessday: bool = True,
    header: str = None,
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

    sns.set(rc={"figure.figsize": size})
    sns.heatmap(
        df.T,
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
