import matplotlib.pyplot as plt
import seaborn as sns
from mlpeople.eda.outliers  import get_outlier_range

def draw_boxplot(
    df,
    col_x,
    col_y,
    title=None,
    hue_col=None,
    drop_outliers=False,
    subplot_position=None,
):
    """
    Draw a boxplot for a categorical and a continuous variable.

    This function visualizes the distribution of a continuous variable (`col_y`)
    across categories of a categorical variable (`col_x`). Optionally, the plot
    can be split by a hue variable and outliers can be excluded based on a
    predefined outlier range.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data to plot.
    col_x : str
        Name of the categorical column to be plotted on the x-axis.
    col_y : str
        Name of the continuous (numeric) column to be plotted on the y-axis.
    title : str, optional
        Title of the plot. If None, no title is shown.
    hue_col : str, optional
        Name of a categorical column used to split the boxplot by color (hue).
    drop_outliers : bool, default False
        If True, values of `col_y` outside the outlier range (as determined by
        `get_outlier_range`) are excluded from the plot.
    subplot_position : int, optional
        Position index for subplot placement when used in a multi-plot figure.
        If None, the plot is rendered as a standalone figure.

    Notes
    -----
    - `col_x` must be categorical.
    - `col_y` must be numeric.
    - Outlier removal only affects the visualization, not the underlying data.
    """

    if subplot_position is not None:
        plt.subplot(1, 2, subplot_position)

    if drop_outliers:
        min_y, max_y = get_outlier_range(df, col_y)
    else:
        min_y, max_y = df[col_y].min(), df[col_y].max()

    filtered_df = df[df[col_y].between(min_y, max_y)]

    plt.title(title or "")

    hue_order = (
        sorted(df[hue_col].dropna().unique(), reverse=True)
        if hue_col
        else None
    )

    sns.boxplot(
        data=filtered_df,
        x=col_x,
        y=col_y,
        hue=hue_col,
        order=sorted(df[col_x].dropna().unique(), reverse=True),
        hue_order=hue_order,
        flierprops={"markerfacecolor": "r", "marker": "D"},
    )

    plt.ticklabel_format(style="plain", axis="y")
    plt.xticks(rotation=90)

    if subplot_position is None:
        plt.show()


def draw_bi_boxplot(
    df,
    col_x,
    col_y,
    target_col=None,
    hue_col=None,
    titles=None,
    drop_outliers=False,
):
    """
    Draw side-by-side boxplots for multiple groups defined by a target column.

    This function creates parallel boxplots for subsets of the data defined by
    unique values in `target_col`. Each subset is plotted in its own subplot,
    enabling visual comparison between groups.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data to plot.
    col_x : str
        Name of the categorical column to be plotted on the x-axis.
    col_y : str
        Name of the continuous (numeric) column to be plotted on the y-axis.
    target_col : str
        Name of the categorical column used to split the data into groups,
        each rendered as a separate subplot.
    hue_col : str, optional
        Name of a categorical column used to further split each boxplot by color.
    titles : dict, optional
        Mapping from values in `target_col` to subplot titles. If a value is not
        present in the dictionary, the value itself is used as the title.
    drop_outliers : bool, default False
        If True, outliers in `col_y` are excluded independently for each group.

    Notes
    -----
    - Each unique value of `target_col` produces one subplot.
    - This function assumes a two-column subplot layout.
    - The plotting logic is delegated to `draw_boxplot` to ensure consistency.
    """

    if titles is None:
        titles = {}

    plt.figure(figsize=(16, 10))

    for index, val in enumerate(df[target_col].dropna().unique(), start=1):
        df_temp = df[df[target_col] == val]

        draw_boxplot(
            df=df_temp,
            col_x=col_x,
            col_y=col_y,
            title=titles.get(val, val),
            hue_col=hue_col,
            drop_outliers=drop_outliers,
            subplot_position=index,
        )

    plt.tight_layout(pad=4)
    plt.show()

