import matplotlib.pyplot as plt
import seaborn as sns
from mlpeople.eda.outliers  import get_outlier_range

def kde_plot(df, col_x, target_col=None, title=None, labels=None, drop_outliers=False):
    """
    Plot kernel density estimates (KDE) for a numeric variable, optionally split by category.

    This function draws one or more KDE curves for a continuous variable (`col_x`).
    If `target_col` is provided, separate KDE curves are plotted for each category
    in `target_col`, allowing comparison of distributions across groups.
    Optionally, outliers can be excluded independently for each group.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data to plot.
    col_x : str
        Name of the continuous (numeric) column for which the KDE is computed.
    target_col : str, optional
        Name of a categorical column used to split the data into groups.
        Each unique value produces a separate KDE curve.
    title : str, optional
        Title of the plot. If None, no title is shown.
    labels : dict, optional
        Mapping from values in `target_col` (or `col_x` when `target_col` is None)
        to legend labels. If a key is not present, the raw value is used.
    drop_outliers : bool, default False
        If True, values of `col_x` outside the outlier range (as determined by
        `get_outlier_range`) are excluded before computing each KDE.

    Notes
    -----
    - `col_x` must be numeric.
    - Outlier removal affects only the visualization, not the underlying data.
    - When `target_col` is None, a single KDE is plotted for the entire dataset.
    """

    if labels is None:
        labels = {}

    plt.figure(figsize=(14, 6))

    def plot_subset(data, label_key):
        if drop_outliers:
            min_val, max_val = get_outlier_range(data, col_x)
        else:
            min_val, max_val = data[col_x].min(), data[col_x].max()

        filtered = data.loc[data[col_x].between(min_val, max_val), col_x]
        label = labels.get(label_key, label_key)

        sns.kdeplot(filtered, label=label)

    if target_col:
        for val in df[target_col].dropna().unique():
            plot_subset(df[df[target_col] == val], val)
    else:
        plot_subset(df, col_x)

    plt.ticklabel_format(style="plain", axis="x")
    plt.xticks(rotation=45)
    plt.legend()
    plt.title(title or "")
    plt.show()

   