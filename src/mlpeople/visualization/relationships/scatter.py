import matplotlib.pyplot as plt
import seaborn as sns
from mlpeople.eda.outliers import get_outlier_range


def scatter_compare_by_category(
    df,
    col_x,
    col_y,
    target_col,
    titles=None,
    drop_outliers=False,
):
    """
    Draw side-by-side scatter plots for groups defined by a categorical variable.

    This function creates parallel scatter plots comparing two continuous variables
    (`col_x` and `col_y`) for each group defined by `target_col`. Each group is
    displayed in its own subplot, enabling visual comparison across categories.
    Optionally, outliers can be excluded independently for each group.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data to plot.
    col_x : str
        Name of the column to be plotted on the x-axis.
        Must be numeric.
    col_y : str
        Name of the column to be plotted on the y-axis.
        Must be numeric.
    target_col : str
        Name of the categorical column used to split the data into groups.
        Each unique value produces a separate subplot.
    titles : dict, optional
        Mapping from values in `target_col` to subplot titles. If a value is not
        present in the dictionary, the value itself is used as the title.
    drop_outliers : bool, default False
        If True, outliers are removed independently for each group using
        `get_outlier_range` before plotting.

    Notes
    -----
    - Both `col_x` and `col_y` must be numeric.
    - Outlier removal affects only the visualization, not the underlying data.
    - The function assumes a two-column subplot layout.
    """

    if titles is None:
        titles = {}

    plt.figure(figsize=(14, 6))

    def get_range(data, col):
        if drop_outliers:
            return get_outlier_range(data[col])
        return data[col].min(), data[col].max()

    for index, val in enumerate(df[target_col].dropna().unique(), start=1):
        df_temp = df[df[target_col] == val]

        x_min, x_max = get_range(df_temp, col_x)
        y_min, y_max = get_range(df_temp, col_y)

        filtered_x = df_temp.loc[df_temp[col_x].between(x_min, x_max), col_x]
        filtered_y = df_temp.loc[df_temp[col_y].between(y_min, y_max), col_y]

        plt.subplot(1, 2, index)
        plt.title(titles.get(val, val))
        sns.scatterplot(x=filtered_x, y=filtered_y, data=df_temp)

        plt.ticklabel_format(style="plain", axis="x")
        plt.ticklabel_format(style="plain", axis="y")

    plt.tight_layout(pad=4)
    plt.show()


def plot_scatter(x, y, title=None, xlabel="X", ylabel="Y", grid=False):
    """
    Plots simple scatter with matplotlib.pyplot

    """
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    plt.show()


def plot_predicted_vs_actual(
    y_train,
    y_train_pred,
    y_test=None,
    y_test_pred=None,
    figsize=(8, 6),
    train_color="blue",
    test_color="red",
    alpha=0.6,
    title="Predicted vs Actual",
    xlabel="Actual values",
    ylabel="Predicted values",
    show_legend=True,
):
    """
    Plots predicted vs actual values for training (and optionally test) sets.

    Parameters
    ----------
    y_train : array-like
        True values for training set.
    y_train_pred : array-like
        Predicted values for training set.
    y_test : array-like, optional
        True values for test set.
    y_test_pred : array-like, optional
        Predicted values for test set.
    figsize : tuple, default (8, 6)
        Figure size.
    train_color : str, default "blue"
        Color for training points.
    test_color : str, default "red"
        Color for test points.
    alpha : float, default 0.6
        Transparency for scatter points.
    title : str, default "Predicted vs Actual"
        Plot title.
    xlabel : str, default "Actual values"
        X-axis label.
    ylabel : str, default "Predicted values"
        Y-axis label.
    show_legend : bool, default True
        Whether to display legend.
    """

    plt.figure(figsize=figsize)

    # Training set
    plt.scatter(y_train, y_train_pred, color=train_color, alpha=alpha, label="Train")

    # Test set (optional)
    if y_test is not None and y_test_pred is not None:
        plt.scatter(y_test, y_test_pred, color=test_color, alpha=alpha, label="Test")

    # Reference line y=x
    all_values = []
    all_values.extend(y_train)
    all_values.extend(y_train_pred)
    if y_test is not None:
        all_values.extend(y_test)
    if y_test_pred is not None:
        all_values.extend(y_test_pred)
    min_val, max_val = min(all_values), max(all_values)
    plt.plot([min_val, max_val], [min_val, max_val], "k--", label="Ideal", lw=1)

    # Labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if show_legend:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
