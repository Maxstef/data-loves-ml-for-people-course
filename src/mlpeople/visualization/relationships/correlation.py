import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


def show_correlation_matrix_filtered(
    df,
    cols=None,
    abs_threshold=0.5,
    width=1200,
    height=1000
):
    """
    Display an interactive filtered correlation matrix using Plotly.

    This function computes a Pearson correlation matrix for the selected
    numeric columns and visualizes only strong correlations whose absolute
    value exceeds `abs_threshold`.

    Self-correlations (diagonal values) are removed, and rows/columns that
    contain no strong correlations are dropped from the visualization.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data.
    cols : list[str] or None, default=None
        List of column names to include in the correlation matrix.
        If None, all numeric columns are used.
    abs_threshold : float, default=0.5
        Minimum absolute correlation value to display.
    width : int, default=1200
        Width of the Plotly figure in pixels.
    height : int, default=1000
        Height of the Plotly figure in pixels.

    Returns
    -------
    None
        Displays an interactive Plotly heatmap.
    """

    if cols is None:
        cols = df.select_dtypes('number').columns.tolist()

    # compute correlation matrix
    corr = df[cols].corr()

    # Remove self-correlations (set diagonal to NaN)
    np.fill_diagonal(corr.values, np.nan)

    # filter weak correlations
    corr = corr.where(corr.abs() > abs_threshold)

    # Drop rows and columns that are fully NaN (no strong correlations)
    corr = corr.dropna(how='all', axis=0).dropna(how='all', axis=1)

    # Plot full square heatmap
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title='Filtered Correlation Matrix (|corr| > ' + str(abs_threshold) + ')',
        aspect='auto'
    )

    fig.update_layout(width=width, height=height)
    fig.show()


def show_correlation_matrix_filtered_static(
    df,
    cols=None,
    abs_threshold=0.5,
    figsize=(12, 10),
    cmap='RdBu_r'
):
    """
    Display a static filtered correlation matrix using Matplotlib.

    This function computes a Pearson correlation matrix and visualizes
    only correlations whose absolute value exceeds `abs_threshold`.

    Self-correlations are removed, and rows/columns with no strong
    correlations are excluded from the plot.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data.
    cols : list[str] or None, default=None
        List of column names to include in the correlation matrix.
        If None, all numeric columns are used.
    abs_threshold : float, default=0.5
        Minimum absolute correlation value to display.
    figsize : tuple, default=(12, 10)
        Size of the matplotlib figure.
    cmap : str, default='RdBu_r'
        Colormap used for the heatmap.

    Returns
    -------
    None
        Displays a static correlation heatmap.
    """

    if cols is None:
        cols = df.select_dtypes('number').columns.tolist()

    # Compute correlation matrix
    corr = df[cols].corr()

    # Remove self-correlations
    np.fill_diagonal(corr.values, np.nan)

    # Filter weak correlations
    corr = corr.where(corr.abs() > abs_threshold)

    # Drop empty rows/columns
    corr = corr.dropna(how='all', axis=0).dropna(how='all', axis=1)

    if corr.empty:
        print("No correlations exceed the specified threshold.")
        return

    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Correlation')

    # Axis ticks and labels
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.index)

    # Annotate values
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            value = corr.values[i, j]
            if not np.isnan(value):
                ax.text(
                    j, i,
                    f"{value:.2f}",
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=9
                )

    ax.set_title(f'Filtered Correlation Matrix (|corr| > {abs_threshold})')
    plt.tight_layout()
    plt.show()
