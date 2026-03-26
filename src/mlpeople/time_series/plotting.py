import matplotlib.pyplot as plt

# columns that are engineered and should not be auto-selected as original
engineered_cols = ["Date", "trend", "detrended", "season_key", "seasonality", "resid"]


def filter_original_col(col):
    """Return True if a column is not engineered (i.e., likely the original series)."""
    return col not in engineered_cols


def plot_ts(
    df,
    column,
    title=None,
    ylabel=None,
    xlabel="Date",
    rolling=None,
    resample=None,
):
    """
    Generic time series plot.

    Parameters
    ----------
    df : pandas.DataFrame with datetime index
    column : column name to plot
    title : chart title
    ylabel : label for Y axis
    rolling : int, optional rolling window to plot moving average
    resample: resample examples: 'W', 'M', 'Q', 'Y'
    """
    plot_df = df.copy()

    if resample:
        plot_df = plot_df.resample(resample).mean()

    fig, ax = plt.subplots()

    ax.plot(plot_df.index, plot_df[column], label="Original", linewidth=2)

    if rolling:
        ax.plot(
            plot_df.index,
            plot_df[column].rolling(rolling).mean(),
            label=f"Rolling Mean ({rolling})",
            linewidth=2,
        )

    ax.set_title(title or column)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or column)

    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_ts_trend(decomposed_df, y_col=None, title=None, figsize=(8, 6)):
    """
    Plot original series and trend on top, detrended series below.

    Parameters
    ----------
    decomposed_df : pandas.DataFrame
        Must contain columns: [y_col, "trend", "detrended"]
    y_col : str
        Name of the original series column
    title : str
        Figure title for top subplot
    figsize : tuple
        Figure size
    """

    if y_col is None:
        y_col = next(
            (col for col in decomposed_df.columns if filter_original_col(col)), None
        )

    if title is None:
        title = y_col

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Top: original + trend
    axes[0].plot(
        decomposed_df.index, decomposed_df[y_col], label="Original", color="blue"
    )
    axes[0].plot(
        decomposed_df.index, decomposed_df["trend"], label="Trend", color="orange"
    )
    axes[0].set_title(title, fontsize=16)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Bottom: detrended
    axes[1].plot(
        decomposed_df.index,
        decomposed_df["detrended"],
        label="Detrended",
        color="green",
    )
    axes[1].set_title("Detrended Series", fontsize=16)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_ts_decompose(decomposed_df, y_col=None, title=None, figsize=(8, 2)):
    """
    Plot full time series decomposition: original, trend, seasonality, residual.

    Parameters
    ----------
    decomposed_df : pandas.DataFrame
        Must contain columns: [y_col, "trend", "seasonality", "resid"]
    y_col : str
        Name of the original series column
    title : str
        Figure title
    figsize : tuple
        Base figure size (height scales with number of subplots)
    """

    if y_col is None:
        y_col = next(
            (col for col in decomposed_df.columns if filter_original_col(col)), None
        )

    if title is None:
        title = y_col

    cols = [y_col, "trend", "seasonality", "resid"]
    y_labels = ["Original", "Trend", "Seasonality", "Residuals"]  # labels for y-axis
    n_plots = len(cols)
    fig, axes = plt.subplots(
        n_plots, 1, figsize=(figsize[0], figsize[1] * n_plots), sharex=True
    )

    for i, col in enumerate(cols):
        axes[i].plot(decomposed_df.index, decomposed_df[col], color=f"C{i}")
        axes[i].set_ylabel(y_labels[i], fontsize=12)  # add y-axis label
        axes[i].set_title(col if i == 0 else "", fontsize=16 if i == 0 else 14)
        axes[i].grid(alpha=0.3)

    axes[-1].set_xlabel("Time")  # label x-axis on bottom subplot

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_ts_df_decomposition(
    decomposed_df,
    y_col=None,
    title_trend="Trend",
    title_decompose="Seasonal Decomposition",
    figsize_trend=(8, 6),
    figsize_decompose=(8, 2),
    show_trend=True,
    show_decompose=True,
):
    """
    High-level wrapper: plot trend/detrended and full decomposition independently.

    Parameters
    ----------
    decomposed_df : pandas.DataFrame
        Must contain columns: [y_col, "trend", "detrended", "seasonality", "resid"]
    y_col : str
        Name of the original series column
    title_trend : str
        Title for original + trend / detrended plots
    title_decompose : str
        Title for full decomposition plots (subplots)
    figsize : tuple
        Figure size for all plots
    show_trend : bool
        Whether to show original + trend / detrended subplots
    show_decompose : bool
        Whether to show full decomposition subplots
    """

    if y_col is None:
        y_col = next(
            (col for col in decomposed_df.columns if filter_original_col(col)), None
        )

    if show_trend:
        # Two aligned subplots: original+trend on top, detrended below
        plot_ts_trend(
            decomposed_df,
            y_col=y_col,
            title=title_trend,
            figsize=figsize_trend,
        )

    if show_decompose:
        # Full decomposition: original, trend, seasonal, residual
        plot_ts_decompose(
            decomposed_df,
            y_col=y_col,
            title=title_decompose,
            figsize=figsize_decompose,
        )
