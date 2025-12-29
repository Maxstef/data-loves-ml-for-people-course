import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from mlpeople.eda.missing import get_null_df


def _filter_by_missing_threshold(df, threshold, *, above: bool):
    null_df = get_null_df(df)
    mask = (
        null_df["null_percentage"] > threshold
        if above
        else null_df["null_percentage"] < threshold
    )
    return null_df[mask].sort_values(
        by="null_percentage",
        ascending=False,
    )


def show_cols_below_missing_threshold(df, threshold=40):
    """
    Display columns with missing-value percentage below a threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float, default 40
        Upper bound for missing-value percentage.

    Returns
    -------
    pd.DataFrame
        Subset of the null statistics DataFrame containing only columns
        with null_percentage < threshold, sorted descending by null_percentage.
    """

    return _filter_by_missing_threshold(df, threshold, above=False)


def show_cols_above_missing_threshold(df, threshold=40):
    """
    Display columns with missing-value percentage above a threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float, default 40
        Lower bound for missing-value percentage.

    Returns
    -------
    pd.DataFrame
        Subset of the null statistics DataFrame containing only columns
        with null_percentage > threshold, sorted descending by null_percentage.
    """

    return _filter_by_missing_threshold(df, threshold, above=True)


def show_numeric_col_report(df, col, *, plot=True, bins=30, show_boxplot=False):
    """
    Display summary statistics and distribution for a numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Name of the numeric column.

    Returns
    -------
    pd.Series
        Summary statistics for the column.
    """

    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")

    series = df[col]

    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(f"Column '{col}' is not numeric")

    total_count = len(series)
    missing_count = series.isna().sum()
    missing_pct = (missing_count / total_count) * 100

    # Compute percentiles
    percentiles = series.quantile([0.25, 0.5, 0.75, 1.0]).to_dict()

    summary = pd.Series(
        {
            "mean": series.mean(),
            "median": series.median(),
            "mode": series.mode().iloc[0] if not series.mode().empty else None,
            "25%": percentiles[0.25],
            "50%": percentiles[0.5],
            "75%": percentiles[0.75],
            "100%": percentiles[1.0],
            "missing_pct": missing_pct,
        },
        name=col,
    )

    # Display summary
    display(summary)

    if plot:
        # Plot histogram
        series.dropna().hist(bins=bins)
        plt.title(f"Distribution of '{col}'")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

    if show_boxplot:
        # Boxplot
        plt.figure(figsize=(6, 2))
        plt.boxplot(series.dropna(), vert=False)
        plt.title(f"Boxplot of '{col}'")
        plt.xlabel(col)
        plt.show()

    return summary


def show_filled_numeric_histogram(
    df,
    col,
    *,
    strategy="mean",
    fill_value=None,
    bins=30,
    display_filled_details=False,
    show_before=False,
):
    """
    Display histogram of a numeric column after filling missing values,
    optionally showing the distribution before filling and details
    about the filling process.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Name of the numeric column to analyze.
    strategy : {"mean", "median", "mode", "constant"}, default "mean"
        Strategy used to fill missing values before plotting.
    fill_value : float, optional
        Constant value used to fill missing values when
        ``strategy="constant"``.
    bins : int, default 30
        Number of bins used for the histogram.
    display_filled_details : bool, default False
        Whether to display information about the filling process,
        including the strategy, fill value, and number of missing
        values before and after filling.
    show_before : bool, default False
        Whether to display the histogram of the column before
        filling missing values.

    Notes
    -----
    This function is intended for exploratory data analysis (EDA).
    It does not modify the original DataFrame.

    Raises
    ------
    KeyError
        If ``col`` is not present in the DataFrame.
    TypeError
        If ``col`` is not a numeric column.
    ValueError
        If ``strategy="constant"`` and ``fill_value`` is not provided.
    """

    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")

    series = df[col]

    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(f"Column '{col}' is not numeric")

    if strategy == "mean":
        value = series.mean()
    elif strategy == "median":
        value = series.median()
    elif strategy == "mode":
        mode = series.mode()
        value = mode.iloc[0] if not mode.empty else None
    elif strategy == "constant":
        if fill_value is None:
            raise ValueError("fill_value must be provided when strategy='constant'")
        value = fill_value
    else:
        raise ValueError(
            "strategy must be one of {'mean', 'median', 'mode', 'constant'}"
        )

    filled = series.fillna(value)

    if display_filled_details:
        # Display fill info
        display(
            {
                "column": col,
                "strategy": strategy,
                "fill_value": value,
                "missing_before": series.isna().sum(),
                "missing_after": filled.isna().sum(),
            }
        )

    # BEFORE histogram
    if show_before:
        series.dropna().hist(bins=bins)
        plt.title(f"'{col}' distribution BEFORE filling missing values")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

    # AFTER histogram
    filled.hist(bins=bins)
    plt.title(f"'{col}' distribution AFTER filling missing values")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


def show_categorical_col_report(df, col, *, show_plot=False, top_n=None):
    """
    Display a summary of a categorical column with counts, percentages,
    and optionally a countplot.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Name of the categorical column.
    show_plot : bool, default False
        Whether to display a countplot for the column.
    top_n : int, optional
        If provided, only show top N most frequent categories in the summary
        and plot.

    Returns
    -------
    pd.DataFrame
        Summary table with value, count, and percentage including missing values.
    """

    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")

    series = df[col]
    total_count = len(series)

    # Compute counts and percentages
    value_counts = series.value_counts(dropna=False)
    pct = (value_counts / total_count * 100).round(2)
    summary = pd.DataFrame(
        {
            "value": value_counts.index.astype(str),
            "count": value_counts.values,
            "percentage": pct.values,
        }
    )

    # Optionally filter top N
    if top_n is not None:
        summary = summary.head(top_n)

    display(summary)

    # Optional countplot
    if show_plot:
        plt.figure(figsize=(8, 5))
        # Replace NaN with string for plotting
        sns.countplot(x=series.fillna("Missing"), order=summary["value"])
        plt.title(f"Distribution of '{col}'")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

    return summary
