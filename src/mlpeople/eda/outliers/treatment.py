import pandas as pd
from mlpeople.eda.outliers import get_outlier_mask

def remove_outliers(series, series_2=None, method="iqr", threshold=1.5, return_type="series"):
    """
    Remove outliers from a numeric series based on IQR or standard deviation method.

    Parameters
    ----------
    series : pd.Series or list
        Primary numeric series.
    series_2 : pd.Series or list, optional
        Secondary series to filter in sync with `series`.
    method : {"iqr", "std"}, default "iqr"
        Outlier detection method.
    threshold : float, default 1.5
        Multiplier for IQR or standard deviation.
    return_type : {"series", "list"}, default "series"
        Return type for filtered samples.

    Returns
    -------
    tuple
        filtered_series : Series or list without outliers
        filtered_series_2 : Series or list corresponding to filtered_series, or None
        removed_indexes : list of indexes removed as outliers
    """
    mask = ~get_outlier_mask(pd.Series(series), method=method, threshold=threshold)

    filtered_series = pd.Series(series)[mask]
    filtered_series_2 = pd.Series(series_2)[mask] if series_2 is not None else None
    removed_indexes = (~mask).nonzero()[0].tolist()  # indexes of removed outliers

    if return_type == "list":
        filtered_series = filtered_series.tolist()
        if filtered_series_2 is not None:
            filtered_series_2 = filtered_series_2.tolist()

    return filtered_series, filtered_series_2, removed_indexes


def filter_outliers_df(df, column, *, method="iqr", threshold=1.5):
    """
    Return a new DataFrame with outliers removed from a specific numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Numeric column to filter outliers from.
    method : {"iqr", "std"}, default "iqr"
        Outlier detection method.
    threshold : float, default 1.5
        Threshold multiplier for IQR or standard deviation.

    Returns
    -------
    pd.DataFrame
        DataFrame with rows containing outliers in the specified column removed.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")

    series = df[column]
    mask = ~get_outlier_mask(series, method=method, threshold=threshold)
    filtered_df = df.loc[mask].copy()

    return filtered_df