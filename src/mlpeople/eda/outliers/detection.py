

def get_outlier_range(series, method="iqr", threshold=1.5):
    """
    Calculate outlier range for a numeric pandas Series.

    Parameters
    ----------
    series : pd.Series
        Numeric data series.
    method : {"iqr", "std"}, default "iqr"
        Method to detect outliers:
        - "iqr": uses Q1 - 1.5*IQR, Q3 + 1.5*IQR
        - "std": uses mean ± threshold * std
    threshold : float, default 1.5
        Multiplier for IQR or standard deviation.

    Returns
    -------
    tuple
        (min_value, max_value) for outlier detection.
    """

    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        min_value = Q1 - threshold * IQR
        max_value = Q3 + threshold * IQR
    elif method == "std":
        mean = series.mean()
        std = series.std()
        min_value = mean - threshold * std
        max_value = mean + threshold * std
    else:
        raise ValueError("method must be 'iqr' or 'std'")

    return min_value, max_value


def get_outlier_mask(series, method="iqr", threshold=1.5):
    """
    Return boolean mask indicating outliers in the series.

    Parameters
    ----------
    series : pd.Series
        Numeric series.
    method : {"iqr", "std"}, default "iqr"
        Outlier detection method.
    threshold : float, default 1.5
        Multiplier for IQR or standard deviation.

    Returns
    -------
    pd.Series (bool)
        True indicates an outlier.
    """
    min_val, max_val = get_outlier_range(series, method, threshold)
    return (series < min_val) | (series > max_val)
