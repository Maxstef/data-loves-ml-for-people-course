

def get_outlier_range_series(series):
    """
    Calculate the typical outlier range for a numeric pandas Series using the IQR method.

    Parameters
    ----------
    series : pandas.Series
        Numeric data series to calculate the outlier range for.

    Returns
    -------
    tuple
        (Min_value, Max_value) representing the lower and upper bounds for outliers.
        Values outside this range are typically considered outliers.

    Notes
    -----
    - Uses the standard 1.5 * IQR rule:
        Min_value = Q1 - 1.5 * IQR
        Max_value = Q3 + 1.5 * IQR
    - Q1 is the 25th percentile, Q3 is the 75th percentile.
    """

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    Min_value = (Q1 - 1.5 * IQR)
    Max_value = (Q3 + 1.5 * IQR)
    return Min_value, Max_value

def get_outlier_range(dataset, column):
    """
    Calculate the outlier range for a specific column in a DataFrame.

    Parameters
    ----------
    dataset : pandas.DataFrame
        DataFrame containing the column to check.
    column : str
        Name of the numeric column to calculate the outlier range.

    Returns
    -------
    tuple
        (Min_value, Max_value) representing the lower and upper bounds for outliers
        in the specified column.
    """

    return get_outlier_range_series(dataset[column])
