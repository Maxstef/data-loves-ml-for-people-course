import pandas as pd
import numpy as np
from mlpeople.eda.outliers.detection  import get_outlier_range

def pd_cut_by_quantiles(
    series,
    quantiles=[0, 0.1, 0.35, 0.7, 0.9, 1],
    labels=['very low', 'low', 'medium', 'high', 'very high']
):
    """
    Bin a numeric pandas Series into categories using explicit quantile boundaries.

    This function computes quantile values from the data and then applies
    `pd.cut` using those quantile-based bin edges.

    Parameters
    ----------
    series : pandas.Series
        Numeric input data to be binned.
    quantiles : list of float, default [0, 0.1, 0.35, 0.7, 0.9, 1]
        Quantile boundaries used to compute bin edges.
        Values must be between 0 and 1 and sorted in ascending order.
    labels : list of str, default ['very low', 'low', 'medium', 'high', 'very high']
        Labels assigned to each resulting bin.
        Length must be `len(quantiles) - 1`.

    Returns
    -------
    pandas.Series
        Categorical Series with values assigned to quantile-based bins.
    """

    return pd.cut(
        series,
        bins=list(series.quantile(quantiles)),
        labels=labels,
        include_lowest=True
    )


def pd_qcut_by_quantiles(
    series,
    quantiles=[0, 0.1, 0.35, 0.7, 0.9, 1],
    labels=['very low', 'low', 'medium', 'high', 'very high']
):
    """
    Bin a numeric pandas Series into categories using equal-sized quantile buckets.

    This function uses `pd.qcut` to divide the data such that each bin contains
    approximately the same number of observations.

    Parameters
    ----------
    series : pandas.Series
        Numeric input data to be binned.
    quantiles : list of float, default [0, 0.1, 0.35, 0.7, 0.9, 1]
        Quantile thresholds defining the bins.
    labels : list of str, default ['very low', 'low', 'medium', 'high', 'very high']
        Labels assigned to each resulting bin.
        Length must be `len(quantiles) - 1`.

    Returns
    -------
    pandas.Series
        Categorical Series with values assigned to quantile-based bins.
    """

    return pd.qcut(
        series,
        q=quantiles,
        labels=labels,
        duplicates='drop'
    )


def pd_cut_by_values(
    series,
    bins=None,
    labels=['lower outliers', 'very low', 'low', 'medium', 'high', 'very high', 'upper outliers']
):
    """
    Bin a numeric pandas Series using explicit value-based thresholds,
    including lower and upper outlier ranges.

    If `bins` are not provided, the function:
    - Computes outlier thresholds using the IQR method
    - Ensures bins cover at least the 10th–90th percentile range
    - Adds open-ended bins for extreme outliers

    Parameters
    ----------
    series : pandas.Series
        Numeric input data to be binned.
    bins : list of float, optional
        Explicit bin edges. Must be sorted and include -inf/inf
        if outliers should be captured.
        If None, bins are computed automatically.
    labels : list of str, default
        ['lower outliers', 'very low', 'low', 'medium', 'high', 'very high', 'upper outliers']
        Labels assigned to each bin.

    Returns
    -------
    pandas.Series
        Categorical Series with values assigned to value-based bins.
    """

    if bins is None:
        Min_value, Max_value = get_outlier_range(series)

        # Ensure Min_value <= 10th percentile, Max_value >= 90th percentile
        Min_value = min(Min_value, series.quantile(0.1))
        Max_value = max(Max_value, series.quantile(0.9))

        bins = [
            -np.inf,
            Min_value,
            series.quantile(0.1),
            series.quantile(0.35),
            series.quantile(0.7),
            series.quantile(0.9),
            Max_value,
            np.inf
        ]
    
    return pd.cut(
        series,
        bins=bins,
        labels=labels,
        include_lowest=True
    )
