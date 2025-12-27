
from mlpeople.eda.missing import get_null_df

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

    null_df = get_null_df(df)
    null_df_under = null_df[null_df['null_percentage'] < threshold]
    return null_df_under.sort_values(by = 'null_percentage', ascending = False)

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

    null_df = get_null_df(df)
    null_df_above = null_df[null_df['null_percentage'] > threshold]
    return null_df_above.sort_values(by = 'null_percentage', ascending = False)
