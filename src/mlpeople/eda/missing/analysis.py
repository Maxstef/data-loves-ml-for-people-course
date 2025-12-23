import pandas as pd

def get_null_df(df):
    """
    Compute null-value statistics for each column in a DataFrame.

    For every column, this function calculates:
    - Total number of null values
    - Percentage of null values relative to total row count

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per column containing:
        - column_name : str
        - null_count : int
        - null_percentage : float (rounded to 2 decimals)
    """

    null_count = df.isnull().sum()
    null_percentage = round((df.isnull().sum()/df.shape[0])*100, 2)
    null_df = pd.DataFrame({'column_name' : df.columns,'null_count' : null_count,'null_percentage': null_percentage})
    null_df.reset_index(drop = True, inplace = True)
    return null_df.sort_values(by = 'null_percentage', ascending = False)

def drop_cols_above_missing_threshold(df, threshold=40):
    """
    Drop columns whose percentage of missing values exceeds a threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float, default 40
        Maximum allowed percentage of missing values.
        Columns with null_percentage > threshold are removed.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with columns above the missing-value threshold removed.
    """

    null_df = get_null_df(df)
    columns_to_be_deleted = null_df[null_df['null_percentage'] > threshold].column_name.to_list()
    return df.drop(columns = columns_to_be_deleted)

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
