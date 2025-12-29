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
    null_percentage = round((df.isnull().sum() / df.shape[0]) * 100, 2)
    null_df = pd.DataFrame(
        {
            "column_name": df.columns,
            "null_count": null_count,
            "null_percentage": null_percentage,
        }
    )
    null_df.reset_index(drop=True, inplace=True)
    return null_df.sort_values(by="null_percentage", ascending=False)
