from mlpeople.eda.missing import get_null_df


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
    columns_to_be_deleted = null_df[
        null_df["null_percentage"] > threshold
    ].column_name.to_list()
    return df.drop(columns=columns_to_be_deleted)
