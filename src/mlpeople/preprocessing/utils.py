def get_low_cardinality_cats(df, threshold=10, include_bool=False):
    """
    Return categorical columns whose number of unique values
    is less than or equal to the threshold.

    Parameters:
        df : pandas DataFrame
        threshold : int
        include_bool : bool (whether to treat boolean as categorical)

    Returns:
        List of column names
    """

    cat_types = ["object", "category"]

    if include_bool:
        cat_types.append("bool")

    cat_cols = df.select_dtypes(include=cat_types).columns

    low_card_cols = [
        col for col in cat_cols if df[col].nunique(dropna=False) <= threshold
    ]

    return low_card_cols
