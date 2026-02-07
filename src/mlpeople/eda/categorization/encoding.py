from sklearn.preprocessing import OneHotEncoder


def encode_binary(df, col):
    """
    Encode a binary column into 0/1.
    The most frequent value is encoded as 1.

    Returns a pandas Series.
    """

    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")

    series = df[col]

    # Get non-null unique values
    unique_vals = series.dropna().unique()

    if len(unique_vals) != 2:
        raise ValueError(
            f"Column '{col}' must contain exactly 2 unique non-null values, "
            f"found {len(unique_vals)}"
        )

    # Count frequencies
    value_counts = series.value_counts(dropna=True)

    # Most frequent value → 1
    most_frequent = value_counts.idxmax()
    least_frequent = value_counts.idxmin()

    mapping = {most_frequent: 1, least_frequent: 0}

    return series.map(mapping)


def get_fitted_one_hot_encoder(
    train_inputs,
    cols=None,
    combine_map=None,
    sparse_output=False,
    handle_unknown="ignore",
    drop="if_binary",
):
    """
    Fit a OneHotEncoder on selected categorical columns, with optional category combination.

    This function allows you to:
    - Combine multiple categories into one group (useful for low-frequency categories)
    - Fit an encoder that can safely transform train and test data consistently
    - Automatically drop one category if column is binary to avoid collinearity in linear models

    Parameters:
    ----------
    train_inputs : pd.DataFrame
        The dataframe containing the data to fit the encoder on.

    cols : list of str, optional (default=None)
        List of categorical columns to encode.
        If None, all columns with dtype 'object' are used.

    combine_map : dict, optional (default=None)
        Dictionary specifying category combination rules.
        Format:
            {
                "ColumnName": {
                    "NewCategory1": ["OldCatA", "OldCatB"],
                    "NewCategory2": ["OldCatC", "OldCatD"]
                },
                ...
            }
        After mapping, each old value is replaced by its new category.

    sparse_output : bool, default=False
        Whether to return sparse matrix from OneHotEncoder.
        False returns a dense array.

    handle_unknown : str, default="ignore"
        Behavior when unknown categories are encountered during transform.
        "ignore" will create all zeros for unknown categories.

    drop : str, default="if_binary"
        Passed to OneHotEncoder.
        "if_binary" automatically drops one category if the column is binary.
        For linear models, drop="first" is often preferred.

    Returns:
    -------
    encoder : OneHotEncoder
        The fitted OneHotEncoder object.

    mappings : dict
        Dictionary containing the applied combine mappings for each column.
        Format:
            {
                "ColumnName": {
                    "combine": {old_val: new_val, ...}
                }
            }

    cols : list of str
        List of columns that were encoded.

    apply_mappings_fn : function or None
        Function to apply the same combine_map to new datasets (like test data).
        If combine_map is None, returns None.

    Usage Example:
    --------------
    combine_map = {
        "Geography": {
            "Germany": ["Germany"],
            "Other": ["Spain", "France", "Italy"]
        }
    }

    encoder, mappings, cols, apply_mappings_fn = get_fitted_one_hot_encoder(
        df_scaled,
        cols=["Gender", "Geography"],
        combine_map=combine_map
    )

    # Apply mappings to a new dataset
    if apply_mappings_fn:
        df_encoded_source = apply_mappings_fn(df_test)
    else:
        df_encoded_source = df_test.copy()

    X_encoded = encoder.transform(df_encoded_source[cols])
    """

    # Make a copy to avoid modifying original dataframe
    df = train_inputs.copy()

    # If no columns provided, select all object-type columns
    if cols is None:
        cols = df.select_dtypes(include="object").columns.tolist()

    mappings = {}

    for col in cols:
        mappings[col] = {}

        # Apply combine_map if provided
        if combine_map and col in combine_map:
            reverse_map = {}
            # Build mapping: each old value -> new combined value
            for new_val, old_vals in combine_map[col].items():
                for val in old_vals:
                    reverse_map[val] = new_val

            # Replace old values in column with new categories
            df[col] = df[col].replace(reverse_map)

            # Store the mapping for reuse on test/new data
            mappings[col]["combine"] = reverse_map

    # Initialize OneHotEncoder
    encoder = OneHotEncoder(
        sparse_output=sparse_output, handle_unknown=handle_unknown, drop=drop
    )

    # Fit encoder to the columns
    encoder.fit(df[cols])

    # If combine_map is provided, define a helper function to apply it to new data
    apply_mappings_fn = None
    if combine_map is not None:

        def apply_mappings_fn(df_new):
            """
            Apply the same category combinations to a new dataframe.
            """
            df_new = df_new.copy()
            for col, rules in mappings.items():
                if "combine" in rules:
                    df_new[col] = df_new[col].replace(rules["combine"])
            return df_new

        return encoder, mappings, cols, apply_mappings_fn

    return encoder


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


def keep_only_top_n(df, col, n):
    """
    Keep only the top N most frequent values in a column,
    replacing all other values with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the column to modify.
    col : str
        The column name to process.
    n : int
        Number of top frequent values to keep.

    Returns
    -------
    pd.DataFrame
        The dataframe with the column modified: only the top N values remain,
        all others are set to NaN.
    """
    df_copy = df.copy()

    # Get top N most frequent values
    top = df_copy[col].value_counts().nlargest(n).index

    # Keep only top N, set others to NaN
    df_copy[col] = df_copy[col].where(df_copy[col].isin(top))

    return df_copy
