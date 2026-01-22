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
