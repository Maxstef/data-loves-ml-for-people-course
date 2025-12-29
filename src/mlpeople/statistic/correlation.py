def get_correlation_details_filtered(
    df, cols=None, abs_threshold=0.5, upper_abs_threshold=1, target_feature=None
):
    """
    Extract filtered pairwise correlation details from a DataFrame.

    This function computes the Pearson correlation matrix for the selected
    numeric columns and returns a flattened table of feature pairs whose
    absolute correlation values fall within a specified range.

    Self-correlations are excluded. Optionally, results can be restricted
    to correlations involving a specific target feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data.
    cols : list[str] or None, default=None
        List of column names to include in the correlation analysis.
        If None, all numeric columns are used.
    abs_threshold : float, default=0.5
        Minimum absolute correlation value to include.
    upper_abs_threshold : float, default=1
        Maximum absolute correlation value to include.
        Useful for excluding perfect or near-perfect correlations.
    target_feature : str or None, default=None
        If provided, only correlations where `Feature_1` equals
        `target_feature` are returned.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - Feature_1 : str
        - Feature_2 : str
        - Correlation : float

        Rows are sorted by absolute correlation value in descending order.
    """

    if cols is None:
        cols = df.select_dtypes("number").columns.tolist()

    corr = df[cols].corr()

    corr_filtered = (
        corr.stack().reset_index()  # Convert to Series with pairs  # Convert to DataFrame
    )
    corr_filtered.columns = ["Feature_1", "Feature_2", "Correlation"]

    # Keep only correlations above threshold and remove self-correlations
    corr_filtered = corr_filtered[
        (corr_filtered["Feature_1"] != corr_filtered["Feature_2"])
        & (corr_filtered["Correlation"].abs() > abs_threshold)
        & (corr_filtered["Correlation"].abs() <= upper_abs_threshold)
    ]

    # Filter by target_feature if provided
    if target_feature is not None:
        corr_filtered = corr_filtered[(corr_filtered["Feature_1"] == target_feature)]

    # Sort by absolute correlation descending and reset index
    corr_filtered = corr_filtered.sort_values(
        by="Correlation", key=abs, ascending=False
    ).reset_index()

    return corr_filtered.reindex(columns=["Feature_1", "Feature_2", "Correlation"])
