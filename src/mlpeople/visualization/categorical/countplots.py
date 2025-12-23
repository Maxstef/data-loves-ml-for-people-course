import matplotlib.pyplot as plt

def draw_countplot(
    df,
    col,
    hue_col,
    normalize=False,
    title=None,
):
    """
    Draw a bar plot showing counts or normalized percentages by category and hue.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data to plot.
    col : str
        Categorical column whose values are counted.
    hue_col : str
        Categorical column used to group bars (hue).
    normalize : bool, default False
        If True, counts are converted to percentages per hue group.
    title : str, optional
        Title of the plot.

    Notes
    -----
    - Bars are grouped by `hue_col` and sorted by the first hue value for consistent ordering.
    - Labels are displayed on top of each bar.
    """

    hue_values = df[hue_col].dropna().unique()

    if normalize:
        data = (
            df.groupby(hue_col)[col]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )
        label_fmt = "{:,.1f}%"
    else:
        data = df.groupby(hue_col)[col].value_counts()
        label_fmt = ''

    ax = (
        data.unstack(hue_col)
        .sort_values(by=hue_values[0], ascending=False)
        .plot.bar(figsize=(8, 5), title=title)
    )

    for container in ax.containers:
        ax.bar_label(container, fmt=label_fmt)

    plt.tight_layout()
    plt.show()


def draw_bi_countplot_target(df, col, hue_col, target_col, normalize=False, titles=None):
    """
    Draw side-by-side bar plots comparing distributions of a categorical variable across two target groups.

    This function creates two bar plots (one per target category defined by `target_col`),
    optionally normalizing counts to percentages. Bars are grouped by `hue_col` and sorted
    by the first hue value for consistent ordering. Labels are displayed on top of each bar.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data to plot.
    col : str
        Categorical column whose values are counted.
    hue_col : str
        Categorical column used to group bars (hue).
    target_col : str
        Categorical column used to split the data into two groups.
        Each unique value produces a separate subplot.
    normalize : bool, default False
        If True, counts are converted to percentages per hue group.
    titles : dict, optional
        Mapping from target column values to subplot titles.
        If a value is not in the dictionary, the value itself is used as the title.

    Notes
    -----
    - Assumes exactly two unique values in `target_col` for side-by-side subplots.
    - Bars are sorted by the first hue value in each subplot for consistent ordering.
    - Out-of-place labels indicate percentages if `normalize=True` or raw counts otherwise.
    """

    if titles is None:
      titles = {}

    main_title = (
        f"Normalized distribution of values by category: {col}"
        if normalize
        else f"Number of values per category {col}"
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))
    fig.suptitle(main_title, fontsize=12, y=1.05)

    for index, val in enumerate(df[target_col].dropna().unique()):
        df_temp = df[df[target_col] == val]
        subplot_title = titles.get(val, val)

        hue_values = df_temp[hue_col].unique()

        if normalize:
            data = (
                df_temp.groupby(hue_col)[col]
                .value_counts(normalize=True)
                .mul(100)
                .round(2)
            )
            label_fmt = "{:,.1f}%"
        else:
            data = df_temp.groupby(hue_col)[col].value_counts()
            label_fmt = ''

        ax = (
            data.unstack(hue_col)
            .sort_values(by=hue_values[0], ascending=False)
            .plot.bar(ax=axes[index], title=subplot_title)
        )

        for container in ax.containers:
            ax.bar_label(container, fmt=label_fmt)
    
    # plt.tight_layout()
    plt.show()


def draw_bi_cat_countplot(df, column, hue_column):
    """
    Draw side-by-side bar plots for a categorical variable,
    showing both normalized percentages and absolute counts
    split by a hue category.

    The function creates two plots:
    1. Normalized distribution (%) of `column` values per `hue_column`
    2. Absolute count of `column` values per `hue_column`

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data.
    column : str
        Name of the categorical column to be analyzed.
    hue_column : str
        Name of the categorical column used to split the data
        (e.g. target variable, segment, class).

    Notes
    -----
    - Bars are sorted by the first unique value of `hue_column`
      for consistent ordering.
    - Percentage values are displayed on the bars in the first plot.
    - Raw counts are displayed on the bars in the second plot.
    """

    unique_hue_values = df[hue_column].unique()

    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(14, 6)

    # -------- Normalized distribution (percentages) --------
    title_normalized = f'Normalized distribution of values by category: {column}'

    proportions = (
        df.groupby(hue_column)[column]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    ax = (
        proportions
        .unstack(hue_column)
        .sort_values(by=unique_hue_values[0], ascending=False)
        .plot.bar(ax=axes[0], title=title_normalized)
    )

    # Annotate percentage values on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.1f}%')

    # -------- Absolute counts --------
    title_counts = f'Number of records by category: {column}'

    counts = df.groupby(hue_column)[column].value_counts()

    ax = (
        counts
        .unstack(hue_column)
        .sort_values(by=unique_hue_values[0], ascending=False)
        .plot.bar(ax=axes[1], title=title_counts)
    )

    # Annotate count values on bars
    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout()
    plt.show()


def uni_cat_target_compare(df, column, hue_column):
    """
    Convenience wrapper for comparing a categorical feature
    against a target or grouping variable using bar plots.

    This function simply delegates to `draw_bi_cat_countplot`
    and exists for semantic clarity when analyzing target variables.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    column : str
        Name of the categorical feature to analyze.
    hue_column : str
        Name of the target or grouping column.
    """

    draw_bi_cat_countplot(df, column, hue_column=hue_column)
