import pandas as pd
import numpy as np

def get_all_bool_flag_cols(df: pd.DataFrame):
    """
    Identify candidate boolean/flag columns in a DataFrame.

    A column is considered a flag column if:
    - It contains at least one non-null value.
    - After normalizing non-null values (string conversion, trimming, case folding),
      it has exactly two unique values.

    This function does NOT validate semantic boolean meaning (e.g. yes/no);
    it only detects structural suitability for boolean optimization.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.

    Returns
    -------
    list[str]
        List of column names that qualify as boolean/flag candidates.

    Notes
    -----
    - Columns with all-null values are ignored.
    - Type coercion is not performed here; normalization is used only
      for uniqueness detection.
    """

    flag_cols = []

    for col in df.columns:
        s = df[col]

        # Ignore columns that are entirely null
        if s.notna().sum() == 0:
            continue

        # Normalize non-null values
        normalized = (
            s.dropna()
             .astype(str)
             .str.strip()
             .str.casefold()
        )

        unique = normalized.unique()

        # Must have exactly 2 unique non-null values
        if len(unique) != 2:
            continue

        flag_cols.append(col)

    return flag_cols



def optimize_bool_flag_cols(df: pd.DataFrame, flag_cols):
    """
    Normalize and optimize boolean/flag columns to int8 (0/1).

    For each column in `flag_cols`, values are normalized and converted
    into integer flags:
    - `1` represents the inferred TRUE value
    - `0` represents the inferred FALSE value
    - NaN values are preserved

    Conversion rules:
    - Integer columns must contain only {0, 1}; otherwise an error is raised.
    - Known boolean-like values (e.g. yes/no, true/false, y/n, 1/0)
      are mapped using predefined sets.
    - Arbitrary two-value columns are deterministically mapped by sorting
      normalized values and choosing the first as TRUE.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing flag columns.
    flag_cols : Iterable[str]
        Column names expected to be boolean/flag columns.
        Typically produced by `get_all_bool_flag_cols`.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with optimized flag columns converted to int8.

    Raises
    ------
    ValueError
        If a column does not contain exactly two non-null unique values,
        or if an integer column contains values other than 0 or 1.

    Notes
    -----
    - The original DataFrame is not mutated.
    - Mapping is deterministic but may not be semantically correct
      for arbitrary value pairs; review results if semantics matter.
    - Resulting columns use pandas nullable semantics for NaN preservation.
    """

    TRUE_VALUES = {"y", "yes", "true", "1", "t"}
    FALSE_VALUES = {"n", "no", "false", "0", "f"}

    df = df.copy()

    for col in flag_cols:
        series = df[col]

        normalized = (
            series.dropna()
             .astype(str)
             .str.strip()
             .str.casefold()
        )

        unique = normalized.unique()

        if len(unique) != 2:
            raise ValueError(
                f"Column '{col}' must contain exactly 2 non-null unique values, got {unique}"
            )

        values = set(unique)

        # Integer flags: allow only 0/1
        if pd.api.types.is_integer_dtype(series):
            if not set(series.dropna().unique()).issubset({0, 1}):
                raise ValueError(
                    f"Column '{col}' contains non-boolean integers: {values}"
                )
            df[col] = series.astype("int8")
            continue

        # Known boolean values
        if values.issubset(TRUE_VALUES | FALSE_VALUES):
            true_value = next(iter(values & TRUE_VALUES))

        # Arbitrary two values → deterministic
        else:
            true_value = sorted(values)[0]

        df[col] = np.where(
            series.isna(),
            np.nan,
            series.astype(str)
             .str.strip()
             .str.casefold()
             .eq(true_value)
             .astype("int8")
        )

    return df
