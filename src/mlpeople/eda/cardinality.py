import pandas as pd


def column_cardinality(df: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    """
    Computes cardinality (number of unique values) for each column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    normalize : bool, default False
        If True, returns cardinality as a ratio of total rows

    Returns
    -------
    pd.DataFrame
        Columns:
        - column_id
        - cardinality
        - cardinality_ratio (optional)
    """

    rows = len(df)
    records = []

    for col in df.columns:
        unique_count = df[col].nunique(dropna=True)

        record = {
            "column_id": col,
            "cardinality": unique_count,
        }

        if normalize:
            record["cardinality_ratio"] = round(unique_count / rows, 4) if rows else 0.0

        records.append(record)

    return (
        pd.DataFrame(records)
        .sort_values(by="cardinality", ascending=False)
        .reset_index(drop=True)
    )
