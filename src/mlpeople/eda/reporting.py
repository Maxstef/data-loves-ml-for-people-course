import pandas as pd
import numpy as np


def describe_dataframe(df: pd.DataFrame, skip_cols=None) -> pd.DataFrame:
    """
    Returns a summary of each column in the DataFrame, including:
    - column id
    - data type
    - number of unique values
    - number of missing values
    - numeric statistics (mean, std, min, max) for numeric columns
    """

    if skip_cols is None:
        skip_cols = []

    summary = []

    for col in df.columns:
        if col in skip_cols:
            continue

        col_data = df[col]

        col_info = {
            "column_id": col,
            "data_type": col_data.dtype,
            "unique_values": col_data.nunique(dropna=True),
            "missing_values": col_data.isna().sum(),
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

        if pd.api.types.is_numeric_dtype(col_data):
            col_info.update(
                {
                    "mean": col_data.mean().round(2),
                    "std": col_data.std().round(2),
                    "min": col_data.min().round(2),
                    "max": col_data.max().round(2),
                }
            )

        summary.append(col_info)

    return pd.DataFrame(summary)
