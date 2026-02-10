from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Optional, Tuple


def split_train_val(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train and validation sets.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset to split.
    test_size : float, default=0.2
        Fraction of data to use as validation set.
    random_state : int, default=42
        Random seed for reproducibility.
    stratify_col : str or None, default=None
        Column name to stratify on (useful for classification).

    Returns
    -------
    train_df : pd.DataFrame
        Training set.
    val_df : pd.DataFrame
        Validation set.
    """
    if stratify_col is not None:
        stratify_vals = df[stratify_col]
    else:
        stratify_vals = None

    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify_vals
    )
    return train_df, val_df


def split_input_target(train_df, val_df, target_col, drop_cols=[], verbose=True):
    if target_col not in drop_cols:
        drop_cols.append(target_col)

    input_cols = train_df.columns.drop(drop_cols)

    if verbose:
        print(f"target col: {target_col}")
        print(f"input cols: {input_cols}")

    train_inputs = train_df[input_cols].copy()
    train_targets = train_df[target_col].copy()

    val_inputs = val_df[input_cols].copy()
    val_targets = val_df[target_col].copy()

    return train_inputs, train_targets, val_inputs, val_targets
