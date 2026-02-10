from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import logging
import operator


def get_fitted_scaler(train_inputs, cols=None, mode="minmax"):
    """
    Fit a scaler to numeric columns of a DataFrame.

    This function allows easy experimentation with different scaling methods
    for numeric features. It returns a fitted scaler that can be used to
    transform training, validation, or test datasets.

    Args:
        train_inputs (pd.DataFrame): The training DataFrame containing numeric features.
        cols (list of str, optional): List of columns to scale. If None, all numeric
            columns in `train_inputs` are used. Defaults to None.
        mode (str, optional): Scaling method to use. Supported values are:
            - "minmax": scales features to [0, 1] using MinMaxScaler
            - "standard": standardizes features to zero mean and unit variance using StandardScaler
            Defaults to "minmax".

    Returns:
        sklearn.preprocessing.MinMaxScaler or sklearn.preprocessing.StandardScaler:
            Fitted scaler object that can be used to transform data via `.transform()` or `.fit_transform()`.

    Raises:
        ValueError: If `mode` is not one of "minmax" or "standard".
    """
    if cols is None:
        cols = train_inputs.select_dtypes(include=np.number).columns.tolist()

    if mode == "minmax":
        scaler = MinMaxScaler()
    elif mode == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError('mode param expected to be "minmax" or "standard"')

    scaler.fit(train_inputs[cols])

    return scaler


OPS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
}


def create_binary_flag(
    train_inputs,
    val_inputs,
    numeric_cols,
    column,
    flag_name=None,
    threshold_func=None,
    threshold_sign="==",
    drop_original=True,
    test_inputs=None,
):
    """
    Create a binary flag column for a numeric column based on a threshold.

    Args:
        train_inputs (pd.DataFrame): Training features.
        val_inputs (pd.DataFrame): Validation features.
        numeric_cols (list): List of numeric column names.
        column (str): Numeric column to create binary flag from.
        flag_name (str, optional): Name of the new binary flag column. Defaults to column+"_flag".
        threshold_func (callable, optional): Function to compute threshold from a Series. Defaults to min().
        drop_original (bool, optional): Whether to drop the original column after creating flag. Defaults to True.
        test_inputs (pd.DataFrame, optional): Optional test features to transform.

    Returns:
        tuple: (train_inputs, val_inputs, test_inputs, numeric_cols)
    """
    if column not in train_inputs.columns or column not in numeric_cols:
        raise ValueError(f"{column} not found in inputs")

    if threshold_sign not in OPS:
        raise ValueError(
            f"Invalid threshold_sign '{threshold_sign}'. Must be one of {list(OPS.keys())}"
        )

    if flag_name is None:
        flag_name = column + "_flag"
    if threshold_func is None:
        threshold_func = lambda x: x.min()

    threshold = threshold_func(train_inputs[column])
    op_func = OPS[threshold_sign]

    def apply_flag(df):
        df[flag_name] = op_func(df[column], threshold).astype(int)
        return df

    train_inputs = apply_flag(train_inputs)
    val_inputs = apply_flag(val_inputs)

    if test_inputs is not None:
        test_inputs = apply_flag(test_inputs)

    numeric_cols.append(flag_name)

    if drop_original:
        numeric_cols.remove(column)
        train_inputs = train_inputs.drop(column, axis=1)
        val_inputs = val_inputs.drop(column, axis=1)
        if test_inputs is not None:
            test_inputs = test_inputs.drop(column, axis=1)

    return train_inputs, val_inputs, test_inputs, numeric_cols
