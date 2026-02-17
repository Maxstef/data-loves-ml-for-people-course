import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder


def get_fitted_imputer(train_inputs, cols=None, strategy="mean"):
    """
    Fit a SimpleImputer on specified columns of the training data.

    Parameters
    ----------
    train_inputs : pandas.DataFrame
        Training feature dataframe.
    cols : list of str, optional
        Columns to apply imputation to. If None, all numeric columns are used.
    strategy : str, default="mean"
        Imputation strategy (e.g., "mean", "median", "most_frequent", "constant").

    Returns
    -------
    sklearn.impute.SimpleImputer
        Fitted imputer instance.
    """
    if cols is None:
        cols = train_inputs.select_dtypes(include=np.number).columns.tolist()

    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(train_inputs[cols])

    return imputer


def get_fitted_scaler(train_inputs, cols=None, mode="minmax"):
    """
    Fit a scaler (MinMaxScaler or StandardScaler) on specified columns.

    Parameters
    ----------
    train_inputs : pandas.DataFrame
        Training feature dataframe.
    cols : list of str, optional
        Columns to scale. If None, all numeric columns are used.
    mode : {"minmax", "standard"}, default="minmax"
        Scaling mode.

    Returns
    -------
    sklearn.preprocessing.MinMaxScaler or sklearn.preprocessing.StandardScaler
        Fitted scaler instance.

    Raises
    ------
    ValueError
        If an unsupported mode is provided.
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


def get_fitted_one_hot_encoder(
    train_inputs,
    cols=None,
    sparse_output=False,
    handle_unknown="ignore",
    drop="if_binary"
):
    """
    Fit a OneHotEncoder on specified categorical columns.

    Parameters
    ----------
    train_inputs : pandas.DataFrame
        Training feature dataframe.
    cols : list of str, optional
        Categorical columns to encode. If None, all object columns are used.
    sparse_output : bool, default=False
        Whether to return sparse matrix output.
    handle_unknown : str, default="ignore"
        How to handle unknown categories during transform.
    drop : str or None, default="if_binary"
        Strategy for dropping categories (e.g., "if_binary", "first", None).

    Returns
    -------
    sklearn.preprocessing.OneHotEncoder
        Fitted encoder instance.
    """
    if cols is None:
        cols = train_inputs.select_dtypes(include="object").columns.tolist()
        
    encoder = OneHotEncoder(
        sparse_output=sparse_output,
        handle_unknown=handle_unknown,
        drop=drop
    )

    encoder.fit(train_inputs[cols])

    return encoder

def split_train_val(df, test_size=0.2, random_state=42, stratify_col=None):
    """
    Split a dataframe into train and validation sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    test_size : float, default=0.2
        Proportion of data to include in validation set.
    random_state : int, default=42
        Random seed for reproducibility.
    stratify_col : str, optional
        Column name used for stratified splitting.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        Train and validation dataframes.
    """
    if stratify_col is not None:
        return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[stratify_col])
    else:
        return train_test_split(df, test_size=test_size, random_state=random_state)


def split_inputs_targets(df, target_col, drop_cols):
    """
    Separate input features and target column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    target_col : str
        Name of the target column.
    drop_cols : list of str
        Columns to exclude from input features.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.Series)
        Feature dataframe and target series.
    """
    drop_from_input_cols = [target_col] + drop_cols
    input_cols = df.columns.drop(drop_from_input_cols)

    inputs = df[input_cols].copy()
    targets = df[target_col].copy()

    return inputs, targets

def get_column_types(df):
    """
    Identify numeric and categorical columns in a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    tuple of (list, list)
        List of numeric columns and list of categorical columns.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    return numeric_cols, categorical_cols


def apply_imputer(train_df, val_df, cols, strategy="mean"):
    """
    Fit an imputer on training data and apply to train and validation sets.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training feature dataframe.
    val_df : pandas.DataFrame
        Validation feature dataframe.
    cols : list of str
        Columns to impute.
    strategy : str, default="mean"
        Imputation strategy.

    Returns
    -------
    tuple
        (train_df, val_df, fitted_imputer)
    """
    imputer = get_fitted_imputer(train_df, cols=cols, strategy=strategy)

    train_df[cols] = imputer.transform(train_df[cols])
    val_df[cols] = imputer.transform(val_df[cols])

    return train_df, val_df, imputer

def apply_scaler(train_df, val_df, cols, mode="minmax"):
    """
    Fit a scaler on training data and apply to train and validation sets.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training feature dataframe.
    val_df : pandas.DataFrame
        Validation feature dataframe.
    cols : list of str
        Columns to scale.
    mode : {"minmax", "standard"}, default="minmax"
        Scaling mode.

    Returns
    -------
    tuple
        (train_df, val_df, fitted_scaler)
    """
    scaler = get_fitted_scaler(train_df, cols=cols, mode=mode)

    train_df[cols] = scaler.transform(train_df[cols])
    val_df[cols] = scaler.transform(val_df[cols])

    return train_df, val_df, scaler

def apply_encoder(train_df, val_df, cols, drop="if_binary"):
    """
    Fit a OneHotEncoder on training data and apply to train and validation sets.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training feature dataframe.
    val_df : pandas.DataFrame
        Validation feature dataframe.
    cols : list of str
        Categorical columns to encode.
    drop : str or None, default="if_binary"
        Category dropping strategy.

    Returns
    -------
    tuple
        (train_df, val_df, fitted_encoder)
    """
    encoder = get_fitted_one_hot_encoder(train_df, cols=cols, drop=drop)

    train_encoded = encoder.transform(train_df[cols])
    val_encoded = encoder.transform(val_df[cols])

    encoded_cols = encoder.get_feature_names_out(cols)

    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=encoded_cols,
        index=train_df.index
    )

    val_encoded_df = pd.DataFrame(
        val_encoded,
        columns=encoded_cols,
        index=val_df.index
    )

    train_df = pd.concat([train_df.drop(columns=cols), train_encoded_df], axis=1)
    val_df = pd.concat([val_df.drop(columns=cols), val_encoded_df], axis=1)

    return train_df, val_df, encoder


def preprocess_data(
    raw_df,
    test_size=0.2,
    target_col="Exited",
    stratify_col="Exited",
    drop_cols=None,
    scaler_numeric=True,
    scaler_mode="minmax",
    encoder_drop="if_binary",
    random_state=42
):
    """
    Full preprocessing pipeline: split, impute, scale, and encode features.

    Parameters
    ----------
    raw_df : pandas.DataFrame
        Original dataset.
    test_size : float, default=0.2
        Validation split size.
    target_col : str, default="Exited"
        Target column name.
    stratify_col : str, default="Exited"
        Column used for stratified splitting.
    drop_cols : list of str, optional
        Columns to drop from features.
    scaler_numeric : bool, default=True
        Whether to scale numeric features.
    scaler_mode : {"minmax", "standard"}, default="minmax"
        Scaling mode.
    encoder_drop : str or None, default="if_binary"
        OneHotEncoder drop strategy.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    dict
        Dictionary containing processed datasets and fitted transformers:
        {
            "X_train",
            "train_targets",
            "X_val",
            "val_targets",
            "input_cols",
            "imputer",
            "scaler",
            "encoder",
        }
    """
    if drop_cols is None:
        drop_cols = ["CustomerId", "Surname"]

    # 1. Split train / validation
    train_df, val_df = split_train_val(
        raw_df,
        test_size=test_size,
        random_state=random_state,
        stratify_col=stratify_col
    )

    # 2. Separate inputs & targets
    train_inputs, train_targets = split_inputs_targets(train_df, target_col, drop_cols)
    val_inputs, val_targets = split_inputs_targets(val_df, target_col, drop_cols)

    # 3. Detect column types
    numeric_cols, categorical_cols = get_column_types(train_inputs)

    # 4. Impute numeric
    train_inputs, val_inputs, imputer = apply_imputer(
        train_inputs, val_inputs, numeric_cols
    )

    # 5. Scale numeric
    if scaler_numeric:
        train_inputs, val_inputs, scaler = apply_scaler(
            train_inputs, val_inputs, numeric_cols, mode=scaler_mode
        )
    else:
        scaler = None

    # 6. Encode categorical
    train_inputs, val_inputs, encoder = apply_encoder(
        train_inputs, val_inputs, categorical_cols, drop=encoder_drop
    )

    return {
        "X_train": train_inputs,
        "train_targets": train_targets,
        "X_val": val_inputs,
        "val_targets": val_targets,
        "input_cols": train_inputs.columns.tolist(),
        "imputer": imputer,
        "scaler": scaler,
        "encoder": encoder,
    }
