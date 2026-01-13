import numpy as np


def rmse(predictions, targets):
    diff = targets - predictions
    return np.sqrt(np.mean(np.square(diff)))


def rmse_for_params(X, y, w, b=0.0, predict_fn=None):
    """
    Compute RMSE for given parameters.

    Parameters
    ----------
    X : array-like
        Features (1D or 2D)
    y : array-like
        True target values
    w : array-like or scalar
        Model weights
    b : float
        Bias term
    predict_fn : callable, optional
        Prediction function. Defaults to linear_predict.

    Returns
    -------
    float
        RMSE value
    """
    if predict_fn is None:
        from .model import linear_predict

        predict_fn = linear_predict

    predictions = predict_fn(X, w, b)
    return rmse(predictions, y)


def rmse_df_for_params(df, w, b=0.0, features=[], target=None):
    """
    Compute RMSE for given parameters using features and target from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing features and target.
    w : array-like or float
        Weight(s) for the feature(s).
    b : float, default 0.0
        Bias/intercept term.
    features : list of str
        Column name(s) for feature(s).
    target : str
        Column name for the target.

    Returns
    -------
    float
        Root Mean Squared Error (RMSE) between predicted and actual values.
    """
    X = df[features]
    y = df[target]

    return rmse_for_params(X, y, w, b)
