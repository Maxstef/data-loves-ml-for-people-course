import numpy as np
from scipy import stats


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Logarithmic Error.

    Notes:
    - y_true and y_pred must be >= 0
    - Use when target is strictly positive and you want to penalize under/over predictions in relative terms
    """
    # Clip negative values to 0 to avoid log of negative numbers
    y_true_clip = np.clip(y_true, 0, None)
    y_pred_clip = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_true_clip) - np.log1p(y_pred_clip)) ** 2))


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


def t_test_coefficients(X, y, beta_hat, fit_intercept=True):
    """
    Compute t-statistics and p-values for OLS coefficients.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix (without intercept)
    y : ndarray of shape (n_samples,)
        Target vector
    beta_hat : ndarray of shape (p + 1,)
        Estimated coefficients (include intercept if fit_intercept=True)
    fit_intercept : bool
        Whether beta_hat includes intercept

    Returns
    -------
    t_stats : ndarray
        t-statistics for each coefficient
    p_values : ndarray
        Two-sided p-values for each coefficient
    """
    n_samples, n_features = X.shape

    # Add intercept if needed
    if fit_intercept:
        X_design = np.hstack([np.ones((n_samples, 1)), X])
    else:
        X_design = X

    # Predictions and residuals
    y_pred = X_design @ beta_hat
    residuals = y - y_pred

    # Estimate of variance of residuals
    df = n_samples - X_design.shape[1]
    sigma_hat_sq = np.sum(residuals**2) / df

    # Covariance matrix of beta_hat
    XtX_inv = np.linalg.inv(X_design.T @ X_design)
    se_beta = np.sqrt(np.diag(sigma_hat_sq * XtX_inv))

    # t-statistics
    t_stats = beta_hat / se_beta

    # two-sided p-values
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df))

    return t_stats, p_values
