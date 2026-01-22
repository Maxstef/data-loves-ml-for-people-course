import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Iterable, Optional


def linear_predict(X, w, b=0.0):
    """
    Universal linear predictor.

    Parameters
    ----------
    X : array-like (n_samples, n_features) or (n_samples,)
    w : array-like (n_features,) or scalar
    b : float
        Intercept term

    Returns
    -------
    np.ndarray
        Predicted values (n_samples,)
    """
    X = np.asarray(X)
    w = np.asarray(w)

    return X @ w + b if X.ndim > 1 else X * w + b


def estimate_linear(X, w, b=0.0):
    return linear_predict(X, w, b)


def fit_ols(X: np.ndarray, y: np.ndarray, fit_intercept: bool = True):
    """
    Fit a linear regression model using Ordinary Least Squares (OLS).

    Solves:
        min ||y - Xβ||²

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Feature matrix.
    y : ndarray of shape (n,)
        Target vector.
    fit_intercept : bool, default=True
        Whether to include an intercept term.

    Returns
    -------
    beta_hat : ndarray of shape (p + 1,) or (p,)
        Estimated coefficients.
    """

    if fit_intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])

    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

    return beta_hat


def predict(
    X: np.ndarray, beta_hat: np.ndarray, fit_intercept: bool = True
) -> np.ndarray:
    """
    Calculate predicted values using estimated coefficients.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix (without intercept column).
    beta_hat : ndarray of shape (n_features + 1,) or (n_features,)
        Estimated coefficients.
    fit_intercept : bool
        Whether beta_hat includes an intercept.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        Predicted target values.
    """
    if fit_intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    return X @ beta_hat


def fit_ols_and_get_params(
    X: pd.DataFrame,
    y: pd.Series,
    include_cols: Optional[Iterable[str]] = None,
    sort_by_abs: bool = False,
    drop_const: bool = False,
):
    """
    Fit an OLS model with statsmodels.api and return sorted coefficients.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    include_cols : iterable of str, optional
        Subset of columns to consider from X before fitting.
    sort_by_abs : bool, default False
        Sort coefficients by absolute value.
    drop_const: bool, default False
        Exclude const from sorted params

    Returns
    -------
    results : statsmodels.regression.linear_model.RegressionResults
        Fitted OLS results object.
    params : pd.Series
        Sorted model coefficients.
    """

    # Optional column filtering
    if include_cols is not None:
        missing = set(include_cols) - set(X.columns)
        if missing:
            raise KeyError(f"Columns not found in X: {missing}")
        X = X.loc[:, include_cols]

    # -------- Full model --------
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const)
    results = model.fit()

    if drop_const:
        params = results.params.drop("const", errors="ignore")
    else:
        params = results.params

    # Sort coefficients
    if sort_by_abs:
        params = params.sort_values(key=abs, ascending=False)
    else:
        params = params.sort_values(ascending=False)

    return results, params
