import numpy as np


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
