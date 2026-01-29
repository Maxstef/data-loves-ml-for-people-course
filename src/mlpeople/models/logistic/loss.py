import numpy as np


def sigmoid(z):
    """
    Numerically stable sigmoid function.
    """
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def hypothesis(theta, X):
    """
    Compute predicted probabilities P(y=1 | X, theta).
    Assumes X already includes a bias column.
    """
    z = X @ theta
    return sigmoid(z)


def log_loss(y_true, y_pred, eps: float = 1e-15):
    """
    Binary cross-entropy (logistic loss).

    Parameters
    ----------
    y_true : np.ndarray of shape (m,)
        True binary labels {0, 1}
    y_pred : np.ndarray of shape (m,)
        Predicted probabilities
    eps : float
        Small value to avoid log(0)

    Returns
    -------
    float
        Mean log loss
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)

    loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

    return np.mean(loss)


def compute_gradient(theta, X, y):
    """
    Compute gradient of log-loss w.r.t. parameters.
    """
    m = X.shape[0]
    probs_predicted = hypothesis(theta, X)
    diff = probs_predicted - y
    return (X.T @ diff) / m
