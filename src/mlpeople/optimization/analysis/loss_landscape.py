import numpy as np

# ==============================
# Linear Regression
# ==============================


def linear_predict(theta, X):
    """
    Linear regression prediction: y_hat = X @ theta
    X may include bias column.
    """
    return X @ theta


def mse_loss_linear(theta, X, y):
    """
    Mean Squared Error loss (linear regression)
    L(theta) = 1/(2m) * sum((y_hat - y)^2)
    """
    m = X.shape[0]
    y_pred = linear_predict(theta, X)
    return np.sum((y_pred - y) ** 2) / (2 * m)


def mse_loss_linear_yhat(y_true, y_pred):
    return np.sum((y_pred - y_true) ** 2) / (2 * len(y_true))


def grad_mse_linear(theta, X, y):
    """
    Gradient of MSE for linear regression
    """
    m = X.shape[0]
    grad = X.T @ (linear_predict(theta, X) - y) / m
    return grad


def grad_norm_mse_linear(theta, X, y):
    """
    Norm of gradient of MSE (useful for plotting)
    """
    return np.linalg.norm(grad_mse_linear(theta, X, y))


# ==============================
# Logistic Regression
# ==============================


def sigmoid(z):
    """
    Standard sigmoid function
    """
    return 1 / (1 + np.exp(-z))


def logistic_predict_proba(theta, X):
    """
    Logistic regression predicted probability: sigmoid(X @ theta)
    """
    return sigmoid(X @ theta)


def mse_loss_logistic(theta, X, y):
    """
    MSE treating logistic outputs as linear (for demonstration)
    L(theta) = 1/(2m) * sum((y_hat - y)^2), where y_hat = sigmoid(X @ theta)
    """
    m = X.shape[0]
    y_pred = logistic_predict_proba(theta, X)
    return np.sum((y_pred - y) ** 2) / (2 * m)


def grad_mse_logistic(theta, X, y):
    """
    Gradient of MSE for logistic regression using chain rule
    grad = X.T @ ((y_hat - y) * y_hat * (1 - y_hat)) / m
    """
    m = X.shape[0]
    y_pred = logistic_predict_proba(theta, X)
    grad = X.T @ ((y_pred - y) * y_pred * (1 - y_pred)) / m
    return grad


def grad_norm_mse_logistic(theta, X, y):
    """
    Norm of gradient of MSE logistic
    """
    return np.linalg.norm(grad_mse_logistic(theta, X, y))


def log_loss(theta, X, y):
    """
    Logistic loss (cross-entropy)
    L(theta) = -1/m * sum(y*log(y_hat) + (1-y)*log(1-y_hat))
    """
    m = X.shape[0]
    y_pred = logistic_predict_proba(theta, X)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def log_loss_yhat(y, y_hat):
    """
    Logistic loss (cross-entropy) with two parameters:
    - y: true labels
    - y_hat: predicted probabilities
    """
    # Clip to avoid log(0)
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)

    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def grad_log_loss(theta, X, y):
    """
    Gradient of logistic loss
    """
    m = X.shape[0]
    y_pred = logistic_predict_proba(theta, X)
    grad = X.T @ (y_pred - y) / m
    return grad


def grad_norm_log(theta, X, y):
    """
    Norm of gradient of logistic loss
    """
    return np.linalg.norm(grad_log_loss(theta, X, y))
