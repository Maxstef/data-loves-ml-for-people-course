import numpy as np


# Define Loss Functions
def predict(theta, X):
    return X @ theta


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def mse_loss_logistic(theta, X, y):
    m = X.shape[0]
    z = predict(theta, X)
    y_pred = sigmoid(z)
    return np.sum((y_pred - y) ** 2) / (2 * m)


def mse_loss_linear(theta, X, y):
    """MSE treating logistic output as linear"""
    m = X.shape[0]
    y_pred = predict(theta, X)
    return np.sum((y_pred - y) ** 2) / (2 * m)


def log_loss(theta, X, y):
    """Logistic loss (cross-entropy)"""
    m = X.shape[0]
    z = predict(theta, X)
    y_pred = sigmoid(z)

    # Clip to avoid log(0)
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)

    return -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / m


# Define Gradients of Loss Functions
def grad_norm_mse_logistic(theta, X, y):
    m = X.shape[0]
    z = predict(theta, X)
    y_pred = sigmoid(z)

    # chain rule
    grad = X.T @ ((y_pred - y) * y_pred * (1 - y_pred)) / m
    return np.linalg.norm(grad)


def grad_norm_mse_linear(theta, X, y):
    grad = X.T @ (predict(theta, X) - y) / X.shape[0]
    return np.linalg.norm(grad)


def grad_norm_log(theta, X, y):
    z = predict(theta, X)
    y_pred = sigmoid(z)
    grad = X.T @ (y_pred - y) / X.shape[0]
    return np.linalg.norm(grad)
