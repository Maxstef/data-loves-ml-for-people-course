from .loss import hypothesis, log_loss, compute_gradient
import numpy as np


def predict_proba(theta, X):
    """
    Compute probabilities P(y=1 | X, theta)
    """
    return hypothesis(theta, X)


def predict(theta, X, threshold=0.5):
    """
    Convert probabilities to binary predictions
    """
    return (predict_proba(theta, X) >= threshold).astype(int)


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    epochs: int = 1000,
    method: str = "batch",
    fit_intercept: bool = True,
    batch_size: int | None = None,
    verbose: bool = False,
):
    """
    Fit logistic regression using gradient descent (batch or mini-batch).

    Parameters
    ----------
    X : np.ndarray, shape (m, n)
        Feature matrix, without bias.
    y : np.ndarray, shape (m,)
        Binary labels {0,1}.
    lr : float
        Learning rate.
    epochs : int
        Number of iterations.
    method : str
        'batch' or 'stochastic' (mini-batch)
    fit_intercept : bool
        If True, adds a bias column.
    batch_size : int | None
        Size of mini-batch for stochastic GD. Defaults to 1 for 'stochastic'.
    verbose : bool
        Print training progress.

    Returns
    -------
    theta : np.ndarray, shape (n + 1,)
        Learned parameters.
    history : list of tuples
        Each tuple = (theta copy, log-loss, gradient norm)
    """
    m, n = X.shape

    # Add bias column if requested
    if fit_intercept:
        X = np.hstack([np.ones((m, 1)), X])

    theta = np.zeros(X.shape[1])
    history = []

    if method not in {"batch", "stochastic"}:
        raise ValueError("method must be 'batch' or 'stochastic'")

    if method == "stochastic" and batch_size is None:
        batch_size = 1  # default SGD

    for epoch in range(epochs):
        if method == "batch":
            grad = compute_gradient(theta, X, y)
            theta -= lr * grad

            # compute full loss and gradient norm
            full_loss = log_loss(y, predict_proba(theta, X))
            grad_norm = np.linalg.norm(grad)
            history.append((theta.copy(), full_loss, grad_norm))

        elif method == "stochastic":
            indices = np.random.permutation(m)

            # mini-batch iteration
            for start in range(0, m, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                X_batch = X[batch_idx, :]
                y_batch = y[batch_idx]

                grad = compute_gradient(theta, X_batch, y_batch)
                theta -= lr * grad

            # track history at the end of epoch
            full_loss = log_loss(y, predict_proba(theta, X))
            grad_norm = np.linalg.norm(grad)
            history.append((theta.copy(), full_loss, grad_norm))

        if verbose and (epoch % max(1, epochs // 10) == 0):
            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {full_loss:.4f}, Grad Norm: {grad_norm:.4f}"
            )

    return theta, history
