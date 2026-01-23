import numpy as np
from .calculus import gradient_function


def batch_gradient_descent(
    loss_f,
    init_w,
    lr=0.01,
    epochs=1000,
    epsilon=1e-6,
    return_history=False,
    grad_f=None,
    h=1e-5,
    central=True,
    divergence_threshold=1e12,
):
    """
    Perform batch gradient descent optimization with optional numerical gradients
    and early stopping.

    The algorithm iteratively updates parameters using:

        w_{k+1} = w_k - lr * grad(loss_f)(w_k)

    Optimization stops early if the gradient norm falls below `epsilon`.

    Supports both:
    - Scalar optimization (1D problem)
    - Vector optimization (multivariate problem)

    Parameters
    ----------
    loss_f : callable
        Objective (loss) function to minimize.
        Must accept a scalar or 1D NumPy array and return a scalar.
    init_w : float or array-like
        Initial parameter value (scalar) or parameter vector.
    lr : float, optional
        Learning rate (step size). Must be positive.
    epochs : int, optional
        Maximum number of optimization iterations.
    epsilon : float, optional
        Convergence threshold on the gradient norm.
    return_history : bool, optional
        If True, return optimization history as a list of tuples:
        (parameters, loss value, gradient norm).
    grad_f : callable, optional
        Gradient function. If None, a numerical gradient is constructed.
    h : float, optional
        Step size for numerical differentiation (used only if grad_f is None).
    central : bool, optional
        Whether to use central differences for numerical gradients.
    divergence_threshold : float, optional
        If the loss, gradient norm, or parameters exceed this value (or become NaN/Inf),
        optimization is considered divergent and a FloatingPointError is raised.

    Returns
    -------
    w : float or np.ndarray
        Optimized parameter(s).
    history : list, optional
        Returned only if return_history=True.

    Raises
    ------
    FloatingPointError
        If divergence or numerical instability is detected.
    ValueError
        If invalid hyperparameters are provided.
    """

    # -------------------- Parameter validation --------------------
    if lr <= 0:
        raise ValueError("Learning rate must be positive")

    if epochs <= 0:
        raise ValueError("epochs must be positive")

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    # -------------------- Gradient setup --------------------
    if grad_f is None:
        grad_f = gradient_function(loss_f, h=h, central=central)

    # Convert initial parameters to NumPy array for uniform handling
    w = np.asarray(init_w, dtype=float)
    is_scalar = w.ndim == 0

    history = []

    # -------------------- Optimization loop --------------------
    for _ in range(epochs):
        grad = grad_f(w)

        # Compute gradient norm (scalar or vector case)
        grad_norm = abs(grad) if is_scalar else np.linalg.norm(grad)

        loss_val = loss_f(w)

        # -------------------- Divergence checks --------------------
        if (
            not np.isfinite(loss_val)
            or not np.isfinite(grad_norm)
            or grad_norm > divergence_threshold
            or np.any(np.abs(w) > divergence_threshold)
        ):
            raise FloatingPointError("Divergence detected")

        # -------------------- Logging (pre-update, consistent state) --------------------
        if return_history:
            history.append((w.item() if is_scalar else w.copy(), loss_val, grad_norm))

        # -------------------- Convergence check --------------------
        if grad_norm < epsilon:
            break

        # -------------------- Gradient descent update --------------------
        w = w - lr * grad

    # Convert back to Python scalar if needed
    if is_scalar:
        w = w.item()

    if return_history:
        return w, history

    return w


def stochastic_gradient_descent(
    X,
    y,
    init_w,
    lr=0.01,
    epochs=100,
    batch_size=1,
    epsilon=1e-6,
    loss_fn=None,
    return_history=False,
    record_batch_loss=False,
):
    """
    Perform stochastic (mini-batch) gradient descent using numerical gradients.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    init_w : array-like
        Initial weight vector.
    lr : float
        Learning rate.
    epochs : int
        Number of passes over the dataset.
    batch_size : int
        Size of mini-batches (1 = pure SGD).
    epsilon : float
        Stop if gradient norm < epsilon.
    loss_fn : callable, optional
        Custom loss function with signature:
            loss_fn(W, X_batch, y_batch) -> scalar
        If None, mean squared error is used.
    return_history : bool
        If True, return optimization history.

    Returns
    -------
    W : np.ndarray
        Optimized weights.
    history : list, optional
        List of (W, loss, grad_norm) tuples recorded per epoch.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    n_samples, n_features = X.shape

    if y.shape[0] != n_samples:
        raise ValueError("X and y must have the same number of samples")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    W = np.asarray(init_w, dtype=float)
    history = []
    batch_losses = []

    # Default loss: Mean Squared Error
    if loss_fn is None:

        def loss_fn(W, Xb, yb):
            preds = Xb @ W
            return np.mean((preds - yb) ** 2)

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        last_grad_norm = None

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Wrap batch loss for numerical gradient
            def batch_loss(w):
                return loss_fn(w, X_batch, y_batch)

            grad = gradient_function(batch_loss)(W)
            grad_norm = np.linalg.norm(grad)
            last_grad_norm = grad_norm

            if grad_norm < epsilon:
                break

            W -= lr * grad

            if record_batch_loss:
                batch_losses.append(loss_fn(W, X_batch, y_batch))

        if return_history:
            full_loss = loss_fn(W, X, y)
            history.append((W.copy(), full_loss, last_grad_norm))

        if last_grad_norm is not None and last_grad_norm < epsilon:
            break

    if record_batch_loss:
        return W, history, batch_losses

    if return_history:
        return W, history
    return W
