import numpy as np
import matplotlib.pyplot as plt

from mlpeople.math.optimization import (
    batch_gradient_descent,
    stochastic_gradient_descent,
)
from mlpeople.math.linear_numeric import add_bias_column
from mlpeople.visualization.optimization.gd import (
    visualize_linear_regression_gd,
    plot_sgd_batch_vs_epoch,
)
from mlpeople.models.linear import generate_linear_regression_data


def run_gd_and_visualize(
    X,
    y,
    loss_fn,
    beta_true=None,
    init_w=None,
    lr=0.01,
    epsilon=1e-6,
    max_epochs=1000,
    plot_predictions=False,
):
    """
    Run batch gradient descent on a linear regression problem and visualize results.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix including bias column (n_samples, n_features)
    y : np.ndarray
        Target values (n_samples,)
    loss_fn : callable
        Loss function to minimize: loss_fn(w) -> scalar
    beta_true : array-like, optional
        True coefficients (including intercept if X has bias column). Used for reference in plots.
    init_w : array-like, optional
        Initial weights for gradient descent. Default: zeros of shape X.shape[1].
    lr : float, optional
        Learning rate.
    epsilon : float, optional
        Convergence threshold on gradient norm.
    max_epochs : int, optional
        Maximum number of iterations for gradient descent.
    plot_predictions : bool, optional
        If True, plot predicted y vs true y. Default: False.

    Returns
    -------
    W_opt : np.ndarray
        Final optimized weights.
    history : list
        Gradient descent history (w, loss, grad_norm) for each step.
    """

    # ---------------- Prepare initial weights ----------------
    n_features = X.shape[1]
    if init_w is None:
        init_w = np.zeros(n_features, dtype=float)
    else:
        init_w = np.array(init_w, dtype=float)

    # ---------------- Run gradient descent ----------------
    W_opt, history = batch_gradient_descent(
        loss_fn,
        init_w=init_w,
        lr=lr,
        epsilon=epsilon,
        epochs=max_epochs,
        return_history=True,
    )

    # ---------------- Visualization ----------------
    visualize_linear_regression_gd(
        X, y, beta_true=beta_true, history=history, plot_predictions=plot_predictions
    )

    return W_opt, history


def run_and_plot_sgd_batch_vs_epoch(
    true_coefs=[10.0, -5.0, 0.5],
    true_intercept=20.0,
    total_size=100,
    batch_size=20,
    epochs=6,
    lr=0.05,
    sigma=2.0,
    seed=42,
    smooth=False,
    window_threshold=None,
):
    """
    Generate linear regression data, run SGD, and visualize
    mini-batch loss vs epoch loss.

    Parameters
    ----------
    true_coefs : list of float
        True regression coefficients (without intercept).
    true_intercept : float
        True intercept term.
    total_size : int
        Number of samples.
    batch_size : int
        Mini-batch size for SGD.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    sigma : float
        Noise standard deviation.
    seed : int
        Random seed.
    smooth : bool, default True
        If True, plot smoothed mini-batch loss.
    window_threshold : int, optional
        Window size for smoothing. If None, calculated automatically.
    """

    # -------- Generate data --------
    X_raw, y, beta_true = generate_linear_regression_data(
        n=total_size,
        p=len(true_coefs),
        coefs=true_coefs,
        intercept=true_intercept,
        sigma=sigma,
        seed=seed,
    )

    X = add_bias_column(X_raw)

    # -------- Loss --------
    def mse_loss(W, X, y):
        return np.mean((X @ W - y) ** 2)

    # -------- SGD --------
    W, epoch_hist, batch_losses = stochastic_gradient_descent(
        X,
        y,
        init_w=np.zeros(X.shape[1]),
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        loss_fn=mse_loss,
        return_history=True,
        record_batch_loss=True,
    )

    epoch_losses = [h[1] for h in epoch_hist]

    # -------- Plot --------
    title = (
        f"SGD: Mini-batch Loss vs Epoch Loss\n"
        f"(total_size={total_size}, batch_size={batch_size}, "
        f"epochs={epochs}, lr={lr})"
    )

    plot_sgd_batch_vs_epoch(
        batch_losses,
        epoch_losses,
        title=title,
        smooth=smooth,
        window_threshold=window_threshold,
    )

    return W, beta_true


def compare_and_visualize_gd(
    X,
    y,
    loss_fn,
    init_w,
    lr_batch=0.05,
    lr_sgd=0.05,
    epochs=200,
    batch_size=1,
    epsilon=1e-6,
):
    """
    Compare Batch Gradient Descent and Stochastic Gradient Descent visually.

    Produces two plots:
    1) Loss vs iteration
    2) Parameter norm vs iteration

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target vector (n_samples,)
    loss_fn : callable
        Loss function: loss_fn(W, X, y) -> scalar
    init_w : array-like
        Initial weights
    lr_batch : float
        Learning rate for batch gradient descent
    lr_sgd : float
        Learning rate for stochastic gradient descent
    epochs : int
        Maximum number of epochs
    batch_size : int
        Mini-batch size for SGD (1 = pure SGD)
    epsilon : float
        Convergence threshold

    Returns
    -------
    W_batch : np.ndarray
        Final weights from batch gradient descent
    W_sgd : np.ndarray
        Final weights from stochastic gradient descent
    """

    # ---------------- Batch Gradient Descent ----------------
    W_batch, hist_batch = batch_gradient_descent(
        lambda w: loss_fn(w, X, y),
        init_w=init_w,
        lr=lr_batch,
        epochs=epochs,
        epsilon=epsilon,
        return_history=True,
    )

    batch_losses = [h[1] for h in hist_batch]
    batch_norms = [np.linalg.norm(h[0]) for h in hist_batch]

    # ---------------- Stochastic Gradient Descent ----------------
    W_sgd, hist_sgd = stochastic_gradient_descent(
        X,
        y,
        init_w=init_w,
        lr=lr_sgd,
        epochs=epochs,
        batch_size=batch_size,
        epsilon=epsilon,
        loss_fn=loss_fn,
        return_history=True,
    )

    sgd_losses = [h[1] for h in hist_sgd]
    sgd_norms = [np.linalg.norm(h[0]) for h in hist_sgd]
    total_size = len(y)

    # ---------------- Loss Plot ----------------
    plt.figure()
    plt.plot(batch_losses, label=f"Batch GD (size={total_size}, lr={lr_batch})")
    plt.plot(sgd_losses, label=f"SGD (batch_size={batch_size}, lr={lr_sgd})")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs Iteration")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------------- Parameter Norm Plot ----------------
    plt.figure()
    plt.plot(batch_norms, label=f"Batch GD (size={total_size}, lr={lr_batch})")
    plt.plot(sgd_norms, label=f"SGD (batch_size={batch_size}, lr={lr_sgd})")
    plt.xlabel("Iteration")
    plt.ylabel("‖W‖")
    plt.title("Parameter Norm vs Iteration")
    plt.legend()
    plt.grid(True)
    plt.show()

    return W_batch, W_sgd


def run_and_compare_gd(
    true_coefs=[10.0, -5.0, 0.1],
    true_intercept=20.0,
    total_size=100,
    sigma=0.5,
    seed=42,
    lr_batch=0.005,
    lr_sgd=0.005,
    epochs=200,
    batch_size=10,
):
    """
    Generate linear regression data, run Batch GD and SGD, and visualize convergence.

    Parameters
    ----------
    true_coefs : list of float
        True regression coefficients (without intercept)
    true_intercept : float
        True intercept term
    total_size : int
        Number of samples
    sigma : float
        Noise standard deviation
    seed : int
        Random seed
    lr_batch : float
        Learning rate for batch GD
    lr_sgd : float
        Learning rate for SGD
    epochs : int
        Number of epochs for both optimizers
    batch_size : int
        Mini-batch size for SGD

    Returns
    -------
    W_batch : np.ndarray
        Final weights from Batch GD
    W_sgd : np.ndarray
        Final weights from SGD
    beta_true : np.ndarray
        True coefficients with bias
    """

    # -------- Generate data --------
    X_raw, y, beta_true = generate_linear_regression_data(
        n=total_size,
        p=len(true_coefs),
        coefs=true_coefs,
        intercept=true_intercept,
        sigma=sigma,
        seed=seed,
    )

    X = add_bias_column(X_raw)

    # -------- Loss --------
    def mse_loss(W, X, y):
        return np.mean((X @ W - y) ** 2)

    # -------- Run and visualize --------
    W_batch, W_sgd = compare_and_visualize_gd(
        X,
        y,
        loss_fn=mse_loss,
        init_w=np.zeros(X.shape[1]),
        lr_batch=lr_batch,
        lr_sgd=lr_sgd,
        epochs=epochs,
        batch_size=batch_size,
    )

    return W_batch, W_sgd, beta_true
