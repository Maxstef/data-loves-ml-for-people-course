import numpy as np
import matplotlib.pyplot as plt

from mlpeople.math import (
    batch_gradient_descent,
    stochastic_gradient_descent,
    add_bias_column,
)

from mlpeople.models.linear import generate_linear_regression_data


def visualize_scalar_gd(f, label="", init_w=10, lr=0.1, epsilon=1e-3):
    """
    Run gradient descent on a scalar function and visualize the process.

    Parameters
    ----------
    f : callable
        Scalar function to minimize.
    label : str
        Label for the function (used in plot titles).
    init_w : float
        Initial point for gradient descent.
    lr : float
        Learning rate.
    epsilon : float
        Convergence threshold for the gradient norm.
    """

    # Run gradient descent
    x_min, history = batch_gradient_descent(
        f, init_w=init_w, lr=lr, epsilon=epsilon, return_history=True
    )

    # Extract points and loss values from history
    points = np.array([h[0] for h in history])
    losses = np.array([h[1] for h in history])

    # Define x values for plotting the function (dynamic range)
    x_min_plot = min(points.min(), x_min) - 1
    x_max_plot = max(points.max(), x_min) + 1
    x_vals = np.linspace(x_min_plot, x_max_plot, 500)
    y_vals = np.array([f(x) for x in x_vals])

    # ---------------- Plot function and GD trajectory ----------------
    plt.figure(figsize=(10, 4))

    # Function + GD path
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, y_vals, label="f(x)")
    plt.scatter(points, losses, color="red", label="GD steps")
    plt.plot(points, losses, color="red", linestyle="--", alpha=0.5)

    # Annotate final point (shift text slightly up)
    plt.scatter(x_min, f(x_min), color="green", s=80, marker="*", label="Converged min")
    plt.text(
        x_min,
        f(x_min) + (y_vals.max() - y_vals.min()) * 0.1,
        f"Min: {x_min:.4f}\nSteps: {len(history)}\nLR: {lr}",
        color="green",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"{label}: Function & GD trajectory")
    plt.legend()
    plt.grid(True)

    # ---------------- Plot loss vs iteration ----------------
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.title(f"{label}: Loss vs iteration")
    plt.grid(True)

    # Annotate final loss (shift text slightly up)
    plt.scatter(len(losses), losses[-1], color="green", s=80, marker="*")
    plt.text(
        len(losses),
        losses[-1] + (losses.max() - losses.min()) * 0.1,
        f"Final loss: {losses[-1]:.4f}\nLR: {lr}",
        color="green",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    plt.tight_layout()
    plt.show()


def visualize_linear_regression_gd(X, y, beta_true, history, plot_predictions=False):
    """
    Visualize gradient descent convergence for linear regression.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix including bias column (n_samples, n_features)
    y : np.ndarray
        True outputs (n_samples,)
    beta_true : array-like
        True coefficients (including intercept if in X)
    history : list
        Gradient descent history: [(w, loss, grad_norm), ...]
    plot_predictions : bool, optional
        If True, create a scatter plot of predicted y vs true y. Default is False.
    """

    history_arr = np.array([h[0] for h in history])  # shape: (steps, n_features)
    losses = np.array([h[1] for h in history])

    n_features = X.shape[1]

    # ---------------- Plot weight convergence ----------------
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for i in range(n_features):
        plt.plot(history_arr[:, i], label=f"w{i} (true={beta_true[i]:.2f})")
        plt.hlines(
            beta_true[i], 0, len(history), colors="gray", linestyles="dashed", alpha=0.5
        )
    plt.xlabel("Iteration")
    plt.ylabel("Weight value")
    plt.title("Gradient Descent: Parameter convergence")
    plt.legend()
    plt.grid(True)

    # ---------------- Plot loss vs iteration ----------------
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Gradient Descent: Loss vs iteration")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ---------------- Optional: Predictions vs True ----------------
    if plot_predictions:
        W_opt = history_arr[-1]
        y_pred = X @ W_opt
        plt.figure(figsize=(5, 5))
        plt.scatter(y, y_pred, color="blue", alpha=0.6)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
        plt.xlabel("True y")
        plt.ylabel("Predicted y")
        plt.title("Predictions vs True y")
        plt.grid(True)
        plt.show()


def plot_sgd_batch_vs_epoch(
    batch_losses,
    epoch_losses,
    title="SGD: Mini-batch Loss vs Epoch Loss",
    smooth=False,
    window_threshold=None,
):
    """
    Visualize SGD loss per mini-batch and per epoch.
    """

    plt.figure()
    plt.plot(batch_losses, alpha=0.4, label="Mini-batch loss")

    # -------- Smoothed batch loss --------
    if smooth:
        if window_threshold is not None:
            window = window_threshold
        else:
            window = max(1, len(batch_losses) // 100)  # fallback
        if window >= 1:
            smooth_values = np.convolve(
                batch_losses, np.ones(window) / window, mode="valid"
            )
            plt.plot(
                range(window - 1, window - 1 + len(smooth_values)),
                smooth_values,
                linewidth=2,
                label=f"Smoothed (window={window})",
            )

    # Epoch loss markers
    epoch_x = np.linspace(0, len(batch_losses), len(epoch_losses))
    plt.plot(epoch_x, epoch_losses, "o-", label="Epoch loss")

    plt.xlabel("Mini-batch update")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


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
