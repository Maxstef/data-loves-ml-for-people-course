import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_sigmoid_fit_1d(
    X,
    y,
    theta,
    true_theta=None,
    show_points=True,
    show_probability_curve=True,
    show_threshold=True,
    show_boundary=True,
    threshold=0.5,
    figsize=(10, 6),
    resolution=300,
    ax=None,
    title="Logistic Regression (1D)",
):
    """
    Visualize 1D logistic regression fit.

    Parameters
    ----------
    X : np.ndarray shape (m, 1) or (m,)
        Single feature input.
    y : np.ndarray shape (m,)
        Binary labels {0,1}.
    theta : np.ndarray shape (2,)
        Learned parameters [bias, coef].
    true_theta : np.ndarray shape (2,) | None
        Ground-truth parameters (for synthetic data).
    show_points : bool
        Plot data points.
    show_probability_curve : bool
        Plot sigmoid probability curve.
    show_threshold : bool
        Show horizontal threshold line.
    show_boundary : bool
        Show decision boundary.
    threshold : float
        Classification threshold.
    resolution : int
        Smoothness of sigmoid curve.
    """

    X = np.asarray(X).reshape(-1, 1)
    y = np.asarray(y)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # ---- Curve grid ----
    x_min, x_max = X.min() - 1, X.max() + 1
    x_curve = np.linspace(x_min, x_max, resolution)

    X_curve_bias = np.c_[np.ones(len(x_curve)), x_curve]

    # ---- Learned curve ----
    logits = X_curve_bias @ theta
    probs = 1 / (1 + np.exp(-logits))

    if show_probability_curve:
        ax.plot(
            x_curve,
            probs,
            label="Learned sigmoid",
            linewidth=2,
        )

    # ---- True curve (if provided) ----
    if true_theta is not None:
        true_probs = 1 / (1 + np.exp(-(X_curve_bias @ true_theta)))
        ax.plot(
            x_curve,
            true_probs,
            linestyle="--",
            linewidth=2,
            label="True sigmoid",
        )

    # ---- Data points ----
    if show_points:
        X0 = X[y == 0]
        X1 = X[y == 1]

        ax.scatter(X0, np.zeros_like(X0), color="red", edgecolor="k", label="Class 0")
        ax.scatter(X1, np.ones_like(X1), color="blue", edgecolor="k", label="Class 1")

    # ---- Threshold ----
    if show_threshold:
        ax.axhline(
            threshold,
            linestyle="--",
            color="grey",
            label=f"Threshold = {threshold}",
        )

    # ---- Learned boundary ----
    if show_boundary and theta[1] != 0:
        boundary = (np.log(threshold / (1 - threshold)) - theta[0]) / theta[1]
        ax.axvline(
            boundary,
            linestyle=":",
            linewidth=2,
            label=f"Learned boundary = {boundary:.2f}",
        )

    # ---- True boundary ----
    if show_boundary and true_theta is not None and true_theta[1] != 0:
        true_boundary = (
            np.log(threshold / (1 - threshold)) - true_theta[0]
        ) / true_theta[1]

        ax.axvline(
            true_boundary,
            linestyle="-.",
            linewidth=2,
            color="orange",
            label=f"True boundary = {true_boundary:.2f}",
        )

    # ---- Styling ----
    ax.set_xlabel("X")
    ax.set_ylabel("Probability / Class")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    plt.show()


def plot_decision_boundary_2d(
    X, y, theta, grid_step=0.01, ax=None, title="Decision Boundary"
):
    """
    Plot 2D logistic regression decision boundary.

    Parameters
    ----------
    X : np.ndarray, shape (m, 2)
        Feature matrix (without bias column).
    y : np.ndarray, shape (m,)
        Labels {0,1}.
    theta : np.ndarray, shape (3,)
        Logistic regression parameters (bias + 2 features).
    grid_step : float
        Resolution of the grid for plotting.
    ax : matplotlib.axes.Axes | None
        Optional axes to plot on.
    title : str
        Plot title.
    """
    if X.shape[1] != 2:
        raise ValueError("plot_decision_boundary_2d only works for 2 features.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step), np.arange(y_min, y_max, grid_step)
    )

    # Add bias term
    grid = np.c_[np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()]
    logits = grid @ theta
    probs = 1 / (1 + np.exp(-logits))
    Z = (probs >= 0.5).reshape(xx.shape)

    cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#0000FF"])

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title(title)
    plt.show()


def plot_log_loss_curve(history, ax=None, title="Log-Loss Curve"):
    """
    Plot log-loss over training epochs.

    Parameters
    ----------
    history : list of tuples
        Each tuple = (theta copy, loss, grad_norm)
    ax : matplotlib.axes.Axes | None
        Optional axes to plot on.
    title : str
        Plot title.
    """
    losses = [h[1] for h in history]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(losses, marker="o", linestyle="-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log-Loss")
    ax.set_title(title)
    ax.grid(True)
    plt.show()


def plot_predicted_probabilities(X, y, theta, ax=None, title="Predicted Probabilities"):
    """
    Scatter plot of predicted probabilities vs true labels.

    Parameters
    ----------
    X : np.ndarray, shape (m, n)
        Feature matrix (can be 1D or 2D, without bias).
    y : np.ndarray, shape (m,)
        True labels.
    theta : np.ndarray
        Learned logistic regression parameters.
    ax : matplotlib.axes.Axes | None
        Optional axes to plot on.
    title : str
        Plot title.
    """
    # Add bias
    X_bias = np.c_[np.ones(X.shape[0]), X]
    probs = 1 / (1 + np.exp(-(X_bias @ theta)))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(np.arange(len(probs)), probs, c=y, cmap="bwr", edgecolor="k", alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Predicted Probability")
    ax.set_title(title)
    plt.show()
