import numpy as np
import matplotlib.pyplot as plt

from mlpeople.math import add_bias_column
from mlpeople.optimization.analysis import (
    mse_loss_logistic,
    log_loss,
    grad_norm_mse_logistic,
    grad_norm_log,
    linear_predict,
    mse_loss_linear_yhat,
    grad_mse_linear,
)


def plot_loss_and_grad_vs_theta1(
    X,
    y,
    fit_intercept=True,
    theta0=0.0,
    theta1_values=np.linspace(-7, 7, 200),
    theta1_true=None,
    predict_fn=linear_predict,
    # Below default values are for linear regression loss and gradient
    loss_fn=mse_loss_linear_yhat,
    grad_fn=grad_mse_linear,
    subplots=False,
):
    """
    Plot loss and gradient norm vs theta1 while fixing theta0.

    Notes
    -----
    - Assumes a single feature (+ optional bias)
    - X is expected WITHOUT bias column with default fit_intercept=True
    """

    m = X.shape[0]
    X_bias = add_bias_column(X) if fit_intercept else X

    loss_values = []
    grad_norms = []

    for t1 in theta1_values:
        theta = np.array([theta0, t1])
        y_pred = predict_fn(theta, X_bias)
        loss = loss_fn(y, y_pred)
        grad = grad_fn(theta, X_bias, y)

        loss_values.append(loss)
        grad_norms.append(np.linalg.norm(grad))

    loss_values = np.array(loss_values)
    grad_norms = np.array(grad_norms)

    # Find minima
    idx_loss_min = np.argmin(loss_values)
    idx_grad_min = np.argmin(grad_norms)

    theta1_loss_min = theta1_values[idx_loss_min]
    theta1_grad_min = theta1_values[idx_grad_min]

    loss_min = loss_values[idx_loss_min]
    grad_min = grad_norms[idx_grad_min]

    # ---- Plotting helpers ----
    def draw_common_lines(ax):
        if theta1_true is not None:
            ax.axvline(
                theta1_true,
                color="green",
                linestyle="--",
                label=f"θ₁ true = {theta1_true:.2f}",
            )

        ax.axvline(
            theta1_loss_min,
            color="red",
            linestyle=":",
            label=f"θ₁ min loss = {theta1_loss_min:.2f}",
        )

        ax.axvline(
            theta1_grad_min,
            color="purple",
            linestyle=":",
            label=f"θ₁ min grad = {theta1_grad_min:.2f}",
        )

    # ---- Plotting ----
    if subplots:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # Loss subplot
        axes[0].plot(theta1_values, loss_values, label="Loss")
        axes[0].scatter(theta1_loss_min, loss_min, color="red", zorder=3)
        draw_common_lines(axes[0])
        axes[0].set_xlabel("Theta1")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss vs Theta1")
        axes[0].grid(True)
        axes[0].legend()

        # Gradient norm subplot
        axes[1].plot(theta1_values, grad_norms, color="orange", label="Gradient Norm")
        axes[1].scatter(theta1_grad_min, grad_min, color="purple", zorder=3)
        draw_common_lines(axes[1])
        axes[1].set_xlabel("Theta1")
        axes[1].set_ylabel("||Gradient||")
        axes[1].set_title("Gradient Norm vs Theta1")
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    else:
        # Loss plot
        plt.figure(figsize=(6, 4))
        plt.plot(theta1_values, loss_values, label="Loss")
        plt.scatter(theta1_loss_min, loss_min, color="red", zorder=3)
        draw_common_lines(plt.gca())
        plt.xlabel("Theta1")
        plt.ylabel("Loss")
        plt.title("Loss vs Theta1")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Gradient norm plot
        plt.figure(figsize=(6, 4))
        plt.plot(theta1_values, grad_norms, color="orange", label="Gradient Norm")
        plt.scatter(theta1_grad_min, grad_min, color="purple", zorder=3)
        draw_common_lines(plt.gca())
        plt.xlabel("Theta1")
        plt.ylabel("||Gradient||")
        plt.title("Gradient Norm vs Theta1")
        plt.grid(True)
        plt.legend()
        plt.show()

    return {
        "theta1_loss_min": theta1_loss_min,
        "theta1_grad_min": theta1_grad_min,
        "loss_min": loss_min,
        "grad_min": grad_min,
        "loss_values": loss_values,
        "grad_norms": grad_norms,
    }


def compare_losses_vs_theta1(
    X,
    y,
    loss1_fn=mse_loss_logistic,
    loss2_fn=log_loss,
    grad1_fn=grad_norm_mse_logistic,
    grad2_fn=grad_norm_log,
    label1="MSE",
    label2="Log-Loss",
    title="Logistic Regression",
    theta0_fixed=0.0,
    theta1_values=np.linspace(-8, 8, 200),
    theta1_true=None,
):
    """
    Compare two loss functions and their gradient norms as a function
    of theta1 while fixing theta0.
    """

    X_bias = add_bias_column(X)
    theta0 = theta0_fixed

    loss1_values = []
    loss2_values = []

    # ---- Compute losses ----
    for t1 in theta1_values:
        theta = np.array([theta0, t1])
        loss1_values.append(loss1_fn(theta, X_bias, y))
        loss2_values.append(loss2_fn(theta, X_bias, y))

    loss1_values = np.array(loss1_values)
    loss2_values = np.array(loss2_values)

    # ---- Minima ----
    idx1 = np.argmin(loss1_values)
    idx2 = np.argmin(loss2_values)

    theta1_min1 = theta1_values[idx1]
    theta1_min2 = theta1_values[idx2]

    # ---- Plot losses ----
    plt.figure(figsize=(8, 5))

    plt.plot(theta1_values, loss1_values, label=f"{label1} Loss")
    plt.plot(theta1_values, loss2_values, label=f"{label2} Loss")

    plt.axvline(
        theta1_min1, linestyle=":", label=f"{label1} θ₁ min ({round(theta1_min1, 2)})"
    )
    plt.axvline(
        theta1_min2,
        linestyle="--",
        color="orange",
        label=f"{label2} θ₁ min ({round(theta1_min2, 2)})",
    )

    if theta1_true is not None:
        plt.axvline(
            theta1_true,
            linestyle="-.",
            color="r",
            label=f"True θ₁ ({round(theta1_true, 2)})",
        )

    plt.xlabel("Theta1")
    plt.ylabel("Loss")
    plt.title(f"{label1} vs {label2} ({title})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---- Gradient norms ----
    grad1 = [grad1_fn(np.array([theta0, t1]), X_bias, y) for t1 in theta1_values]
    grad2 = [grad2_fn(np.array([theta0, t1]), X_bias, y) for t1 in theta1_values]

    grad1 = np.array(grad1)
    grad2 = np.array(grad2)

    # Min gradient norms
    gidx1 = np.argmin(grad1)
    gidx2 = np.argmin(grad2)

    # ---- Plot gradients ----
    plt.figure(figsize=(8, 5))

    plt.plot(theta1_values, grad1, label=f"{label1} Grad Norm")
    plt.plot(theta1_values, grad2, label=f"{label2} Grad Norm")

    plt.axvline(
        theta1_values[gidx1],
        linestyle=":",
        label=f"{label1} grad min ({round(theta1_values[gidx1], 2)})",
    )
    plt.axvline(
        theta1_values[gidx2],
        linestyle="--",
        color="orange",
        label=f"{label2} grad min ({round(theta1_values[gidx2], 2)})",
    )

    if theta1_true is not None:
        plt.axvline(
            theta1_true,
            linestyle="-.",
            color="r",
            label=f"True θ₁ ({round(theta1_true, 2)})",
        )

    plt.xlabel("Theta1")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm vs Theta1")
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        "theta1_min_loss1": theta1_min1,
        "theta1_min_loss2": theta1_min2,
        "loss1_min": loss1_values[idx1],
        "loss2_min": loss2_values[idx2],
    }
