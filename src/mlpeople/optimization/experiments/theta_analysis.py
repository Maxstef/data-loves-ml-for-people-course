import numpy as np
import matplotlib.pyplot as plt

from mlpeople.visualization.optimization import plot_loss_and_grad_vs_theta1
from mlpeople.models.logistic import generate_logistic_regression_data
from mlpeople.models.linear import generate_linear_regression_data
from mlpeople.optimization.analysis.loss_landscape import (
    linear_predict,
    mse_loss_linear_yhat,
    grad_mse_linear,
    logistic_predict_proba,
    log_loss_yhat,
    grad_log_loss,
)


def run_linear_regression_theta1_analysis(
    theta0=0.0,
    theta1=1.0,
    plot_data=True,
    n=100,
    sigma=1,
    seed=42,
    theta0_fixed=0.0,
    theta1_values=np.linspace(-7, 7, 200),
    predict_fn=linear_predict,
    loss_fn=mse_loss_linear_yhat,
    grad_fn=grad_mse_linear,
    subplots=False,
):
    """
    Generate 1D linear regression data and analyze loss & gradient
    behavior as a function of theta1 while fixing theta0.
    """

    # Generate data
    X_linear, y_linear, beta_linear_true = generate_linear_regression_data(
        n=n, p=1, coefs=[theta1], intercept=theta0, sigma=sigma, seed=seed
    )

    if plot_data:
        plt.scatter(X_linear, y_linear)
        plt.title(f"Generated Data (sigma={sigma})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    # Calculate and plot loss and gradient for fixed theta0
    return plot_loss_and_grad_vs_theta1(
        X_linear,
        y_linear,
        theta1_true=theta1,
        theta1_values=theta1_values,
        theta0=theta0_fixed,
        predict_fn=predict_fn,
        loss_fn=loss_fn,
        grad_fn=grad_fn,
        subplots=subplots,
    )


def run_logistic_regression_theta1_analysis(
    theta0=0.0,
    theta1=1.0,
    plot_data=True,
    n=100,
    sigma=0.5,
    seed=42,
    theta0_fixed=0.0,
    theta1_values=np.linspace(-7, 7, 200),
    predict_fn=logistic_predict_proba,
    loss_fn=log_loss_yhat,
    grad_fn=grad_log_loss,
    subplots=False,
):
    """
    Generate 1D logistic regression data and analyze loss & gradient
    behavior as a function of theta1 while fixing theta0.
    """

    X_logistic, y_logistic, _ = generate_logistic_regression_data(
        n=n,
        p=1,
        coefs=[theta1],
        intercept=theta0,
        sigma=sigma,
        seed=seed,
    )

    if plot_data:
        plt.scatter(X_logistic[:, 0], y_logistic, alpha=0.7)
        plt.title(f"Logistic Data (sigma={sigma})")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.show()

    return plot_loss_and_grad_vs_theta1(
        X_logistic,
        y_logistic,
        theta1_true=theta1,
        theta1_values=theta1_values,
        theta0=theta0_fixed,
        predict_fn=predict_fn,
        loss_fn=loss_fn,
        grad_fn=grad_fn,
        subplots=subplots,
    )
