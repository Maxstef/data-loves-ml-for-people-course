import numpy as np
import matplotlib.pyplot as plt

from mlpeople.models.linear import plot_1d_predictions, fit_ols


def run_linear_underfit_experiment(
    xs_range=np.arange(-10, 10, 0.5),
    w_true: float = 1.0,
    b_true: float = 0.0,
    ys_non_linearity_fn=None,
    noise_scale: float = 1.0,
    show_plots: bool = True,
    seed: int | None = 99,
):
    """
    Generate synthetic regression data with optional nonlinearity,
    fit a linear model, and visualize underfitting behavior.

    Parameters
    ----------
    xs_range : np.ndarray
        Range of x values.
    w_true : float
        True slope.
    b_true : float
        True intercept.
    ys_non_linearity_fn : callable, optional
        Function f(xs) that adds nonlinear structure.
    noise_scale : float
        Standard deviation of Gaussian noise.
    show_plots : bool
        Whether to display comparison plots.
    seed : int | None
        Random seed for reproducibility.
    Returns
    -------
    xs : np.ndarray
    ys_noisy : np.ndarray
    beta_ols : np.ndarray
    """
    rng = np.random.default_rng(seed)
    xs = xs_range.reshape(-1, 1)

    # True relationship
    ys = w_true * xs + b_true

    # Add non linearity
    if ys_non_linearity_fn is not None:
        ys = ys + ys_non_linearity_fn(xs)

    # Add noise
    noise = rng.normal(0, noise_scale, size=xs.shape).reshape(-1, 1)
    ys_noisy = ys + noise

    # Fit OLS
    beta_ols = fit_ols(xs, ys_noisy, fit_intercept=True)

    # Show plots
    if show_plots:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        plot_1d_predictions(
            xs[:, 0],
            ys_noisy,
            w_true,
            b=b_true,
            title="True Relationship (Possibly Nonlinear)",
            ax=axes[0],
        )

        plot_1d_predictions(
            xs[:, 0],
            ys_noisy,
            beta_ols[1],
            b=beta_ols[0],
            title="Linear OLS — Underfitting Example",
            ax=axes[1],
        )

        plt.tight_layout()
        plt.show()

    return xs, ys_noisy, beta_ols
