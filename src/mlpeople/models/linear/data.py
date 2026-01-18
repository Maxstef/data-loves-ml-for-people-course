from sklearn.datasets import make_regression
import numpy as np


def generate_linear_regression_data(
    n: int = 100,
    p: int = 3,
    coefs=None,
    intercept: float = 0.0,
    sigma: float = 0.5,
    seed: int | None = None,
    min_target: float | None = None,
):
    """
    Generate synthetic data for a linear regression model.

    Data-generating process:
        y = Xβ + β₀ + ε,
        ε ~ N(0, σ²)

    Parameters
    ----------
    n : int, default=100
        Number of samples.
    p : int, default=3
        Number of features (excluding intercept).
    coefs : array-like of shape (p,), optional
        True regression coefficients β.
        If None, coefficients are sampled from N(0, 1).
    intercept : float, default=0.0
        True intercept term β₀.
    sigma : float, default=0.5
        Standard deviation of Gaussian noise.
    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n, p)
        Feature matrix without intercept.
    y : ndarray of shape (n,)
        Target vector.
    beta_true : ndarray of shape (p + 1,)
        True coefficients including intercept.
    """

    if seed is not None:
        np.random.seed(seed)

    # Generate feature matrix
    X = np.random.randn(n, p)

    # Set coefficients
    if coefs is None:
        beta = np.random.randn(p)
    else:
        beta = np.asarray(coefs)

    # Noise
    epsilon = sigma * np.random.randn(n)

    # Add intercept
    X_with_bias = np.hstack([np.ones((n, 1)), X])
    beta_true = np.insert(beta, 0, intercept)

    # Generate target
    y = X_with_bias @ beta_true + epsilon

    return X, y, beta_true


def generate_regression_data_sklearn(
    n_samples=100, n_features=3, noise=0.5, coef=None, bias=0.0, random_state=None
):
    """
    Generate synthetic linear regression data using sklearn.datasets.make_regression.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples.
    n_features : int, default=3
        Number of features.
    noise : float, default=0.5
        Standard deviation of Gaussian noise.
    coef : bool or array-like, default=None
        If True, returns the true coefficients from make_regression.
        If array-like, uses these coefficients instead of random ones.
    bias : float, default=0.0
        Intercept (bias term) for the data.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target vector.
    beta_true : ndarray of shape (n_features + 1,)
        True coefficients including intercept.
    """
    if coef is True or coef is None:
        X, y, beta_sk = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            coef=True,
            bias=bias,
            random_state=random_state,
        )
        beta_true = np.insert(beta_sk, 0, bias)
    elif isinstance(coef, (list, np.ndarray)):
        # Use provided coefficients
        X, _ = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            coef=False,
            random_state=random_state,
        )
        beta_sk = np.asarray(coef)
        beta_true = np.insert(beta_sk, 0, bias)
        X_with_bias = np.hstack([np.ones((n_samples, 1)), X])
        y = X_with_bias @ beta_true + np.random.normal(0, noise, n_samples)
    else:
        raise ValueError(
            "coef must be True, None, or an array-like of length n_features"
        )

    return X, y, beta_true


def data_unbiased_homoscedastic(
    n: int = 200,
    slope: float = 3.0,
    intercept: float = 5.0,
    sigma: float = 1.0,
    x_min: float = 0.0,
    x_max: float = 10.0,
    seed: int | None = 42,
):
    """
    Generate linear data: unbiased mean, constant variance (homoscedastic).
    y = intercept + slope * X + ε, ε ~ N(0, sigma^2)
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.uniform(x_min, x_max, size=(n, 1))
    epsilon = np.random.normal(0, sigma, size=n)
    y = intercept + slope * X[:, 0] + epsilon
    return X, y


def data_unbiased_heteroscedastic(
    n: int = 200,
    slope: float = 3.0,
    intercept: float = 5.0,
    base_sigma: float = 0.5,
    sigma_slope: float = 0.2,
    x_min: float = 0.0,
    x_max: float = 10.0,
    seed: int | None = 42,
):
    """
    Generate synthetic linear data: unbiased mean, heteroscedastic noise.

    y = intercept + slope * X + ε,
    where ε ~ N(0, sigma(X)^2), sigma(X) = base_sigma + sigma_slope * X

    Parameters
    ----------
    n : int
        Number of samples
    slope : float
        True slope of linear relationship
    intercept : float
        True intercept
    base_sigma : float
        Base standard deviation of noise
    sigma_slope : float
        How much noise increases with X
    x_min : float
        Minimum X value
    x_max : float
        Maximum X value
    seed : int or None
        Random seed

    Returns
    -------
    X : ndarray of shape (n, 1)
        Feature matrix
    y : ndarray of shape (n,)
        Target values
    """
    if seed is not None:
        np.random.seed(seed)

    # Feature
    X = np.random.uniform(x_min, x_max, size=(n, 1))

    # Heteroscedastic noise
    sigma = base_sigma + sigma_slope * X[:, 0]
    epsilon = np.random.normal(0, sigma)

    # Target
    y = intercept + slope * X[:, 0] + epsilon

    return X, y


def data_biased_homoscedastic(
    n: int = 200,
    nonlinear_func=None,
    sigma: float = 1.0,
    x_min: float = -3.0,
    x_max: float = 3.0,
    seed: int | None = 42,
):
    """
    Generate biased data: mean is nonlinear, variance constant.
    """
    if seed is not None:
        np.random.seed(seed)

    if nonlinear_func is None:
        nonlinear_func = lambda x: 2.0 * x**2

    X = np.random.uniform(x_min, x_max, size=(n, 1))
    epsilon = np.random.normal(0, sigma, size=n)
    y = nonlinear_func(X[:, 0]) + epsilon
    return X, y


def data_biased_heteroscedastic(
    n: int = 200,
    nonlinear_func=None,
    base_sigma: float = 0.2,
    sigma_slope: float = 0.3,
    x_min: float = 0.0,
    x_max: float = 5.0,
    seed: int | None = 42,
):
    """
    Generate biased data: mean is nonlinear, variance increases with X.
    """
    if seed is not None:
        np.random.seed(seed)

    if nonlinear_func is None:
        nonlinear_func = lambda x: 2.0 * x**2

    X = np.random.uniform(x_min, x_max, size=(n, 1))
    sigma = base_sigma + sigma_slope * X[:, 0]
    epsilon = np.random.normal(0, sigma, size=n)
    y = nonlinear_func(X[:, 0]) + epsilon
    return X, y
