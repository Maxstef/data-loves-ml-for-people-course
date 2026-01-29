import numpy as np
from sklearn.datasets import make_classification


def generate_logistic_regression_data(
    n=100,
    p=1,
    coefs=None,
    intercept=0.0,
    sigma=0.0,
    seed=None,
):
    """
    Generate synthetic logistic regression data with controlled parameters.

    Supports 1D, 2D, or higher-dimensional feature spaces by setting p.

    Parameters
    ----------
    n : int
        Number of samples.
    p : int
        Number of features.
    coefs : array-like of shape (p,), optional
        True feature coefficients. If None, sampled randomly.
    intercept : float
        True bias term.
    sigma : float
        Standard deviation of Gaussian noise added to logits.
    seed : int | None
        Random seed.

    Returns
    -------
    X : np.ndarray of shape (n, p)
        Feature matrix.
    y : np.ndarray of shape (n,)
        Binary labels {0,1}.
    true_theta : np.ndarray of shape (p + 1,)
        Ground-truth parameters (bias + coefficients).
    """

    rng = np.random.default_rng(seed)

    if coefs is None:
        coefs = rng.normal(size=p)

    X = rng.normal(0, 1, size=(n, p))

    logits = intercept + X @ np.array(coefs)
    logits += rng.normal(0, sigma, size=n)

    probs = 1 / (1 + np.exp(-logits))
    y = (probs >= 0.5).astype(int)

    return X, y, np.r_[intercept, coefs]


def generate_binary_classification_sklearn(
    n_samples: int = 500,
    n_features: int = 2,
    n_informative: int = 2,
    n_redundant: int = 0,
    class_sep: float = 1.0,
    flip_y: float = 0.0,
    random_state: int | None = None,
):
    """
    Generate a binary classification dataset using sklearn.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix
    y : np.ndarray of shape (n_samples,)
        Binary labels {0, 1}
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=1,
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_state,
    )

    return X, y
