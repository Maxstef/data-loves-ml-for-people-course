import numpy as np


def manhattan_distance(a, b):
    """
    Compute Manhattan (L1) distance between a set of points and a single point.

    Parameters
    ----------
    a : np.ndarray of shape (n_samples, n_features)
        Dataset points.
    b : np.ndarray of shape (n_features,)
        Single reference point (centroid).

    Returns
    -------
    np.ndarray of shape (n_samples,)
        L1 distances from each point in `a` to `b`.
    """
    return np.sum(np.abs(a - b), axis=1)


def init_centroids(X, k, random_state=None):
    """
    Initialize k centroids by randomly selecting points from the dataset.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input dataset.
    k : int
        Number of clusters.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (k, n_features)
        Initial centroids.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=k, replace=False)
    return X[idx]


def assign_clusters_median(X, centroids):
    """
    Assign each point to the nearest centroid using Manhattan distance.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Dataset.
    centroids : np.ndarray of shape (k, n_features)
        Current cluster centers.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Cluster labels for each point.
    """
    distances = np.array([manhattan_distance(X, c) for c in centroids])
    return np.argmin(distances, axis=0)


def recompute_medians(X, labels, k):
    """
    Recompute cluster centers as coordinate-wise medians.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Dataset.
    labels : np.ndarray of shape (n_samples,)
        Cluster assignments.
    k : int
        Number of clusters.

    Returns
    -------
    np.ndarray of shape (k, n_features)
        Updated centroids (medians).
    """
    return np.array([np.median(X[labels == i], axis=0) for i in range(k)])


def kmedian(X, k, max_iter=100, tol=1e-4, random_state=None):
    """
    K-Median clustering algorithm.

    This is a robust alternative to K-Means that uses:
    - Manhattan (L1) distance
    - Median as cluster center

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input dataset.
    k : int
        Number of clusters.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-4
        Convergence tolerance for centroid movement.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    centroids : np.ndarray of shape (k, n_features)
        Final cluster centers (medians).
    labels : np.ndarray of shape (n_samples,)
        Cluster assignments.
    """
    centroids = init_centroids(X, k, random_state)

    for _ in range(max_iter):
        labels = assign_clusters_median(X, centroids)
        new_centroids = recompute_medians(X, labels, k)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if shift < tol:
            break

    return centroids, labels
