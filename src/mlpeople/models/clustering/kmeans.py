import numpy as np


def euclidean_distance(a, b):
    """
    Compute Euclidean distance between each row in `a` and a single point `b`.

    Parameters
    ----------
    a : np.ndarray of shape (n_samples, n_features)
        Data points.
    b : np.ndarray of shape (n_features,)
        Single centroid or reference point.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Distance from each row in `a` to point `b`.
    """
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def init_centroids(X, k, random_state=None):
    """
    Initialize centroids by randomly selecting `k` samples from the dataset.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data.
    k : int
        Number of clusters.
    random_state : int or None, optional
        Seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (k, n_features)
        Initial centroid positions.
    """
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X), size=k, replace=False)
    return X[indices]


def assign_clusters(X, centroids):
    """
    Assign each sample in `X` to the nearest centroid.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data.
    centroids : np.ndarray of shape (k, n_features)
        Current centroid positions.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Cluster label for each sample.
    """
    distances = np.array([euclidean_distance(X, centroid) for centroid in centroids])
    return np.argmin(distances, axis=0)


def recompute_centroids(X, labels, k):
    """
    Recompute centroid positions as the mean of assigned points.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data.
    labels : np.ndarray of shape (n_samples,)
        Cluster assignments for each sample.
    k : int
        Number of clusters.

    Returns
    -------
    np.ndarray of shape (k, n_features)
        Updated centroid positions.
    """
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids


def compute_inertia(X, centroids, labels):
    """
    Compute the Sum of Squared Errors (SSE), also known as inertia.

    This is the objective function minimized by K-Means.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data.
    centroids : np.ndarray of shape (k, n_features)
        Current centroid positions.
    labels : np.ndarray of shape (n_samples,)
        Cluster assignments for each sample.

    Returns
    -------
    float
        Total SSE across all clusters.
    """
    sse = 0.0
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        sse += np.sum((cluster_points - centroid) ** 2)
    return sse


def kmeans_from_scratch(X, k, max_iter=100, tol=1e-4, random_state=None):
    """
    Perform K-Means clustering from scratch using Euclidean distance.

    The algorithm iteratively:
      1. Assigns points to the nearest centroid.
      2. Recomputes centroids as cluster means.
      3. Stops when centroid movement is below `tol`
         or when `max_iter` is reached.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data.
    k : int
        Number of clusters.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Convergence threshold based on centroid shift.
    random_state : int or None, optional
        Seed for reproducibility.

    Returns
    -------
    centroids : np.ndarray of shape (k, n_features)
        Final centroid positions.
    labels : np.ndarray of shape (n_samples,)
        Final cluster assignments.
    history : dict
        Dictionary containing:
            - "centroids": list of centroid positions per iteration
            - "inertia": list of SSE values per iteration
    """
    centroids = init_centroids(X, k, random_state)

    history = {"centroids": [centroids.copy()], "inertia": []}

    for _ in range(max_iter):
        labels = assign_clusters(X, centroids)
        inertia = compute_inertia(X, centroids, labels)
        history["inertia"].append(inertia)

        new_centroids = recompute_centroids(X, labels, k)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        history["centroids"].append(centroids.copy())

        if shift < tol:
            break

    return centroids, labels, history
