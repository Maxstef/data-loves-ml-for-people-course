import numpy as np


def euclidean(a, b):
    """
    Compute Euclidean distance between two vectors.

    Parameters
    ----------
    a : np.ndarray of shape (n_features,)
    b : np.ndarray of shape (n_features,)

    Returns
    -------
    float
        Euclidean distance between a and b.
    """
    return np.sqrt(np.sum((a - b) ** 2))


def compute_centroid(X, cluster):
    """
    Compute centroid of a cluster.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Dataset.
    cluster : np.ndarray of int
        Indices of points in the cluster.

    Returns
    -------
    np.ndarray of shape (n_features,)
        Centroid of the cluster.
    """
    return X[cluster].mean(axis=0)


def compute_distance_matrix(X, clusters):
    """
    Compute pairwise distance matrix between clusters.

    Distance is computed between cluster centroids.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Dataset.
    clusters : list of np.ndarray
        List of clusters, each containing indices of points.

    Returns
    -------
    np.ndarray of shape (n_clusters, n_clusters)
        Symmetric distance matrix with np.inf on diagonal.
    """
    n = len(clusters)
    dist = np.full((n, n), np.inf)

    centroids = [compute_centroid(X, c) for c in clusters]

    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = euclidean(centroids[i], centroids[j])

    return dist


def merge_clusters(clusters, i, j):
    """
    Merge two clusters into one.

    Parameters
    ----------
    clusters : list of np.ndarray
        Current list of clusters.
    i : int
        Index of first cluster.
    j : int
        Index of second cluster.

    Returns
    -------
    list of np.ndarray
        Updated list of clusters after merging.
    """
    new_clusters = clusters.copy()
    merged = np.concatenate([new_clusters[i], new_clusters[j]])

    # remove old clusters
    new_clusters.pop(max(i, j))
    new_clusters.pop(min(i, j))

    # add merged cluster
    new_clusters.append(merged)

    return new_clusters


def agglomerative_clustering(X, k):
    """
    Perform agglomerative hierarchical clustering (bottom-up).

    Algorithm:
      1. Start with each point as its own cluster
      2. Iteratively merge closest clusters
      3. Stop when k clusters remain

    Distance between clusters is defined using centroid linkage.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input dataset.
    k : int
        Desired number of clusters.

    Returns
    -------
    labels : np.ndarray of shape (n_samples,)
        Cluster assignment for each point.
    clusters : list of np.ndarray
        Final clusters (lists of point indices).
    """
    clusters = [np.array([i]) for i in range(len(X))]

    while len(clusters) > k:
        dist = compute_distance_matrix(X, clusters)

        i, j = np.unravel_index(np.argmin(dist), dist.shape)

        clusters = merge_clusters(clusters, i, j)

    labels = np.zeros(len(X), dtype=int)

    for idx, cluster in enumerate(clusters):
        labels[cluster] = idx

    return labels, clusters
