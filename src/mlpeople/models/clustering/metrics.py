import numpy as np


def euclidean_distance_matrix(X):
    """Compute full pairwise distance matrix."""
    n = len(X)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X[i] - X[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def silhouette_from_scratch(X, labels):
    """
    Compute silhouette value for each sample manually.

    Returns
    -------
    sil_values : np.ndarray
        Silhouette value per sample.
    sil_avg : float
        Average silhouette score.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    dist = euclidean_distance_matrix(X)

    sil_values = np.zeros(len(X))
    unique_labels = np.unique(labels)

    for i in range(len(X)):
        same_cluster = labels == labels[i]
        other_clusters = unique_labels[unique_labels != labels[i]]

        # a(i): mean intra-cluster distance
        a = np.mean(dist[i][same_cluster]) if np.sum(same_cluster) > 1 else 0

        # b(i): smallest mean distance to other clusters
        b = np.inf
        for c in other_clusters:
            mask = labels == c
            b = min(b, np.mean(dist[i][mask]))

        sil_values[i] = (b - a) / max(a, b)

    return sil_values, np.mean(sil_values)
