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
        same_cluster = np.where(labels == labels[i])[0]

        # --- a(i) exclude self
        if len(same_cluster) > 1:
            a = np.mean(dist[i, same_cluster[same_cluster != i]])
        else:
            sil_values[i] = 0
            continue

        # --- b(i)
        b = np.inf
        for c in unique_labels:
            if c == labels[i]:
                continue
            mask = labels == c
            b = min(b, np.mean(dist[i, mask]))

        sil_values[i] = (b - a) / max(a, b)

    return sil_values, np.mean(sil_values)
