import numpy as np


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def init_centroids(X, k, random_state=None):
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X), size=k, replace=False)
    return X[indices]


def assign_clusters(X, centroids):
    distances = np.array([euclidean_distance(X, centroid) for centroid in centroids])
    return np.argmin(distances, axis=0)


def recompute_centroids(X, labels, k):
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids


def compute_inertia(X, centroids, labels):
    sse = 0.0
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        sse += np.sum((cluster_points - centroid) ** 2)
    return sse


def kmeans_from_scratch(X, k, max_iter=100, tol=1e-4, random_state=None):
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
