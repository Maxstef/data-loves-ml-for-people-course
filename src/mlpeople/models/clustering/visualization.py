import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score


def plot_kmeans_result(X, centroids, labels, history, title="K-Means from scratch"):
    """
    Visualize K-Means results in three subplots:
      1. Final cluster assignment with centroids.
      2. Inertia (SSE) over iterations.
      3. Centroid movement across iterations.

    This function is intended for 2D feature spaces for educational
    visualization of how K-Means converges.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
        Input data used for clustering (must be 2D for plotting).
    centroids : np.ndarray of shape (k, 2)
        Final centroid positions.
    labels : np.ndarray of shape (n_samples,)
        Cluster labels assigned to each sample.
    history : dict
        History dictionary returned by `kmeans_from_scratch`, containing:
            - "centroids": list of centroid arrays per iteration
            - "inertia": list of SSE values per iteration
    title : str, optional
        Figure title.

    Returns
    -------
    None
        Displays the matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 1. Final clusters ---
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=30)
    ax.scatter(centroids[:, 0], centroids[:, 1], c="red", s=200, marker="X")
    ax.set_title("Final clusters")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    # --- 2. Inertia curve ---
    ax = axes[1]
    ax.plot(history["inertia"])
    ax.set_title("Inertia (SSE) over iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("SSE")

    # --- 3. Centroid movement ---
    ax = axes[2]
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=20, alpha=0.3)

    for i in range(len(history["centroids"]) - 1):
        c1 = history["centroids"][i]
        c2 = history["centroids"][i + 1]

        for j in range(len(c1)):
            ax.plot([c1[j, 0], c2[j, 0]], [c1[j, 1], c2[j, 1]], "k--", alpha=0.4)

    ax.scatter(centroids[:, 0], centroids[:, 1], c="red", s=200, marker="X")
    ax.set_title("Centroid movement")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_kmeans_centroid_movement(
    X, centroids, labels, history, title="Centroid movement"
):
    """
    Plot only the centroid movement across iterations on top of clustered data.

    Useful when you want to focus specifically on how centroids traveled
    during convergence.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
        Input data used for clustering (must be 2D for plotting).
    centroids : np.ndarray of shape (k, 2)
        Final centroid positions.
    labels : np.ndarray of shape (n_samples,)
        Cluster labels assigned to each sample.
    history : dict
        History dictionary returned by `kmeans_from_scratch`, containing:
            - "centroids": list of centroid arrays per iteration
    title : str, optional
        Plot title.

    Returns
    -------
    None
        Displays the matplotlib figure.
    """
    for i in range(len(history["centroids"]) - 1):
        c1 = history["centroids"][i]
        c2 = history["centroids"][i + 1]
        plt.plot([c1[:, 0], c2[:, 0]], [c1[:, 1], c2[:, 1]], "k--", alpha=0.3)

    plt.scatter(X[:, 0], X[:, 1], c=labels, s=20)
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", s=200, marker="X")
    plt.title(title)
    plt.show()


def plot_silhouette(X, labels, ax, title="Silhouette plot"):
    """
    Draw silhouette plot on a provided matplotlib axis.

    This function does NOT create or show a figure.
    It is intended to be used inside composite plots.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    sil_vals = silhouette_samples(X, labels)
    silhouette_avg = silhouette_score(X, labels)

    y_lower = 10
    unique_labels = np.unique(labels)

    for i, cluster in enumerate(unique_labels):
        cluster_sil_vals = sil_vals[labels == cluster]
        cluster_sil_vals.sort()

        size = cluster_sil_vals.shape[0]
        y_upper = y_lower + size

        color = cm.nipy_spectral(float(i) / len(unique_labels))
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_sil_vals,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax.text(-0.05, y_lower + 0.5 * size, str(cluster))
        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title(f"{title}\nAvg = {silhouette_avg:.3f}")


def plot_show_silhouette(X, labels, title="Silhouette plot"):
    """
    Create a new figure and display a standalone silhouette plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_silhouette(X, labels, ax=ax, title=title)
    plt.tight_layout()
    plt.show()


def plot_clusters_and_silhouette(X, labels):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # clusters
    axes[0].scatter(X[:, 0], X[:, 1], c=labels, s=30)
    axes[0].set_title("Clusters")

    # silhouette
    plot_silhouette(X, labels, ax=axes[1], title="Silhouette")

    plt.tight_layout()
    plt.show()
