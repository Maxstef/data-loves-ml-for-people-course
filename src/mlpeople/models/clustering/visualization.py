import matplotlib.pyplot as plt
import numpy as np


def plot_kmeans_result(X, centroids, labels, history, title="K-Means from scratch"):
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
    for i in range(len(history["centroids"]) - 1):
        c1 = history["centroids"][i]
        c2 = history["centroids"][i + 1]
        plt.plot([c1[:, 0], c2[:, 0]], [c1[:, 1], c2[:, 1]], "k--", alpha=0.3)

    plt.scatter(X[:, 0], X[:, 1], c=labels, s=20)
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", s=200, marker="X")
    plt.title(title)
    plt.show()
