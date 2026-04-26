import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def pca_from_scratch(X, n_components):
    """
    PCA via covariance eigen-decomposition.
    Returns projected data, components, and eigenvalues.
    """
    # Center
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Covariance
    cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)

    # Eigen
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select components
    components = eigenvectors[:, :n_components]

    # Project
    X_reduced = X_centered @ components

    return X_reduced, components, eigenvalues


def plot_explained_variance(eigenvalues, title="Explained Variance Ratio"):
    """
    Plots proportion of variance explained by each component.
    """
    explained = eigenvalues / np.sum(eigenvalues)
    n = len(explained)

    plt.plot(range(1, n + 1), explained, marker="o")
    plt.title(title)
    plt.xlabel("Component")
    plt.ylabel("Variance Ratio")
    plt.show()


def plot_cumulative_variance(
    eigenvalues, threshold=0.9, title="Cumulative Explained Variance"
):
    """
    Plots cumulative variance and returns components needed for threshold.
    """
    explained = eigenvalues / np.sum(eigenvalues)
    cumulative = np.cumsum(explained)
    n = len(cumulative)

    plt.plot(range(1, n + 1), cumulative, marker="o")
    plt.axhline(threshold, c="r")
    plt.title(title)
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative Variance")
    plt.show()

    k = np.argmax(cumulative >= threshold) + 1
    print(f"Components needed for {int(threshold*100)}% variance: {k}")
    return k


def plot_variance_bars(eigenvalues, title="Explained vs Cumulative Variance"):
    """
    Bar + line plot of explained and cumulative variance.
    """
    explained = eigenvalues / np.sum(eigenvalues)
    cumulative = np.cumsum(explained)
    n = len(explained)

    plt.bar(
        range(1, n + 1),
        explained,
        alpha=0.5,
        align="center",
        label="Individual explained variance",
    )

    plt.step(
        range(1, n + 1), cumulative, where="mid", label="Cumulative explained variance"
    )

    plt.xlabel("Principal Component")
    plt.ylabel("Explained variance ratio")
    plt.title(title)
    plt.legend(loc="best")
    plt.show()


def plot_3d_original(X, y, title="Original Wine Data (first 3 features)"):
    """
    3D scatter plot of original feature space.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    ax.set_title(title)

    plt.show()


def plot_3d_pca(X_reduced, y, title="PCA 3D Projection"):
    """
    3D scatter plot of PCA-transformed data.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)

    plt.show()


def plot_pca_axes(X, components, mean, title="PCA axes in original space"):
    """
    Visualizes PCA directions as vectors in original feature space.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.2)

    labels = ["PC1", "PC2", "PC3"]

    for i in range(3):
        vec = components[:, i]
        end = mean + 5 * vec
        ax.plot(
            [mean[0], mean[0] + 5 * vec[0]],
            [mean[1], mean[1] + 5 * vec[1]],
            [mean[2], mean[2] + 5 * vec[2]],
            linewidth=3,
        )
        ax.text(end[0], end[1], end[2], labels[i], fontsize=10)

    ax.set_title(title)
    plt.show()


def reconstruction_error_curve(X, max_components=20):
    """
    Computes reconstruction error for different PCA dimensions.
    """
    errors = []

    for k in range(1, max_components + 1):
        pca = PCA(n_components=k)
        X_red = pca.fit_transform(X)
        X_rec = pca.inverse_transform(X_red)

        error = np.mean((X - X_rec) ** 2)
        errors.append(error)

    return errors


def plot_reconstruction_error(X, max_components=20, title="Reconstruction Error"):
    """
    Plots reconstruction error vs number of components.
    """
    errors = reconstruction_error_curve(X, max_components)

    plt.plot(errors, marker="o")
    plt.title(title)
    plt.xlabel("Components")
    plt.ylabel("MSE error")
    plt.show()


def plot_pca_loadings(components, feature_names, pc=0):
    """
    Shows feature contributions for a principal component.
    """
    plt.bar(feature_names, components[pc])
    plt.xticks(rotation=90)
    plt.title(f"PC{pc+1} Loadings")
    plt.show()


def plot_sorted_loadings(components, feature_names, pc=0):
    """
    Sorted feature importance for a principal component.
    """
    idx = np.argsort(np.abs(components[pc]))[::-1]

    plt.bar(np.array(feature_names)[idx], components[pc][idx])
    plt.xticks(rotation=90)
    plt.title(f"PC{pc+1} Sorted Loadings")
    plt.show()
