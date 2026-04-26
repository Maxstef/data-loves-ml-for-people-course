import numpy as np
import matplotlib.pyplot as plt


class PCAExplorer:
    """
    PCA implemented from scratch (educational version).
    """

    def __init__(self, n_components=3):
        self.n_components = n_components

        self.mean_ = None
        self.components_ = None
        self.eigenvalues_ = None
        self.X_pca = None

    # -------------------------------------------------------
    # FIT (FROM SCRATCH PCA)
    # -------------------------------------------------------
    def fit(self, X):
        # 1. center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. covariance matrix
        cov = np.cov(X_centered, rowvar=False)

        # 3. eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 4. sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 5. store results
        self.eigenvalues_ = eigenvalues
        self.components_ = eigenvectors[:, : self.n_components]

        # 6. project
        self.X_pca = X_centered @ self.components_

        return self

    # -------------------------------------------------------
    # TRANSFORM / INVERSE
    # -------------------------------------------------------
    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_

    def inverse_transform(self, X_pca):
        return X_pca @ self.components_.T + self.mean_

    # -------------------------------------------------------
    # VARIANCE ANALYSIS
    # -------------------------------------------------------
    def explained_variance_ratio(self):
        return self.eigenvalues_ / np.sum(self.eigenvalues_)

    def plot_explained_variance(self):
        evr = self.explained_variance_ratio()

        plt.plot(evr, marker="o")
        plt.title("Explained Variance Ratio (PCA)")
        plt.xlabel("Component")
        plt.ylabel("Variance ratio")
        plt.show()

    def plot_cumulative_variance(self, threshold=0.9):
        evr = self.explained_variance_ratio()
        cum = np.cumsum(evr)

        plt.plot(cum, marker="o")
        plt.axhline(threshold, c="r")
        plt.title("Cumulative Explained Variance")
        plt.show()

        return np.argmax(cum >= threshold) + 1

    def plot_variance_bars(self):
        evr = self.explained_variance_ratio()
        cum = np.cumsum(evr)

        n = len(evr)

        plt.bar(range(1, n + 1), evr, alpha=0.5)
        plt.step(range(1, n + 1), cum, where="mid")

        plt.title("Explained + Cumulative Variance")
        plt.show()

    # -------------------------------------------------------
    # 3D VISUALIZATION
    # -------------------------------------------------------
    def plot_3d(self, y=None):
        if self.X_pca.shape[1] < 3:
            raise ValueError("Need at least 3 components")

        from mpl_toolkits.mplot3d import Axes3D  # noqa

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            self.X_pca[:, 0],
            self.X_pca[:, 1],
            self.X_pca[:, 2],
            c=y if y is not None else "blue",
            alpha=0.6,
        )

        ax.set_title("PCA 3D Projection")
        plt.show()

    def plot_2d(self, y=None):
        """
        2D PCA visualization using first two components.
        """

        if self.X_pca.shape[1] < 2:
            raise ValueError("Need at least 2 components for 2D plot")

        plt.figure(figsize=(8, 6))

        scatter = plt.scatter(
            self.X_pca[:, 0],
            self.X_pca[:, 1],
            c=y if y is not None else "blue",
            alpha=0.6,
        )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA 2D Projection")

        if y is not None:
            plt.legend(*scatter.legend_elements(), title="Class")

        plt.grid(True, alpha=0.3)
        plt.show()

    # -------------------------------------------------------
    # LOADINGS (FEATURE CONTRIBUTIONS)
    # -------------------------------------------------------
    def plot_loadings(self, feature_names, pc=0):
        loadings = self.components_[:, pc]

        plt.bar(feature_names, loadings)
        plt.xticks(rotation=90)
        plt.title(f"PC{pc+1} Loadings")
        plt.show()

    def plot_sorted_loadings(self, feature_names, pc=0):
        loadings = self.components_[:, pc]
        idx = np.argsort(np.abs(loadings))[::-1]

        plt.bar(np.array(feature_names)[idx], loadings[idx])
        plt.xticks(rotation=90)
        plt.title(f"PC{pc+1} Sorted Loadings")
        plt.show()

    # -------------------------------------------------------
    # RECONSTRUCTION ERROR
    # -------------------------------------------------------
    def reconstruction_error_curve(self, X, max_components=20):
        errors = []

        for k in range(1, max_components + 1):
            X_centered = X - self.mean_

            cov = np.cov(X_centered, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)

            idx = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, idx]

            components = eigvecs[:, :k]

            X_red = X_centered @ components
            X_rec = X_red @ components.T + self.mean_

            error = np.mean((X - X_rec) ** 2)
            errors.append(error)

        return errors

    def plot_reconstruction_error(self, X, max_components=20):
        errors = self.reconstruction_error_curve(X, max_components)

        plt.plot(errors, marker="o")
        plt.title("Reconstruction Error vs Components")
        plt.xlabel("Components")
        plt.ylabel("MSE")
        plt.show()
