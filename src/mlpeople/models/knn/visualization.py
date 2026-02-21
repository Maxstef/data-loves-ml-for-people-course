import numpy as np
import matplotlib.pyplot as plt
from .knn import KNN


def visualize_knn_classifier(
    X_train,
    X_test,
    y_train,
    y_test,
    k=5,
    p=2,
    figsize=(8, 6),
    show_neighbors=False,
    test_point=None,
):
    """
    Train a KNN classifier and visualize its decision boundary.

    Parameters
    ----------
    X_train : ndarray of shape (n_train, 2)
        Training features (2D).

    X_test : ndarray of shape (n_test, 2)
        Test features (2D).

    y_train : ndarray of shape (n_train,)
        Training labels (0 or 1).

    y_test : ndarray of shape (n_test,)
        Test labels (0 or 1).

    k : int, default=5
        Number of nearest neighbors to use.

    p : int or float, default=2
        Minkowski distance parameter.
        - p=2 → Euclidean
        - p=1 → Manhattan
        - p>2 → higher-order Minkowski

    figsize : tuple, default=(8, 6)
        Figure size.

    show_neighbors : bool, default=False
        If True, highlights the k nearest neighbors of `test_point`.

    test_point : array-like of shape (2,), optional
        Coordinates of the test point to visualize neighbors.
        Required if `show_neighbors=True`.

    Notes
    -----
    - Training points are plotted as circles ('o'), test points as crosses ('x').
    - Contour plot shows KNN decision boundary.
    """
    # -----------------------------
    # Create mesh grid for visualization
    # -----------------------------
    x_min = min(X_train[:, 0].min(), X_test[:, 0].min()) - 1
    x_max = max(X_train[:, 0].max(), X_test[:, 0].max()) + 1
    y_min = min(X_train[:, 1].min(), X_test[:, 1].min()) - 1
    y_max = max(X_train[:, 1].max(), X_test[:, 1].max()) + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # -----------------------------
    # Train KNN and predict grid
    # -----------------------------
    model = KNN(k=k, p=p)  # Euclidean distance
    model.fit(X_train, y_train)
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    # -----------------------------
    # Plot decision boundary
    # -----------------------------
    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    # Plot training points
    plt.scatter(
        X_train[y_train == 0][:, 0],
        X_train[y_train == 0][:, 1],
        marker="o",
        color="blue",
        label="Train - Class 0",
    )
    plt.scatter(
        X_train[y_train == 1][:, 0],
        X_train[y_train == 1][:, 1],
        marker="o",
        color="red",
        label="Train - Class 1",
    )

    # Plot test points
    plt.scatter(
        X_test[y_test == 0][:, 0],
        X_test[y_test == 0][:, 1],
        marker="x",
        s=100,
        color="blue",
        label="Test - Class 0",
    )
    plt.scatter(
        X_test[y_test == 1][:, 0],
        X_test[y_test == 1][:, 1],
        marker="x",
        s=100,
        color="red",
        label="Test - Class 1",
    )

    # -----------------------------
    # Optionally show neighbors of a single test point
    # -----------------------------
    if show_neighbors:
        if test_point is None:
            raise ValueError("test_point must be provided when show_neighbors=True")
        test_point = np.array(test_point)
        distances = np.sum(np.abs(X_train - test_point) ** p, axis=1) ** (1 / p)
        k_indices = np.argsort(distances)[:k]
        neighbors = X_train[k_indices]

        # Plot test point
        plt.scatter(
            test_point[0],
            test_point[1],
            color="green",
            marker="*",
            s=75,
            label="Test Point",
        )

        # Highlight neighbors
        plt.scatter(
            neighbors[:, 0],
            neighbors[:, 1],
            edgecolor="k",
            facecolor="none",
            s=200,
            linewidth=2,
            label=f"{k} Neighbors",
        )

        # Connect test point to neighbors
        for neighbor in neighbors:
            plt.plot(
                [test_point[0], neighbor[0]],
                [test_point[1], neighbor[1]],
                color="gray",
                linestyle="--",
                linewidth=1,
            )

    plt.title(f"KNN Decision Boundary (k={k}, p={p})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()
