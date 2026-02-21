import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def generate_train_test_data(
    n_samples=50, cluster_std=5.0, random_state=42, test_size=0.2, show_plot=True
):
    """
    Generate a synthetic 2D classification dataset using Gaussian blobs,
    split it into training and test sets, and optionally visualize it.

    Parameters
    ----------
    n_samples : int, default=50
        Total number of samples to generate (including train + test).

    cluster_std : float, default=5.0
        Standard deviation of each cluster. Higher values increase overlap
        between classes, introducing more "noise" into the dataset.

    random_state : int, default=42
        Seed for reproducibility of the random number generator.

    test_size : float, default=0.2
        Fraction of the dataset to allocate to the test set.

    show_plot : bool, default=True
        If True, displays a scatter plot of the train and test points.

    Returns
    -------
    X_train : ndarray of shape (n_train_samples, 2)
        Training features.

    X_test : ndarray of shape (n_test_samples, 2)
        Test features.

    y_train : ndarray of shape (n_train_samples,)
        Training labels (0 or 1).

    y_test : ndarray of shape (n_test_samples,)
        Test labels (0 or 1).

    Notes
    -----
    - Training points are plotted as circles ('o'), and test points as crosses ('x').
    - Useful for experimenting with KNN classifiers and visualizing decision boundaries.
    """

    # Generate data
    X, y = make_blobs(
        n_samples=n_samples,
        centers=2,
        cluster_std=cluster_std,
        random_state=random_state,
    )

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if show_plot:
        # Plot
        plt.figure(figsize=(8, 6))

        # Training points
        plt.scatter(
            X_train[y_train == 0][:, 0],
            X_train[y_train == 0][:, 1],
            marker="o",
            label="Train - Class 0",
        )

        plt.scatter(
            X_train[y_train == 1][:, 0],
            X_train[y_train == 1][:, 1],
            marker="o",
            label="Train - Class 1",
        )

        # Test points
        plt.scatter(
            X_test[y_test == 0][:, 0],
            X_test[y_test == 0][:, 1],
            marker="x",
            s=100,
            label="Test - Class 0",
        )

        plt.scatter(
            X_test[y_test == 1][:, 0],
            X_test[y_test == 1][:, 1],
            marker="x",
            s=100,
            label="Test - Class 1",
        )

        plt.title("Train/Test Split Visualization")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True)
        plt.show()

    return X_train, X_test, y_train, y_test
