"""
Utilities for generating and visualizing 2D multilabel classification datasets.

Multilabel classification differs from multiclass:

Multiclass:
    One sample → exactly ONE class.

Multilabel:
    One sample → ANY number of labels (including zero).

Because of this, decision boundaries are not shared —
each label effectively learns its own binary classifier.

This module focuses on building intuition by plotting
each label's decision surface independently.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.datasets import make_multilabel_classification


def generate_2D_multilabel_classification_data(
    n_samples=200,
    n_classes=3,
    n_labels=2,
    random_state=42,
    show_plot=True,
    noise=0.01,
):
    """
    Generate a synthetic 2D multilabel classification dataset.

    Uses sklearn's `make_multilabel_classification`, constrained
    to two features so decision regions can be visualized.

    Parameters
    ----------
    n_samples : int
        Number of samples.

    n_classes : int
        Total number of possible labels.

    n_labels : int
        Average number of labels per sample.

    random_state : int
        Ensures reproducibility.

    show_plot : bool
        If True, plots each label in a separate subplot.

    noise : float (0–1)
        Fraction of label entries randomly flipped.
        Useful for making the problem less separable.

        Example:
            noise=0.05 → 5% of label bits are inverted.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Feature matrix.

    y : ndarray of shape (n_samples, n_classes)
        Binary indicator matrix.
    """

    X, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=2,
        n_classes=n_classes,
        n_labels=n_labels,
        random_state=random_state,
    )

    # Inject label noise by flipping random bits
    if noise > 0:
        rng = np.random.default_rng(random_state)

        n_entries = y.size
        n_noisy = int(noise * n_entries)

        indices = np.unravel_index(
            rng.choice(n_entries, n_noisy, replace=False),
            y.shape,
        )

        y[indices] = 1 - y[indices]

    if show_plot:
        _plot_multilabel_data(X, y)

    return X, y


def _plot_multilabel_data(X, Y):
    """
    Internal helper for visualizing multilabel datasets.

    Each subplot corresponds to one binary classification task.
    """

    n_labels = Y.shape[1]

    plt.figure(figsize=(4 * n_labels, 4))

    legend_elements = [
        Patch(facecolor="blue", edgecolor="k", label="0"),
        Patch(facecolor="red", edgecolor="k", label="1"),
    ]

    for i in range(n_labels):
        ax = plt.subplot(1, n_labels, i + 1)

        ax.scatter(
            X[:, 0],
            X[:, 1],
            c=Y[:, i],
            cmap="coolwarm",
            edgecolor="k",
            alpha=0.8,
        )

        ax.set_title(f"Label {i}")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.legend(handles=legend_elements, title="Label value")
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()


def plot_multilabel_boundaries(X, Y, model, h=0.02, figsize=(12, 4)):
    """
    Visualize decision boundaries for multilabel classifiers.

    IMPORTANT:
    There is no single boundary in multilabel classification.
    Instead, each label has its own binary classifier.

    This function plots one subplot per label.

    Parameters
    ----------
    X : ndarray
        Feature matrix (must be 2D).

    Y : ndarray
        Binary label matrix.

    model : fitted multilabel classifier
        Must expose `.estimators_`
        (e.g. MultiOutputClassifier, OneVsRestClassifier).

    h : float
        Mesh grid resolution.
        Smaller values produce smoother regions but cost more CPU.

    figsize : tuple
        Size of the matplotlib figure.
    """

    n_labels = Y.shape[1]

    plt.figure(figsize=figsize)

    legend_elements = [
        Patch(facecolor="blue", edgecolor="k", label="0"),
        Patch(facecolor="red", edgecolor="k", label="1"),
    ]

    # Shared mesh bounds (avoids recomputing per subplot)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    for i in range(n_labels):
        ax = plt.subplot(1, n_labels, i + 1)

        estimator = model.estimators_[i]

        Z = estimator.predict(grid)
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

        ax.scatter(
            X[:, 0],
            X[:, 1],
            c=Y[:, i],
            cmap="coolwarm",
            edgecolor="k",
            alpha=0.8,
        )

        ax.legend(handles=legend_elements, title="Label value")

        ax.set_title(f"Label {i}")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()
