"""
Utilities for generating and visualizing 2D single-label classification datasets.

This module is primarily intended for:

- Educational purposes
- Visual intuition building
- Experimenting with decision boundaries
- Understanding OvR vs OvO strategies

All plotting functions assume **2D feature space** so decision regions
and separator lines can be visualized.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.datasets import make_classification


def generate_2D_classification_data(
    n_samples=200,
    n_classes=3,
    flip_y=0.01,
    random_state=42,
    show_plot=True,
):
    """
    Generate a synthetic 2D multiclass classification dataset.

    Uses sklearn's `make_classification` with parameters tuned for
    visual separability in two dimensions.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.

    n_classes : int
        Number of target classes.

    flip_y : float
        Fraction of labels randomly flipped to introduce noise.
        Higher values make the classification problem harder.

    random_state : int
        Ensures reproducible results.

    show_plot : bool
        If True, displays a scatter plot of the generated dataset.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Feature matrix.

    y : ndarray of shape (n_samples,)
        Target labels.
    """

    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=n_classes,
        flip_y=flip_y,
        random_state=random_state,
    )

    if show_plot:
        plt.figure(figsize=(6, 5))

        plt.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
        )

        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title(f"2D Visualization of {n_classes}-Class Dataset")

        plt.show()

    return X, y


def plot_decision_regions(model, X, y, ax=None, h=0.02, alpha=0.3):
    """
    Plot decision regions for a fitted classifier.

    A mesh grid is created over the feature space and predictions are made
    for every point to visualize how the classifier partitions the space.

    Parameters
    ----------
    model : fitted sklearn classifier
        Must implement `.predict()`.

    X : ndarray
        Feature matrix (must be 2D).

    y : ndarray
        Target labels.

    ax : matplotlib.axes, optional
        Axis to plot on. If None, uses current axis.

    h : float
        Step size of the mesh grid.
        Smaller values produce smoother boundaries but are slower.

    alpha : float
        Transparency of the decision region shading.
    """

    if ax is None:
        ax = plt.gca()

    # Define plotting bounds
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict over grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot filled contour
    ax.contourf(xx, yy, Z, alpha=alpha)

    # Plot original samples
    for cls in np.unique(y):
        ax.scatter(X[y == cls, 0], X[y == cls, 1], label=f"Class {cls}")

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_aspect("equal", adjustable="box")


def plot_classifier_lines(X, y, classifier_type="ovr", ax=None):
    """
    Overlay linear separator lines for OvR or OvO logistic regression.

    Helpful for understanding how linear classifiers split feature space.

    OvR:
        One classifier per class vs the rest.

    OvO:
        One classifier for every pair of classes.

    Parameters
    ----------
    X : ndarray
        Feature matrix (2D).

    y : ndarray
        Target labels.

    classifier_type : {"ovr", "ovo"}
        Strategy used to build the multiclass classifier.

    ax : matplotlib.axes, optional
        Axis to plot on.
    """

    if ax is None:
        ax = plt.gca()

    base_clf = LogisticRegression()

    if classifier_type == "ovr":
        clf = OneVsRestClassifier(base_clf)
    elif classifier_type == "ovo":
        clf = OneVsOneClassifier(base_clf)
    else:
        raise ValueError("classifier_type must be 'ovr' or 'ovo'")

    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_vals = np.linspace(x_min, x_max, 200)

    if classifier_type == "ovr":
        estimators = enumerate(clf.estimators_)
    else:
        n_classes = len(np.unique(y))
        pairs = [(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)]
        estimators = zip(pairs, clf.estimators_)

    for label, estimator in estimators:
        w1, w2 = estimator.coef_[0]
        b = estimator.intercept_[0]

        if abs(w2) < 1e-6:
            # Vertical separator
            x_line = -b / w1
            if x_min <= x_line <= x_max:
                ax.axvline(x_line, linestyle="--", label=f"Separator {label}")
        else:
            y_vals = -(w1 * x_vals + b) / w2
            mask = (y_vals >= y_min) & (y_vals <= y_max)

            ax.plot(
                x_vals[mask],
                y_vals[mask],
                linestyle="--",
                label=f"Separator {label}",
            )


def plot_multiclass_combined(
    X, y, classifier_type="ovr", show_lines=True, figsize=(14, 8), title=None
):
    """
    Plot decision regions together with linear separator lines.

    This is the most informative visualization for linear classifiers
    because it shows both:

    • the predicted regions
    • the exact hyperplanes producing them

    Recommended for teaching and debugging models.
    """

    fig, ax = plt.subplots(figsize=figsize)

    clf = (
        OneVsRestClassifier(LogisticRegression())
        if classifier_type == "ovr"
        else OneVsOneClassifier(LogisticRegression())
    )

    clf.fit(X, y)

    plot_decision_regions(clf, X, y, ax=ax)

    if show_lines:
        plot_classifier_lines(X, y, classifier_type=classifier_type, ax=ax)

    ax.legend()

    if title is None:
        title = f"{classifier_type.upper()} Combined Plot"

    ax.set_title(title)
    plt.show()


def plot_multiclass_boundaries(X, y, classifier_type="ovr", figsize=(7, 5), title=None):
    """
    Plot only the linear separator boundaries (no region shading).

    Useful when you want a cleaner visualization focused purely on
    the geometry of the decision hyperplanes.
    Parameters:
    - X: np.ndarray of shape (n_samples, 2)
    - y: np.ndarray of shape (n_samples,)
    - classifier_type: "ovr" for OneVsRest, "ovo" for OneVsOne
    - figsize: tuple for figure size
    - title: plot title
    """

    # choose classifier
    base_clf = LogisticRegression()
    if classifier_type == "ovr":
        clf = OneVsRestClassifier(base_clf)
    elif classifier_type == "ovo":
        clf = OneVsOneClassifier(base_clf)
    else:
        raise ValueError("classifier_type must be 'ovr' or 'ovo'")

    clf.fit(X, y)

    # plot data
    plt.figure(figsize=figsize)
    for cls in np.unique(y):
        plt.scatter(X[y == cls, 0], X[y == cls, 1], label=f"Class {cls}")

    # plot bounds
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect("equal", adjustable="box")

    x_vals = np.linspace(x_min, x_max, 200)

    # plot separators
    if classifier_type == "ovr":
        # each estimator corresponds to one class vs rest
        for i, estimator in enumerate(clf.estimators_):
            w1, w2 = estimator.coef_[0]
            b = estimator.intercept_[0]

            if abs(w2) < 1e-6:  # vertical line
                x_line = -b / w1
                if x_min <= x_line <= x_max:
                    plt.axvline(
                        x_line, linestyle="--", label=f"Separator for class {i}"
                    )
            else:
                y_vals = -(w1 * x_vals + b) / w2
                mask = (y_vals >= y_min) & (y_vals <= y_max)
                plt.plot(
                    x_vals[mask],
                    y_vals[mask],
                    linestyle="--",
                    label=f"Separator for class {i}",
                )

    elif classifier_type == "ovo":
        # get class pairs
        n_classes = len(np.unique(y))
        pairs = []
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                pairs.append((i, j))

        for (cls1, cls2), estimator in zip(pairs, clf.estimators_):
            w1, w2 = estimator.coef_[0]
            b = estimator.intercept_[0]

            if abs(w2) < 1e-6:
                x_line = -b / w1
                if x_min <= x_line <= x_max:
                    plt.axvline(x_line, linestyle="--", label=f"{cls1} vs {cls2}")
            else:
                y_vals = -(w1 * x_vals + b) / w2
                mask = (y_vals >= y_min) & (y_vals <= y_max)
                plt.plot(
                    x_vals[mask],
                    y_vals[mask],
                    linestyle="--",
                    label=f"{cls1} vs {cls2}",
                )

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    if title is None:
        title = f"{classifier_type.upper()} Decision Boundaries"
    plt.title(title)
    plt.show()
