import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap


def plot_decision_boundary(clf, X, y, title="Decision Boundary"):

    # --- Create grid ---
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    # --- Predict on grid ---
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)

    # --- Custom colormap for -1 / +1 ---
    cmap = ListedColormap(["#3b4cc0", "#b40426"])  # blue / red

    # --- Plot regions ---
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

    # --- Plot boundary line explicitly (where prediction changes) ---
    plt.contour(
        xx, yy, Z, levels=[0], colors="black", linewidths=1, linestyles="dotted"
    )

    # --- Plot points ---
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor="k", s=60)

    # --- Manual legend ---
    plt.scatter([], [], c="#3b4cc0", edgecolor="k", label="Class -1")
    plt.scatter([], [], c="#b40426", edgecolor="k", label="Class +1")
    plt.legend()

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def plot_boosting_stages(clf, X, y, feature_names=None):
    """
    Visualize how AdaBoost builds its decision boundary stage by stage,
    highlight misclassified points, and show stump details (alpha, feature, threshold).

    Parameters
    ----------
    clf : AdaBoost
        Trained AdaBoost model.
    X : np.ndarray (n_samples, 2)
        Feature matrix (must be 2D for visualization).
    y : np.ndarray (n_samples,)
        Labels (-1, +1)
    feature_names : list of str, optional
        Names of features to display instead of indices
    """

    if X.shape[1] != 2:
        raise ValueError("plot_boosting_stages works only for 2D data.")

    n_stages = len(clf.clfs)

    # ---- Determine grid layout (max 4 per row) ----
    max_cols = 4
    n_cols = min(max_cols, n_stages)
    n_rows = math.ceil(n_stages / max_cols)

    # ---- Create mesh grid ----
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    grid = np.c_[xx.ravel(), yy.ravel()]

    # ---- Updated subplot layout ----
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # Ensure axes is always iterable
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    axes = axes.flatten()

    for t in range(n_stages):

        stump = clf.clfs[t]

        # ---- Partial ensemble prediction ----
        F_train = np.zeros(len(X))
        F_grid = np.zeros(grid.shape[0])

        for s in clf.clfs[: t + 1]:
            F_train += s.alpha * s.predict(X)
            F_grid += s.alpha * s.predict(grid)

        y_pred_train = np.sign(F_train)
        y_pred_grid = np.sign(F_grid).reshape(xx.shape)

        # ---- Identify wrong predictions ----
        wrong_mask = y_pred_train != y

        # ---- Plot decision boundary ----
        axes[t].contourf(xx, yy, y_pred_grid, alpha=0.3)

        # Correct points
        axes[t].scatter(
            X[~wrong_mask][:, 0],
            X[~wrong_mask][:, 1],
            c=y[~wrong_mask],
            cmap="bwr",
            edgecolors="k",
            label="Correct",
        )

        # Wrong points (highlighted)
        axes[t].scatter(
            X[wrong_mask][:, 0],
            X[wrong_mask][:, 1],
            facecolors="none",
            edgecolors="red",
            s=150,
            linewidths=2,
            label="Wrong",
        )

        # ---- Create title with stump info ----
        feat_idx = stump.feature_index
        feat_name = feature_names[feat_idx] if feature_names else f"Feature {feat_idx}"
        thresh = stump.threshold
        alpha = stump.alpha

        axes[t].set_title(f"Stage {t+1}\n" f"{feat_name} ≤ {thresh:.2f}, α={alpha:.2f}")
        axes[t].legend()

    # ---- Hide unused subplots ----
    for i in range(n_stages, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
