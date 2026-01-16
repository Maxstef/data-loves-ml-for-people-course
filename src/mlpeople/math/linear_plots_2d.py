import numpy as np
import matplotlib.pyplot as plt
from .utils import lighten_color


def plot_vectors(
    vectors,
    labels=None,
    colors=None,
    show_sums=False,
    show_dot=True,
    xlim=(-5, 5),
    ylim=(-5, 5),
    title="Vector Plot",
):
    """
    Plots multiple 2D vectors with optional sum vectors.

    Parameters:
    - vectors: list of vectors (each as a NumPy array)
               e.g., [np.array([1,0]), np.array([0,1])]
    - labels: list of labels for the vectors
    - colors: list of colors for the vectors
    - show_sums: if True, plots sum of all vectors as an additional vector
    - show_dot: if True, calculates and shows dot products for all vectors
    - xlim: tuple for x-axis limits
    - ylim: tuple for y-axis limits
    - title: title of the plot
    """
    plt.figure(figsize=(6, 6))

    n = len(vectors)

    # Display dot products on the plot
    dot_text = ""
    if n > 1 and show_dot:
        for i in range(n):
            for j in range(i + 1, n):
                dot = np.dot(vectors[i], vectors[j])
                label_i = labels[i] if labels else f"v{i+1}"
                label_j = labels[j] if labels else f"v{j+1}"
                dot_text += f"{label_i} · {label_j} = {dot}\n"

    # Default labels and colors
    if labels is None:
        labels = [f"v{i+1} = {vec}" for i, vec in enumerate(vectors)]
    if colors is None:
        # Use a cycle of colors
        default_colors = ["r", "b", "g", "m", "c", "y", "k"]
        colors = [default_colors[i % len(default_colors)] for i in range(n)]

    # Plot each vector
    for i, vec in enumerate(vectors):
        plt.quiver(
            0,
            0,
            vec[0],
            vec[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=colors[i],
            label=labels[i],
        )

    # Optionally plot sum vector
    if show_sums:
        sum_vec = np.sum(vectors, axis=0)
        plt.quiver(
            0,
            0,
            sum_vec[0],
            sum_vec[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="orange",
            alpha=0.6,
            linewidth=2,
            label=f"Sum = {sum_vec}",
        )

    # Axes and grid
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.grid(True)

    # Labels and title
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()

    # Display dot products at bottom-left corner
    if dot_text:
        plt.text(
            xlim[0] + 0.1,
            ylim[0] + 0.1,
            dot_text,
            color="purple",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    plt.show()


def plot_linear_transformation_2d(
    A,
    vectors,
    labels=None,
    show_length=True,
    colors=None,
    xlim=(-5, 5),
    ylim=(-5, 5),
    title="Linear Transformation (2D)",
):
    """
    Plots 2D vectors and their transformed versions under a 2x2 matrix A.

    Features:
    - Original vectors are shown in solid colors.
    - Transformed vectors are shown in lighter shades of the same color.
    - Vector labels can include components and optionally vector lengths.

    Parameters:
    - A: 2x2 NumPy array representing the linear transformation.
    - vectors: list of 2D vectors (NumPy arrays) to transform.
    - labels: optional list of labels for each vector (e.g., ['e1', 'e2']).
              Defaults to ['v1', 'v2', ...].
    - show_length: if True, includes vector magnitude in labels.
    - colors: optional list of colors for original vectors. Defaults to ['r','b','g',...].
    - xlim, ylim: axis limits for the plot.
    - title: plot title.

    Notes:
    - Useful for visualizing linear transformations, including scaling, rotation, shearing, and reflection.
    - Works with any number of 2D vectors.
    """
    plt.figure(figsize=(6, 6))
    n = len(vectors)

    if labels is None:
        labels = [f"v{i+1}" for i in range(n)]

    if colors is None:
        default_colors = ["r", "b", "g", "m", "c", "y", "k"]
        colors = [default_colors[i % len(default_colors)] for i in range(n)]

    # Plot original vectors
    for i, vec in enumerate(vectors):
        label_text = f"{labels[i]} = {vec}"
        if show_length:
            label_text += f", |v|={np.linalg.norm(vec):.2f}"
        plt.quiver(
            0,
            0,
            vec[0],
            vec[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=colors[i],
            label=f"Original: {label_text}",
            alpha=0.9,
        )

    # Plot transformed vectors (lighter shade)
    for i, vec in enumerate(vectors):
        transformed = A @ vec
        label_text = f"{labels[i]}' = {transformed}"
        if show_length:
            label_text += f", |v|={np.linalg.norm(transformed):.2f}"
        lighter_color = lighten_color(colors[i], amount=0.5)
        plt.quiver(
            0,
            0,
            transformed[0],
            transformed[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=lighter_color,
            label=f"Transformed: {label_text}",
            alpha=0.7,
        )

    # Axes, grid, labels
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.show()


def plot_grid_transformation_2d(
    A,
    grid_range=(-5, 5),
    step=1,
    xlim=(-10, 10),
    ylim=(-10, 10),
    title="Grid Transformation",
):
    """
    Visualizes how a 2x2 matrix transforms a 2D grid.

    Parameters:
    - A: 2x2 NumPy array (linear transformation matrix)
    - grid_range: tuple (min, max) defining the grid extent
    - step: spacing between grid lines
    - xlim, ylim: axis limits for visualization
    - title: plot title
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Create grid points
    x = np.arange(grid_range[0], grid_range[1] + step, step)
    y = np.arange(grid_range[0], grid_range[1] + step, step)

    # ----- Original grid -----
    for xi in x:
        axes[0].plot([xi] * len(y), y, color="gray", linewidth=0.8)
    for yi in y:
        axes[0].plot(x, [yi] * len(x), color="gray", linewidth=0.8)

    axes[0].set_title("Original Grid")
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].axvline(0, color="black", linewidth=0.5)
    axes[0].set_aspect("equal")
    axes[0].grid(False)

    # ----- Transformed grid -----
    for xi in x:
        points = np.array([[xi, yi] for yi in y])
        transformed = (A @ points.T).T
        axes[1].plot(transformed[:, 0], transformed[:, 1], color="gray", linewidth=0.8)

    for yi in y:
        points = np.array([[xi, yi] for xi in x])
        transformed = (A @ points.T).T
        axes[1].plot(transformed[:, 0], transformed[:, 1], color="gray", linewidth=0.8)

    axes[1].set_title("Transformed Grid")
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].axvline(0, color="black", linewidth=0.5)
    axes[1].set_aspect("equal")
    axes[1].grid(False)

    fig.suptitle(title, fontsize=14)
    plt.show()


def plot_linear_transformation_with_grid_2d(
    A,
    vectors,
    labels=None,
    grid_range=(-5, 5),
    step=1,
    colors=None,
    xlim=(-10, 10),
    ylim=(-10, 10),
    title="Linear Transformation (Grid + Vectors)",
):
    """
    Visualizes a 2D linear transformation using both grid deformation
    and vector transformation.

    Parameters:
    - A: 2x2 NumPy array (linear transformation matrix)
    - vectors: list of 2D NumPy vectors
    - labels: optional list of vector labels (e.g. ['e1', 'e2'])
    - grid_range: tuple defining grid extent
    - step: grid spacing
    - colors: optional list of colors for vectors
    - xlim, ylim: axis limits
    - title: plot title
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    n = len(vectors)

    if labels is None:
        labels = [f"v{i+1}" for i in range(n)]

    if colors is None:
        base_colors = ["r", "b", "g", "m", "c", "y", "k"]
        colors = [base_colors[i % len(base_colors)] for i in range(n)]

    x = np.arange(grid_range[0], grid_range[1] + step, step)
    y = np.arange(grid_range[0], grid_range[1] + step, step)

    # -------- Original space --------
    ax = axes[0]
    for xi in x:
        ax.plot([xi] * len(y), y, color="lightgray", linewidth=0.8)
    for yi in y:
        ax.plot(x, [yi] * len(x), color="lightgray", linewidth=0.8)

    for i, v in enumerate(vectors):
        ax.quiver(
            0,
            0,
            v[0],
            v[1],
            color=colors[i],
            angles="xy",
            scale_units="xy",
            scale=1,
            label=f"{labels[i]} = {v}",
        )

    ax.set_title("Original Space")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

    # -------- Transformed space --------
    ax = axes[1]
    for xi in x:
        points = np.array([[xi, yi] for yi in y])
        transformed = (A @ points.T).T
        ax.plot(transformed[:, 0], transformed[:, 1], color="lightgray", linewidth=0.8)

    for yi in y:
        points = np.array([[xi, yi] for xi in x])
        transformed = (A @ points.T).T
        ax.plot(transformed[:, 0], transformed[:, 1], color="lightgray", linewidth=0.8)

    for i, v in enumerate(vectors):
        tv = A @ v
        ax.quiver(
            0,
            0,
            tv[0],
            tv[1],
            color=lighten_color(colors[i]),
            angles="xy",
            scale_units="xy",
            scale=1,
            label=f"{labels[i]}' = {tv}",
        )

    ax.set_title("Transformed Space")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

    # -------- Determinant --------
    detA = np.linalg.det(A)
    ax.text(
        0.02,
        0.02,
        f"det(A) = {detA:.2f}\nArea scale = {abs(detA):.2f}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.suptitle(title, fontsize=14)
    plt.show()
