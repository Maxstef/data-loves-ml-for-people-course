import numpy as np
import matplotlib.pyplot as plt


def plot_vectors_3d(
    vectors,
    labels=None,
    colors=None,
    show_sums=False,
    show_dot=True,
    xlim=(-5, 5),
    ylim=(-5, 5),
    zlim=(-5, 5),
    title="3D Vector Plot",
):
    """
    Plots multiple 3D vectors and optionally shows:
      - Sum vector
      - Scalar (dot) products in a corner of the plot

    Parameters:
    - vectors: list of 3D vectors (NumPy arrays)
    - labels: list of labels; defaults to 'vec = [x, y, z]'
    - colors: list of colors
    - show_sums: if True, plots sum of vectors
    - xlim, ylim, zlim: axis limits
    - title: plot title
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    n = len(vectors)

    # Compute scalar products
    dot_text = ""
    if n > 1 and show_dot:
        for i in range(n):
            for j in range(i + 1, n):
                dot = np.dot(vectors[i], vectors[j])
                label_i = labels[i] if labels else f"v{i+1}"
                label_j = labels[j] if labels else f"v{j+1}"
                dot_text += f"{label_i}·{label_j}={dot}\n"

    # Default labels
    if labels is None:
        labels = [f"vec = {vec}" for vec in vectors]

    # Default colors
    if colors is None:
        default_colors = ["r", "b", "g", "m", "c", "y", "k"]
        colors = [default_colors[i % len(default_colors)] for i in range(n)]

    # Plot each vector
    for i, vec in enumerate(vectors):
        ax.quiver(
            0,
            0,
            0,
            vec[0],
            vec[1],
            vec[2],
            color=colors[i],
            label=labels[i],
            linewidth=1.5,
        )

    # Plot sum vector
    if show_sums:
        sum_vec = np.sum(vectors, axis=0)
        ax.quiver(
            0,
            0,
            0,
            sum_vec[0],
            sum_vec[1],
            sum_vec[2],
            color="orange",
            alpha=0.6,
            linewidth=2,
            label=f"Sum = {sum_vec}",
        )

    # Display dot products as text in the corner
    if dot_text:
        ax.text2D(
            0.05,
            0.95,
            dot_text,
            transform=ax.transAxes,
            fontsize=10,
            color="purple",
            fontweight="bold",
        )

    # Set limits and labels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()

    plt.show()


def plot_linear_transformation_with_grid_3d(
    A,
    vectors,
    labels=None,
    grid_range=(-3, 3),
    step=1,
    colors=None,
    title="3D Linear Transformation",
):
    """
    Visualizes a 3D linear transformation using coordinate grid lines
    and vector transformation.

    Parameters:
    - A: 3x3 NumPy array (linear transformation matrix)
    - vectors: list of 3D NumPy vectors
    - labels: optional list of vector labels
    - grid_range: tuple defining grid extent
    - step: grid spacing
    - colors: optional list of colors
    - title: plot title
    """

    A = np.asarray(A)
    fig = plt.figure(figsize=(14, 6))

    detA = np.linalg.det(A)

    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    if labels is None:
        labels = [f"v{i+1}" for i in range(len(vectors))]

    if colors is None:
        base_colors = ["r", "b", "g", "m", "c", "y", "k"]
        colors = [base_colors[i % len(base_colors)] for i in range(len(vectors))]

    x = np.arange(grid_range[0], grid_range[1] + step, step)

    # -------- Original grid (coordinate lines) --------
    for xi in x:
        ax1.plot([xi] * len(x), x, 0, color="lightgray", linewidth=0.8)
        ax1.plot(x, [xi] * len(x), 0, color="lightgray", linewidth=0.8)
        ax1.plot(0, x, [xi] * len(x), color="lightgray", linewidth=0.8)

    # -------- Original vectors --------
    for v, label, color in zip(vectors, labels, colors):
        ax1.quiver(0, 0, 0, v[0], v[1], v[2], color=color, linewidth=2)
        ax1.text(v[0], v[1], v[2], f"{label} = {v}", color=color)

    ax1.set_title("Original Space")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_box_aspect([1, 1, 1])

    # -------- Transformed grid --------
    for xi in x:
        pts = np.array([[xi, yi, 0] for yi in x])
        tpts = (A @ pts.T).T
        ax2.plot(tpts[:, 0], tpts[:, 1], tpts[:, 2], color="lightgray", linewidth=0.8)

        pts = np.array([[0, xi, zi] for zi in x])
        tpts = (A @ pts.T).T
        ax2.plot(tpts[:, 0], tpts[:, 1], tpts[:, 2], color="lightgray", linewidth=0.8)

        pts = np.array([[xi, 0, zi] for zi in x])
        tpts = (A @ pts.T).T
        ax2.plot(tpts[:, 0], tpts[:, 1], tpts[:, 2], color="lightgray", linewidth=0.8)

    # -------- Transformed vectors --------
    for v, label, color in zip(vectors, labels, colors):
        tv = A @ v
        ax2.quiver(0, 0, 0, tv[0], tv[1], tv[2], color=color, linewidth=2)
        ax2.text(tv[0], tv[1], tv[2], f"{label}' = {tv}", color=color)

    ax2.set_title("Transformed Space")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_box_aspect([1, 1, 1])

    ax2.text2D(
        0.02,
        0.02,
        f"det(A) = {detA:.2f}\nVolume scale = {abs(detA):.2f}",
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
