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


def plot_dimension_reduction_3d_to_2d(
    A,
    vectors,
    labels=None,
    colors=None,
    title_3d="Original 3D Space",
    title_2d="Reduced 2D Space",
    scale_2d=1,
    show_plane=False,
):
    """
    Visualizes a linear dimension reduction from R^3 to R^2.

    Parameters
    ----------
    A : np.ndarray
        2x3 matrix representing linear reduction from 3D to 2D.
    vectors : list of np.ndarray
        List of 3D vectors to transform.
    labels : list of str, optional
        Labels for the vectors. Defaults to v1, v2, ...
    colors : list of str, optional
        Colors for the vectors. Defaults to ['r','b','g',...].
    title_3d : str
        Title for the 3D original space plot.
    title_2d : str
        Title for the 2D reduced space plot.
    scale_2d : float
        Factor to scale 2D vectors for visibility.
    show_plane : bool
        If True, show the plane spanned by the projection in the 3D plot.
    """

    A = np.asarray(A)
    if A.shape != (2, 3):
        raise ValueError("Matrix A must have shape (2, 3) for 3D → 2D reduction.")

    n = len(vectors)

    # Default labels
    if labels is None:
        labels = [f"v{i+1}" for i in range(n)]

    # Default colors
    if colors is None:
        base_colors = ["r", "b", "g", "m", "c", "y", "k"]
        colors = [base_colors[i % len(base_colors)] for i in range(n)]

    # ---------- 3D plot ----------
    fig_3d = plt.figure(figsize=(10, 8))
    ax3d = fig_3d.add_subplot(111, projection="3d")

    all_points_3d = np.array(vectors)
    margin_3d = max(0.5, all_points_3d.max() * 0.1)
    ax3d.set_xlim(
        all_points_3d[:, 0].min() - margin_3d, all_points_3d[:, 0].max() + margin_3d
    )
    ax3d.set_ylim(
        all_points_3d[:, 1].min() - margin_3d, all_points_3d[:, 1].max() + margin_3d
    )
    ax3d.set_zlim(
        all_points_3d[:, 2].min() - margin_3d, all_points_3d[:, 2].max() + margin_3d
    )
    ax3d.set_box_aspect([1, 1, 1])

    # Optional: show the projection plane
    if show_plane:
        u, v = A[0], A[1]  # two row vectors of the projection
        s = np.linspace(-1.5, 1.5, 10)
        t = np.linspace(-1.5, 1.5, 10)
        S, T = np.meshgrid(s, t)
        X = S * u[0] + T * v[0]
        Y = S * u[1] + T * v[1]
        Z = S * u[2] + T * v[2]
        # ax3d.plot_surface(X, Y, Z, alpha=0.2, color="gray")
        ax3d.plot_wireframe(X, Y, Z, color="blue", linewidth=1, alpha=0.25)
        # ax3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.25)

    for v, label, color in zip(vectors, labels, colors):
        ax3d.quiver(0, 0, 0, v[0], v[1], v[2], color=color, linewidth=2)
        ax3d.text(
            v[0] + margin_3d * 0.05,
            v[1] + margin_3d * 0.05,
            v[2] + margin_3d * 0.05,
            label,
            color=color,
            fontsize=10,
        )

    ax3d.set_title(title_3d, fontsize=12)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    # ---------- 2D plot ----------
    fig_2d, ax2d = plt.subplots(figsize=(8, 6))

    # Compute transformed vectors and scale them for visibility
    transformed_vectors = [A @ v * scale_2d for v in vectors]

    # Compute axis limits based on scaled vectors
    transformed_points_scaled = np.array(transformed_vectors)
    margin_2d = max(0.5, transformed_points_scaled.max() * 0.1)
    ax2d.set_xlim(
        transformed_points_scaled[:, 0].min() - margin_2d,
        transformed_points_scaled[:, 0].max() + margin_2d,
    )
    ax2d.set_ylim(
        transformed_points_scaled[:, 1].min() - margin_2d,
        transformed_points_scaled[:, 1].max() + margin_2d,
    )

    for v2, label, color in zip(transformed_vectors, labels, colors):
        ax2d.quiver(
            0, 0, v2[0], v2[1], color=color, angles="xy", scale_units="xy", scale=1
        )
        ax2d.text(
            v2[0] + margin_2d * 0.05,
            v2[1] + margin_2d * 0.05,
            f"{label}'",
            color=color,
            fontsize=10,
        )

    ax2d.set_title(title_2d, fontsize=12)
    ax2d.set_xlabel("X")
    ax2d.set_ylabel("Y")
    ax2d.axhline(0, color="black", linewidth=0.5)
    ax2d.axvline(0, color="black", linewidth=0.5)
    ax2d.set_aspect("equal")
    ax2d.grid(True)

    plt.show()
