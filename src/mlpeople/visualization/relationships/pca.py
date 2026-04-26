import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def plot_3d_pca_depth(
    X,
    y,
    n_components=3,
    limit=22,
    cmap="tab10",
    point_base_size=3,
    point_scale=60,
    figsize=(16, 12),
    name="",
    random_state=0,
):
    """
    3D PCA scatter with depth-based point scaling.
    """

    # --- PCA ---
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)

    # --- Figure ---
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)

    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap=cmap, alpha=0.6)

    ax.set_title(f"3D PCA with Depth-Scaled Points {name}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    # --- update function ---
    def update_sizes(event=None):
        elev = np.deg2rad(ax.elev)
        azim = np.deg2rad(ax.azim)

        cam_dir = np.array(
            [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)]
        )

        depth = X_pca @ cam_dir

        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())

        sizes = point_base_size + (1 - depth_norm) * point_scale

        sc.set_sizes(sizes)
        fig.canvas.draw_idle()

    # --- legend ---
    legend = ax.legend(*sc.legend_elements(), title="Class")
    ax.add_artist(legend)

    # --- interactivity ---
    fig.canvas.mpl_connect("motion_notify_event", update_sizes)

    # initial render
    update_sizes()

    plt.show()

    return pca, X_pca


def plot_3d_pca_depth_plotly(
    X,
    y,
    n_components=3,
    point_size=5,
    opacity=0.7,
    title="3D PCA with Depth Effect (Plotly)",
):
    """
    3D PCA visualization with simulated depth effect in Plotly.
    """

    # --- PCA ---
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # --- camera-based depth approximation ---
    # (Plotly handles rotation internally; we simulate static depth scaling)
    z = X_pca[:, 2]

    z_norm = (z - z.min()) / (z.max() - z.min())
    sizes = 3 + (1 - z_norm) * 10  # closer = bigger

    # --- figure ---
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            z=X_pca[:, 2],
            mode="markers",
            marker=dict(
                size=sizes, color=y, colorscale="Turbo", opacity=opacity, showscale=True
            ),
            name="Digits",
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.show()

    return pca, X_pca
