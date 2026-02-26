import matplotlib.pyplot as plt
import numpy as np


def _compute_positions(
    node, depth, x_offset, positions, level_width=1.0, max_depth=None
):
    """
    Recursively compute (x, y) positions for each node.

    Returns:
        next available x position after placing this subtree
    """

    # Stop expanding deeper if max_depth reached
    if max_depth is not None and depth >= max_depth:
        positions[id(node)] = (x_offset, -depth)
        return x_offset + level_width

    if "value" in node:
        positions[id(node)] = (x_offset, -depth)
        return x_offset + level_width

    # Compute left subtree
    x_left = _compute_positions(
        node["left"], depth + 1, x_offset, positions, level_width, max_depth
    )

    # Compute right subtree
    x_right = _compute_positions(
        node["right"], depth + 1, x_left, positions, level_width, max_depth
    )

    # Place current node centered above children
    mid = (positions[id(node["left"])][0] + positions[id(node["right"])][0]) / 2
    positions[id(node)] = (mid, -depth)

    return x_right


def _draw_tree(ax, node, positions, parent=None, depth=0, max_depth=None):
    """
    Draw nodes and edges recursively.
    """

    x, y = positions[id(node)]

    node_color = _get_class_color(node)

    ax.text(
        x,
        y,
        _node_label(node),
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc=node_color, ec="black", alpha=0.9),
    )

    # Draw edge from parent
    if parent is not None:
        px, py = positions[id(parent)]
        ax.plot([px, x], [py, y])

    # Stop expanding deeper if max_depth reached
    if max_depth is not None and depth >= max_depth:
        return

    # Recursively draw children
    if "value" not in node:
        _draw_tree(ax, node["left"], positions, node, depth + 1, max_depth)
        _draw_tree(ax, node["right"], positions, node, depth + 1, max_depth)


def _node_label(node):
    """
    Returns formatted string displayed inside a node box.
    """

    # Leaf node
    if "value" in node:
        if "counts" in node:
            counts = node["counts"]
        elif "stats" in node:
            counts = node["stats"]["counts"]
        else:
            counts = None

        if counts is not None:
            return (
                f"Leaf\n"
                f"class = {node['value']}\n"
                f"samples = {node['samples']}\n"
                f"counts = {counts}"
            )
        else:
            return f"Leaf\nvalue = {node['value']}"

    # Internal node
    return (
        f"X[{node['feature']}] <= {node['threshold']:.2f}\n"
        f"samples = {node['samples']}"
    )


def _get_class_color(node, alpha=0.9):
    """
    Returns color based on majority class in node.
    Works for binary and multi-class.
    """

    # If leaf → use node["value"]
    if "value" in node:
        majority_class = node["value"]
    else:
        # Internal node → majority from counts
        if "counts" in node:
            majority_class = np.argmax(node["counts"])
        elif "stats" in node:
            majority_class = np.argmax(node["stats"]["counts"])
        else:
            return "lightgray"

    # Simple binary palette (extend if needed)
    colors = {
        0: "#8ecae6",  # light blue
        1: "#ffb703",  # orange
        2: "#90be6d",  # green
        3: "#f28482",  # red-ish
    }

    return colors.get(majority_class, "lightgray")


def plot_tree_matplotlib(tree, figsize=(12, 6), max_depth=None):
    """
    Plots decision tree using matplotlib.

    Parameters
    ----------
    tree : dict
        Root node of trained DecisionTree (tree.tree)
    max_depth : int or None
        Maximum depth to display (None = full tree)
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    positions = {}
    _compute_positions(tree, 0, 0, positions, max_depth=max_depth)

    _draw_tree(ax, tree, positions, max_depth=max_depth)

    plt.show()


def plot_reg_tree_matplotlib(tree, figsize=(12, 6)):
    """
    Plot a regression tree using matplotlib.

    Parameters
    ----------
    tree : dict
        Root node of trained DecisionTreeRegression (tree.tree)
    figsize : tuple
        Figure size
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    # -----------------------------
    # Helper to compute x,y positions
    # -----------------------------
    def _compute_positions(node, depth=0, x=0, positions=None, x_offset=[0]):
        if positions is None:
            positions = {}

        # Leaf node
        if "value" in node:
            positions[id(node)] = (x_offset[0], -depth)
            x_offset[0] += 1
            return positions

        # Internal node
        positions = _compute_positions(node["left"], depth + 1, x, positions, x_offset)
        positions = _compute_positions(node["right"], depth + 1, x, positions, x_offset)
        # Parent node at middle of children
        left_x = positions[id(node["left"])][0]
        right_x = positions[id(node["right"])][0]
        positions[id(node)] = ((left_x + right_x) / 2, -depth)
        return positions

    # -----------------------------
    # Helper to draw nodes and edges
    # -----------------------------
    def _draw_tree(node, positions, ax):
        x, y = positions[id(node)]
        # Leaf
        if "value" in node:
            ax.text(
                x,
                y,
                f"{node['value']:.2f}",
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="lightblue", edgecolor="black", boxstyle="round,pad=0.3"
                ),
            )
            return

        # Internal node
        ax.text(
            x,
            y,
            f"X[{node['feature']}] <= {node['threshold']:.2f}",
            ha="center",
            va="center",
            bbox=dict(
                facecolor="lightgreen", edgecolor="black", boxstyle="round,pad=0.3"
            ),
        )

        # Draw edges
        for child in ["left", "right"]:
            cx, cy = positions[id(node[child])]
            ax.plot([x, cx], [y, cy], color="black")
            _draw_tree(node[child], positions, ax)

    # Compute positions and draw tree
    positions = _compute_positions(tree)
    _draw_tree(tree, positions, ax)
    plt.show()
