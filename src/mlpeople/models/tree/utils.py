def print_tree(node, depth=0):
    """
    Recursively prints tree structure in readable format.

    Parameters
    ----------
    node : dict
        Root node (tree.tree)
    depth : int
        Current recursion depth (used for indentation)

    Tree node structure expected:
        Internal node:
            {
                "feature": int,
                "threshold": float,
                "left": dict,
                "right": dict,
                "samples": int,
                "counts": np.array
            }

        Leaf node:
            {
                "value": int,
                "samples": int,
                "counts": np.array
            }
    """

    indent = "  " * depth

    # Leaf node
    if "value" in node:
        print(f"{indent}Leaf: class={node['value']} " f"(samples={node['samples']})")
        return

    # Internal node
    print(
        f"{indent}Node: X[{node['feature']}] "
        f"<= {node['threshold']:.4f} "
        f"(samples={node['samples']})"
    )

    print_tree(node["left"], depth + 1)
    print_tree(node["right"], depth + 1)


def pretty_print(node, prefix=""):
    """
    Prints tree with branch structure.

    Uses ASCII tree-like formatting for better readability.
    """

    # Leaf
    if "value" in node:
        print(
            prefix + f"└── Leaf "
            f"(class={node['value']}, "
            f"samples={node['samples']})"
        )
        return

    # Internal node
    print(prefix + f"├── X[{node['feature']}] " f"<= {node['threshold']:.4f}")

    # Left branch
    pretty_print(node["left"], prefix + "│   ")

    # Right branch
    pretty_print(node["right"], prefix + "    ")


def decision_path(x, node):
    """
    Returns decision path for a single sample.

    Parameters
    ----------
    x : np.array shape (n_features,)
        Single input sample

    node : dict
        Root node of trained tree (tree.tree)

    Returns
    -------
    path : list[str]
        Human-readable explanation of decisions
    """

    path = []

    while "value" not in node:

        feature = node["feature"]
        threshold = node["threshold"]

        if x[feature] <= threshold:
            path.append(f"X[{feature}] <= {threshold:.4f} → go LEFT")
            node = node["left"]
        else:
            path.append(f"X[{feature}] > {threshold:.4f} → go RIGHT")
            node = node["right"]

    path.append(f"Predict class {node['value']}")

    return path
