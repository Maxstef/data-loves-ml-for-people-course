from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def compute_pruning_path(X_train, y_train):
    """
    Compute cost complexity pruning path for a DecisionTreeClassifier.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets

    Returns:
        tuple: (ccp_alphas, impurities)
    """
    model = DecisionTreeClassifier(random_state=42)
    path = model.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # Plot total impurity vs alpha
    plt.figure(figsize=(8, 5))
    plt.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    plt.xlabel("Effective alpha")
    plt.ylabel("Total impurity of leaves")
    plt.title("Total Impurity vs alpha for training set")
    plt.xscale("log")
    plt.show()

    return ccp_alphas, impurities


def train_trees_over_alphas(X_train, y_train, X_test, y_test, ccp_alphas, step=200):
    """
    Train multiple DecisionTreeClassifiers over a list of ccp_alphas.

    Args:
        X_train, y_train, X_test, y_test
        ccp_alphas (list[float]): list of candidate alphas
        step (int): step size to subsample alphas

    Returns:
        dict: {
            "alphas": list of alphas,
            "clfs": list of fitted classifiers,
            "train_scores": list of train accuracy,
            "test_scores": list of test accuracy,
            "node_counts": list of number of nodes,
            "depths": list of max depth
        }
    """
    clfs = []
    tested_ccp_alphas = []

    for i in range(0, len(ccp_alphas), step):
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alphas[i])
        clf.fit(X_train, y_train)
        clfs.append(clf)
        tested_ccp_alphas.append(ccp_alphas[i])

    # add last alpha explicitly (more than 1 node)
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alphas[-2])
    clf.fit(X_train, y_train)
    clfs.append(clf)
    tested_ccp_alphas.append(ccp_alphas[-2])

    node_counts = [clf.tree_.node_count for clf in clfs]
    depths = [clf.tree_.max_depth for clf in clfs]
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    print(
        f"Number of nodes in the last tree: {clfs[-1].tree_.node_count} with ccp_alpha: {tested_ccp_alphas[-1]}"
    )

    return {
        "alphas": tested_ccp_alphas,
        "clfs": clfs,
        "node_counts": node_counts,
        "depths": depths,
        "train_scores": train_scores,
        "test_scores": test_scores,
    }


def plot_tree_metrics(results):
    """
    Plot number of nodes and depth vs alpha.
    """
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].plot(
        results["alphas"][:-2],
        results["node_counts"][:-2],
        marker="o",
        drawstyle="steps-post",
    )
    ax[0].set_xlabel("Alpha")
    ax[0].set_ylabel("Number of nodes")
    ax[0].set_title("Number of nodes vs alpha")

    ax[1].plot(
        results["alphas"][:-2],
        results["depths"][:-2],
        marker="o",
        drawstyle="steps-post",
    )
    ax[1].set_xlabel("Alpha")
    ax[1].set_ylabel("Depth")
    ax[1].set_title("Depth vs alpha")

    fig.tight_layout()
    plt.show()


def plot_accuracy_vs_alpha(results):
    """
    Plot training and testing accuracy vs alpha.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(
        results["alphas"],
        results["train_scores"],
        marker="o",
        label="Train",
        drawstyle="steps-post",
    )
    plt.plot(
        results["alphas"],
        results["test_scores"],
        marker="o",
        label="Test",
        drawstyle="steps-post",
    )
    plt.xscale("log")
    plt.xlabel("Alpha (log scale)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs alpha")
    plt.legend()
    plt.show()


def prune_decision_tree_experiment(
    X_train, y_train, X_test, y_test, step_coarse=200, step_fine=20, refine_range=None
):
    """
    Perform cost-complexity pruning analysis for a Decision Tree using
    a coarse search over `ccp_alpha` values and optionally a refined search.

    The function:
    1. Computes the cost-complexity pruning path.
    2. Runs a coarse search over `ccp_alpha` values using the given step size.
    3. Automatically suggests a refinement range around the best alpha
       (based on validation accuracy) if `refine_range` is not provided.
    4. Optionally performs a refined search within the specified index range.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix.

    y_train : array-like of shape (n_samples,)
        Training target values.

    X_test : array-like of shape (n_samples, n_features)
        Validation/test feature matrix.

    y_test : array-like of shape (n_samples,)
        Validation/test target values.

    step_coarse : int, default=200
        Step size used to subsample `ccp_alpha` values during the coarse search.

    step_fine : int, default=20
        Step size used during the refined search (if executed).

    refine_range : tuple[int, int] or None, default=None
        Index range (start, end) within `ccp_alphas` to perform refined search.
        If None, the function automatically computes and returns a suggested
        refinement range based on the best coarse result.

    Returns
    -------
    dict
        Dictionary containing:
            - "coarse": results from the coarse search.
            - "refined": results from the refined search (None if not executed).
            - "refine_range": tuple[int, int] suggesting the index range
              for refined search within `ccp_alphas`.

    Notes
    -----
    The refinement range is computed around the best coarse alpha index
    using `step_coarse // 2` as the window size.

    The returned `refine_range` refers to indices of the full `ccp_alphas`
    array, not indices of the coarse search results.
    """

    ccp_alphas, impurities = compute_pruning_path(X_train, y_train)

    # Coarse search
    coarse_results = train_trees_over_alphas(
        X_train, y_train, X_test, y_test, ccp_alphas, step=step_coarse
    )
    plot_tree_metrics(coarse_results)
    plot_accuracy_vs_alpha(coarse_results)

    refined_results = None
    if refine_range is None:
        best_idx = np.argmax(coarse_results["test_scores"])
        best_alpha_idx = best_idx * step_coarse
        half_step = step_coarse // 2  # integer division
        start_idx = max(0, best_alpha_idx - half_step)
        end_idx = min(len(ccp_alphas), best_alpha_idx + half_step + 1)
        refine_range = (start_idx, end_idx)
        print(
            f"Suggested refine range: indices {start_idx}-{end_idx-1}, "
            f"alphas {ccp_alphas[start_idx]:.6f}-{ccp_alphas[end_idx-1]:.6f}"
        )
    else:
        # ------------------ 4. Execute refined search ------------------
        start, end = refine_range
        refined_results = train_trees_over_alphas(
            X_train, y_train, X_test, y_test, ccp_alphas[start:end], step=step_fine
        )
        plot_accuracy_vs_alpha(refined_results)
        print("Max test score in refined search:", max(refined_results["test_scores"]))

    # ------------------ 5. Return results dictionary ------------------
    return {
        "coarse": coarse_results,
        "refined": refined_results,
        "refine_range": refine_range,
    }
