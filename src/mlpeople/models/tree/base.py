import numpy as np


class BaseDecisionTree:
    """
    Base class for CART-style decision trees.

    Implements:
        - Tree growth
        - Best split search
        - Cost-complexity pruning
        - Traversal

    Subclasses must implement:
        - _leaf_value(y)
        - _init_split_stats(y_sorted)
        - _update_split_stats(...)
        - _impurity_from_stats(...)
        - _merge_stats(left, right)
        - _node_stats(y)
        - _collapse_node(node)
    """

    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        """
        Parameters
        ----------
        max_depth : int
            Maximum allowed depth of tree.
            Controls overfitting (deep tree → high variance).

        min_samples_split : int
            Minimum number of samples required to attempt a split.

        min_samples_leaf : int
            Minimum number of samples required in EACH child node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    # ============================================================
    # Public API
    # ============================================================

    def fit(self, X, y):
        """
        Build the tree recursively starting from root.
        """
        self.n_classes_ = len(np.unique(y))
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        """
        Predict target for each sample by traversing the tree.
        """
        return np.array([self._traverse(x, self.tree) for x in X])

    def prune(self, alpha):
        """
        Post-prune the tree using cost-complexity pruning.

        CART defines regularized objective:

            R_α(T) = R(T) + α |T|

        where:
            R(T) = total impurity of leaves
            |T|  = number of leaves
            α    = complexity penalty (regularization parameter)

        Interpretation:
            - Small α → keep complex tree
            - Large α → prefer smaller tree
        """
        self.tree, _, _ = self._prune_node(self.tree, alpha)

    # ============================================================
    # Tree Growing
    # ============================================================

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively builds the decision tree.

        Stopping conditions:
            1. Reached max depth
            2. Node is pure (only one unique value)
            3. Not enough samples to split
        """

        n_samples, n_features = X.shape

        # ----------------------------
        # STOP CONDITIONS
        # ----------------------------
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or self._is_pure(y)
        ):

            # Leaf node: store value and statistics
            return self._create_leaf(y)

        # ----------------------------
        # Find best feature + threshold
        # ----------------------------
        best_feature, best_threshold = self._best_split(X, y)

        # If no valid split found → make leaf
        if best_feature is None:
            return self._create_leaf(y)

        # ----------------------------
        # Split dataset
        # ----------------------------
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx

        # Recursively grow children
        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        # Node dictionary stores split info and statistics
        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left,
            "right": right,
            "samples": len(y),
            "stats": self._node_stats(y),
        }

    # ============================================================
    # Best Split (Generic)
    # ============================================================

    def _best_split(self, X, y):
        """
        Finds the best split using:

            1. Sort feature values
            2. Sweep threshold from left to right
            3. Update stats incrementally

        Complexity:
            O(n_features × n_samples log n_samples) due to sorting

        Objective:
            Maximize Information Gain / impurity reduction
        """
        n_samples, n_features = X.shape
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature in range(n_features):
            # Sort data by this feature
            sorted_indices = np.argsort(X[:, feature])
            X_sorted = X[sorted_indices]
            y_sorted = y[sorted_indices]

            # Initialize left and right stats
            left_stats, right_stats = self._init_split_stats(y_sorted)

            # Sweep possible split positions
            for i in range(1, n_samples):
                # Move one sample from right → left
                self._update_split_stats(left_stats, right_stats, y_sorted[i - 1])

                # Skip identical values (no meaningful split)
                if X_sorted[i][feature] == X_sorted[i - 1][feature]:
                    continue

                left_size = i
                right_size = n_samples - i

                # Enforce min_samples_leaf
                if (
                    left_size < self.min_samples_leaf
                    or right_size < self.min_samples_leaf
                ):
                    continue

                # Threshold = midpoint between consecutive values
                threshold = (X_sorted[i][feature] + X_sorted[i - 1][feature]) / 2

                # Compute gain for this split
                gain = self._information_gain(left_stats, right_stats)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_threshold = threshold

        return split_idx, split_threshold

    # ============================================================
    # Generic Gain
    # ============================================================

    def _information_gain(self, left_stats, right_stats):
        """
        Generic Information Gain / impurity reduction formula:

            IG = I(parent) - (N_left/N_total)*I(left) - (N_right/N_total)*I(right)
        """
        n_left = left_stats["n"]
        n_right = right_stats["n"]
        n_total = n_left + n_right

        parent_stats = self._merge_stats(left_stats, right_stats)

        parent_imp = self._impurity_from_stats(parent_stats)
        left_imp = self._impurity_from_stats(left_stats)
        right_imp = self._impurity_from_stats(right_stats)

        weighted = (n_left / n_total) * left_imp + (n_right / n_total) * right_imp

        return parent_imp - weighted

    # ============================================================
    # Pruning (Generic)
    # ============================================================

    def _prune_node(self, node, alpha):
        """
        Bottom-up cost-complexity pruning.

        Returns three values:
            1. pruned_node
            2. subtree_error (sum of leaf impurities)
            3. subtree_leaf_count
        """

        # ----------------------------
        # CASE 1: Node is already a leaf
        # ----------------------------
        if "value" in node:
            # Subtree error = N_t * impurity
            error = node["samples"] * self._impurity_from_stats(node["stats"])
            return node, error, 1

        # ----------------------------
        # CASE 2: Internal node → prune children first
        # ----------------------------
        left_node, left_error, left_leaves = self._prune_node(node["left"], alpha)
        right_node, right_error, right_leaves = self._prune_node(node["right"], alpha)

        # Update children (they may have been pruned)
        node["left"] = left_node
        node["right"] = right_node

        # ----------------------------
        # Compute Subtree Statistics
        # ----------------------------
        subtree_error = left_error + right_error
        subtree_leaves = left_leaves + right_leaves

        # ----------------------------
        # Compute error if collapsed
        # ----------------------------
        node_error = node["samples"] * self._impurity_from_stats(node["stats"])

        # ----------------------------
        # Compute effective alpha (weakest link)
        # ----------------------------
        if subtree_leaves > 1:
            alpha_effective = (node_error - subtree_error) / (subtree_leaves - 1)
        else:
            alpha_effective = float("inf")

        # ----------------------------
        # Pruning decision
        # ----------------------------
        if alpha >= alpha_effective:
            # Collapse entire subtree
            return self._collapse_node(node), node_error, 1

        return node, subtree_error, subtree_leaves

    # ============================================================
    # Traversal
    # ============================================================

    def _traverse(self, x, node):
        """
        Recursively traverse tree until leaf is reached.
        """
        if "value" in node:
            return node["value"]

        # Decision rule:
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse(x, node["left"])
        return self._traverse(x, node["right"])

    # ============================================================
    # Abstract Methods / Subclass Responsibilities
    # ============================================================

    def _leaf_value(self, y):
        """Return the prediction for a leaf node"""
        raise NotImplementedError

    def _impurity_from_stats(self, stats):
        """Compute node impurity"""
        raise NotImplementedError

    def _init_split_stats(self, y_sorted):
        """Initialize left and right stats for split"""
        raise NotImplementedError

    def _update_split_stats(self, left, right, value):
        """Update left/right stats when moving a sample"""
        raise NotImplementedError

    def _merge_stats(self, left, right):
        """Merge left/right stats into parent stats"""
        raise NotImplementedError

    def _node_stats(self, y):
        """Compute statistics for a node"""
        raise NotImplementedError

    def _collapse_node(self, node):
        """Convert internal node into a leaf"""
        raise NotImplementedError

    def _is_pure(self, y):
        """Check if all samples belong to the same class/value"""
        return len(np.unique(y)) == 1

    def _create_leaf(self, y):
        """Return a leaf node dictionary"""
        return {
            "value": self._leaf_value(y),
            "samples": len(y),
            "stats": self._node_stats(y),
        }
