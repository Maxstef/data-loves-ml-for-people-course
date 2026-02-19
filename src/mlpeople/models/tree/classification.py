import numpy as np
from .base import BaseDecisionTree


class DecisionTreeClassificationBase(BaseDecisionTree):
    """
    Base-inherited version of a CART classification tree.

    Differences from standalone version:
        - Delegates tree growth, splitting, pruning, and traversal to BaseDecisionTree.
        - Stores node statistics in a structured dict (`stats`) for generic computations.
        - Cleaner and more maintainable; easier to extend for other tree types.

    Supports:
        - Entropy (Information Gain)
        - Gini Impurity

    Tree structure (delegated to BaseDecisionTree):
        Internal node:
            {
                "feature": feature_index,
                "threshold": value,
                "left": {...},
                "right": {...},
                "stats": node_stats
            }

        Leaf node:
            {
                "value": predicted_class,
                "stats": node_stats
            }
    """

    def __init__(
        self, max_depth=5, min_samples_split=2, min_samples_leaf=1, criterion="entropy"
    ):

        super().__init__(max_depth, min_samples_split, min_samples_leaf)
        self.criterion = criterion

    # ============================================================
    # Subclass-specific methods
    # ============================================================

    def _leaf_value(self, y):
        """Majority class in the node"""
        return np.bincount(y).argmax()

    def _node_stats(self, y):
        """Compute class counts at this node"""
        return {"counts": np.bincount(y, minlength=self.n_classes_), "n": len(y)}

    def _impurity_from_stats(self, stats):
        """Compute node impurity based on criterion"""
        counts = stats["counts"]
        total = np.sum(counts)
        if total == 0:
            return 0

        probs = counts / total

        if self.criterion == "entropy":
            # Entropy: H = - Σ p_i log2(p_i)
            return -np.sum([p * np.log2(p) for p in probs if p > 0])
        elif self.criterion == "gini":
            # Gini impurity: G = 1 - Σ p_i²
            return 1 - np.sum(probs**2)
        else:
            raise ValueError("Criterion must be 'entropy' or 'gini'")

    def _init_split_stats(self, y_sorted):
        """Initialize left (empty) and right (full) stats for split"""
        left_stats = {"counts": np.zeros(self.n_classes_, dtype=int), "n": 0}
        right_stats = {
            "counts": np.bincount(y_sorted, minlength=self.n_classes_),
            "n": len(y_sorted),
        }
        return left_stats, right_stats

    def _update_split_stats(self, left, right, value):
        """Move one sample from right → left"""
        left["counts"][value] += 1
        left["n"] += 1
        right["counts"][value] -= 1
        right["n"] -= 1

    def _merge_stats(self, left, right):
        """Combine left and right stats into parent"""
        return {"counts": left["counts"] + right["counts"], "n": left["n"] + right["n"]}

    def _collapse_node(self, node):
        """Convert internal node into a leaf using majority class"""
        return {
            "value": np.argmax(node["stats"]["counts"]),
            "stats": node["stats"],
            "samples": node["samples"],
        }

    # ============================================================
    # Extra API
    # ============================================================

    def predict_proba(self, X):
        """
        Predict class probabilities for each sample.

        Returns:
            numpy array of shape (n_samples, n_classes)
        """
        proba_list = []

        for x in X:
            node = self.tree
            # Traverse to leaf
            while "value" not in node:
                if x[node["feature"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]

            counts = node["stats"]["counts"]
            total = np.sum(counts)
            probs = counts / total
            proba_list.append(probs)

        return np.array(proba_list)


class DecisionTreeClassification:
    """
    Standalone CART-style Decision Tree for classification.

    Differences from Base-inherited version:
        - All logic (tree growth, split search, pruning, traversal) implemented inside this class.
        - Uses simpler `counts` array directly in nodes.
        - Slightly longer and more “classic” implementation, useful for learning and debugging.

    Supports:
        - Entropy (Information Gain)
        - Gini Impurity

    Tree structure:
        Internal node:
            {
                "feature": feature_index,
                "threshold": value,
                "left": {...},
                "right": {...}
            }

        Leaf node:
            {
                "value": predicted_class
            }
    """

    def __init__(
        self, max_depth=5, min_samples_split=2, min_samples_leaf=1, criterion="entropy"
    ):
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

        criterion : str
            "entropy"  → Information Gain
            "gini"     → Gini impurity
        """

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree = None

    # ============================================================
    # Public API
    # ============================================================

    def fit(self, X, y):
        """
        Build the tree recursively starting from root.
        """
        self.n_classes_ = len(np.unique(y))  # Store global class count
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        """
        Predict class for each sample by traversing the tree.
        """
        return np.array([self._traverse(x, self.tree) for x in X])

    def prune(self, alpha):
        """
        Post-prune the tree using cost-complexity pruning.
        """
        self.tree, _, _ = self._prune_node(self.tree, alpha)

    def predict_proba(self, X):
        """
        Predict class probabilities for each sample.

        Returns:
            numpy array of shape (n_samples, n_classes)
        """
        proba_list = []

        for x in X:
            node = self.tree
            # Traverse to leaf
            while "value" not in node:
                if x[node["feature"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]

            counts = node["counts"]
            total = np.sum(counts)
            probs = counts / total
            proba_list.append(probs)

        return np.array(proba_list)

    # ============================================================
    # Tree Construction
    # ============================================================

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively builds the decision tree.

        Stopping conditions:
            1. Reached max depth
            2. Node is pure (only 1 class)
            3. Not enough samples to split
        """

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # ----------------------------
        # STOP CONDITIONS
        # ----------------------------
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):

            # Leaf prediction = majority class
            leaf_value = self._most_common_label(y)
            return {
                "value": leaf_value,
                "samples": len(y),
                "counts": np.bincount(y, minlength=self.n_classes_),
            }

        # ----------------------------
        # Find best feature + threshold
        # ----------------------------
        best_feature, best_threshold = self._best_split(X, y)

        # If no valid split found → make leaf
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return {
                "value": leaf_value,
                "samples": len(y),
                "counts": np.bincount(y, minlength=self.n_classes_),
            }

        # ----------------------------
        # Split dataset
        # ----------------------------
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold

        # Recursively grow children
        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left,
            "right": right,
            "samples": len(y),
            "counts": np.bincount(y, minlength=self.n_classes_),
        }

    # ============================================================
    # Best Split Search (Optimized Version)
    # ============================================================

    def _best_split(self, X, y):
        """
        Finds the best split using:

            1. Sort feature values
            2. Sweep threshold from left to right
            3. Update class counts incrementally

        Complexity:
            O(n_features × n_samples log n_samples)
            (sorting dominates)

        We maximize:

            Information Gain = Parent Impurity − Weighted Child Impurity
        """

        n_samples, n_features = X.shape
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature in range(n_features):

            # Sort data by this feature
            sorted_indices = np.argsort(X[:, feature])
            X_sorted = X[sorted_indices]
            y_sorted = y[sorted_indices]

            # Count class frequencies
            right_counts = np.bincount(y_sorted, minlength=self.n_classes_)
            left_counts = np.zeros(self.n_classes_, dtype=int)

            # Sweep possible split positions
            for i in range(1, n_samples):

                # Move one sample from right → left
                label = y_sorted[i - 1]
                left_counts[label] += 1
                right_counts[label] -= 1

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

                gain = self._information_gain_from_counts(left_counts, right_counts)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_threshold = threshold

        return split_idx, split_threshold

    # ============================================================
    # Information Gain
    # ============================================================

    def _information_gain_from_counts(self, left_counts, right_counts):
        """
        Information Gain formula:

            IG = I(parent)
                 − (N_left/N_total) * I(left)
                 − (N_right/N_total) * I(right)

        where:
            I(.) = impurity measure (Entropy or Gini)
        """

        n_left = np.sum(left_counts)
        n_right = np.sum(right_counts)
        n_total = n_left + n_right

        parent_counts = left_counts + right_counts

        parent_impurity = self._impurity_from_counts(parent_counts)
        left_impurity = self._impurity_from_counts(left_counts)
        right_impurity = self._impurity_from_counts(right_counts)

        weighted_impurity = (n_left / n_total) * left_impurity + (
            n_right / n_total
        ) * right_impurity

        return parent_impurity - weighted_impurity

    # ============================================================
    # Impurity Measures
    # ============================================================

    def _impurity_from_counts(self, counts):
        """
        Dispatch impurity function based on criterion.
        """
        if self.criterion == "entropy":
            return self._entropy_from_counts(counts)

        elif self.criterion == "gini":
            return self._gini_from_counts(counts)

        else:
            raise ValueError("Criterion must be 'entropy' or 'gini'")

    def _entropy_from_counts(self, counts):
        """
        Entropy formula:

            H = − Σ p_i log2(p_i)

        where:
            p_i = class probability

        Measures uncertainty.
        Maximum when classes equally distributed.
        """

        total = np.sum(counts)
        if total == 0:
            return 0

        probs = counts / total
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def _gini_from_counts(self, counts):
        """
        Gini impurity formula:

            G = 1 − Σ p_i²

        Interpretation:
            Probability of misclassifying a random sample
            if labeled according to node distribution.
        """

        total = np.sum(counts)
        if total == 0:
            return 0

        probs = counts / total
        return 1 - np.sum(probs**2)

    def _node_impurity(self, counts):
        return self._impurity_from_counts(counts)

    # ============================================================
    # Pruning Logic
    # ============================================================

    def _prune_node(self, node, alpha):
        """
        Performs bottom-up cost-complexity pruning.

        This implements CART's cost-complexity objective:

            R_α(T) = R(T) + α |T|

        Where:
            R(T)  = total impurity of leaves
            |T|   = number of leaves
            α     = complexity penalty (regularization parameter)

        Interpretation:
            - Small α → keep complex tree
            - Large α → prefer smaller tree

        -----------------------------------------------------------
        This function returns THREE values:

            1. pruned_node
            2. subtree_error (R(T))
            3. subtree_leaf_count (|T|)

        This allows parent nodes to evaluate whether collapsing
        the entire subtree is beneficial.
        -----------------------------------------------------------
        """

        # ============================================================
        # CASE 1: Node is already a leaf
        # ============================================================
        if "value" in node:
            """
            If node is a leaf:

            Its subtree error is:

                R(t) = N_t * I(t)

            where:
                N_t = number of samples at node
                I(t) = impurity (Gini or Entropy)

            Since it is already a leaf:
                |T_t| = 1
            """

            leaf_error = node["samples"] * self._node_impurity(node["counts"])
            return node, leaf_error, 1

        # ============================================================
        # CASE 2: Internal node → prune children first (bottom-up)
        # ============================================================

        left_node, left_error, left_leaves = self._prune_node(node["left"], alpha)
        right_node, right_error, right_leaves = self._prune_node(node["right"], alpha)

        # Update children (they may have been pruned)
        node["left"] = left_node
        node["right"] = right_node

        # ============================================================
        # Compute Subtree Statistics
        # ============================================================

        """
        Subtree error is sum of leaf errors:

            R(T_t) = R(T_left) + R(T_right)

        Subtree leaf count:

            |T_t| = |T_left| + |T_right|
        """

        subtree_error = left_error + right_error
        subtree_leaves = left_leaves + right_leaves

        # ============================================================
        # Compute Collapsed (Single Leaf) Statistics
        # ============================================================

        """
        If we collapse this entire subtree into ONE leaf,
        its error becomes:

            R(t) = N_t * I(t)

        and number of leaves becomes:

            |T_t| = 1
        """

        node_error = node["samples"] * self._node_impurity(node["counts"])
        collapsed_leaves = 1

        # ============================================================
        # Compute Effective Alpha (Weakest Link)
        # ============================================================

        """
        CART defines effective alpha for subtree rooted at t as:

            α_effective =
                (R(t) - R(T_t)) / (|T_t| - 1)

        Where:
            R(t)     = error if collapsed
            R(T_t)   = error of full subtree
            |T_t|    = number of leaves in subtree

        Interpretation:

            Numerator:
                Increase in training error if we collapse subtree

            Denominator:
                Number of leaves removed

        So α_effective measures:
            "Error increase per leaf removed"

        Smaller α_effective → weaker link → prune first.
        """

        if subtree_leaves > 1:
            alpha_effective = (node_error - subtree_error) / (subtree_leaves - 1)
        else:
            alpha_effective = float("inf")

        # ============================================================
        # Pruning Decision
        # ============================================================

        """
        We prune this subtree if:

            α >= α_effective

        Why?

        Because the regularized objective is:

            R_α(T) = R(T) + α|T|

        Compare:

            Keep subtree:
                R(T_t) + α|T_t|

            Collapse subtree:
                R(t) + α

        Collapse if:

            R(t) + α <= R(T_t) + α|T_t|

        Rearranging gives exactly:

            α >= (R(t) - R(T_t)) / (|T_t| - 1)
        """

        if alpha >= alpha_effective:
            """
            Collapse entire subtree into a single leaf.
            Majority class becomes prediction.
            """

            return (
                {
                    "value": np.argmax(node["counts"]),
                    "samples": node["samples"],
                    "counts": node["counts"],
                },
                node_error,
                1,
            )

        # Otherwise, keep subtree as is
        return node, subtree_error, subtree_leaves

    # ============================================================
    # Utilities
    # ============================================================

    def _most_common_label(self, y):
        """
        Returns majority class in node.
        """
        return np.bincount(y).argmax()

    def _traverse(self, x, node):
        """
        Recursively traverse tree until leaf is reached.
        """

        # If leaf node → return stored class
        if "value" in node:
            return node["value"]

        # Decision rule:
        #   if feature_value <= threshold → go left
        #   else → go right
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse(x, node["left"])
        return self._traverse(x, node["right"])
