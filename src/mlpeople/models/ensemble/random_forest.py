import numpy as np

from mlpeople.models.tree import DecisionTreeClassification
from mlpeople.models.tree import DecisionTreeRegression


class RandomForestClassification:
    """
    Random Forest classifier built on top of DecisionTreeClassification.

    Implements:
        - Bootstrap sampling
        - Random feature subspacing
        - Majority voting
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="entropy",
        max_features="sqrt",
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []
        self.features_per_tree = []

        if random_state is not None:
            np.random.seed(random_state)

    # ============================================================
    # Fit
    # ============================================================

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.n_classes_ = len(np.unique(y))
        self.trees = []
        self.features_per_tree = []

        for _ in range(self.n_estimators):

            # -----------------------------
            # Bootstrap sampling
            # -----------------------------
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # -----------------------------
            # Random feature selection
            # -----------------------------
            n_selected = self._get_max_features(n_features)
            feature_indices = np.random.choice(n_features, n_selected, replace=False)

            X_boot_subset = X_boot[:, feature_indices]

            # -----------------------------
            # Train tree
            # -----------------------------
            tree = DecisionTreeClassification(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
            )

            tree.fit(X_boot_subset, y_boot)

            self.trees.append(tree)
            self.features_per_tree.append(feature_indices)

    # ============================================================
    # Predict
    # ============================================================

    def predict(self, X):
        predictions = []

        for tree, feature_indices in zip(self.trees, self.features_per_tree):
            X_subset = X[:, feature_indices]
            preds = tree.predict(X_subset)
            predictions.append(preds)

        predictions = np.array(predictions)

        # Majority vote
        return np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes_).argmax(),
            axis=0,
            arr=predictions,
        )

    # ============================================================
    # Predict Probabilities
    # ============================================================

    def predict_proba(self, X):
        proba_sum = np.zeros((X.shape[0], self.n_classes_))

        for tree, feature_indices in zip(self.trees, self.features_per_tree):
            X_subset = X[:, feature_indices]
            proba_sum += tree.predict_proba(X_subset)

        return proba_sum / self.n_estimators

    # ============================================================
    # Helpers
    # ============================================================

    def _get_max_features(self, n_features):
        if self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features == "log2":
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        else:
            return n_features


class RandomForestRegression:
    """
    Random Forest Regressor (CART-based).

    Uses:
        - Bootstrap sampling (bagging)
        - Random feature subsampling
        - Averaging of tree predictions
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=None,
    ):
        """
        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest.

        max_depth : int or None
            Maximum depth of each tree.

        min_samples_split : int
        min_samples_leaf : int

        max_features : {"sqrt", "log2", int, None}
            Number of features to consider at each tree.
                - "sqrt" → sqrt(n_features)
                - "log2" → log2(n_features)
                - int → fixed number
                - None → use all features

        random_state : int or None
        """

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []
        self.feature_subsets = []

    # ============================================================
    # Fit
    # ============================================================

    def fit(self, X, y):

        rng = np.random.default_rng(self.random_state)

        n_samples, n_features = X.shape

        self.trees = []
        self.feature_subsets = []

        for _ in range(self.n_estimators):

            # ----------------------------
            # 1. Bootstrap sampling
            # ----------------------------
            indices = rng.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            # ----------------------------
            # 2. Feature subsampling
            # ----------------------------
            n_selected = self._get_n_features(n_features)
            feature_indices = rng.choice(n_features, n_selected, replace=False)

            X_sample_subset = X_sample[:, feature_indices]

            # ----------------------------
            # 3. Train regression tree
            # ----------------------------
            tree = DecisionTreeRegression(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )

            tree.fit(X_sample_subset, y_sample)

            self.trees.append(tree)
            self.feature_subsets.append(feature_indices)

    # ============================================================
    # Predict
    # ============================================================

    def predict(self, X):
        """
        Final prediction = average of tree predictions
        """

        tree_preds = []

        for tree, feature_indices in zip(self.trees, self.feature_subsets):
            X_subset = X[:, feature_indices]
            preds = tree.predict(X_subset)
            tree_preds.append(preds)

        tree_preds = np.array(tree_preds)

        # Mean across trees
        return np.mean(tree_preds, axis=0)

    # ============================================================
    # Helpers
    # ============================================================

    def _get_n_features(self, n_features):

        if self.max_features == "sqrt":
            return int(np.sqrt(n_features))

        elif self.max_features == "log2":
            return int(np.log2(n_features))

        elif isinstance(self.max_features, int):
            return self.max_features

        else:
            return n_features
