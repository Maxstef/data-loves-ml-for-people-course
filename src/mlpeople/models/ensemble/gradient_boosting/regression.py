import numpy as np
from mlpeople.models.tree import DecisionTreeRegression


# ============================================================
# Simple Squared-Error Gradient Boosting (Regression)
# ============================================================
class SimpleGradientBoostingRegressor:
    """
    Gradient Boosting Regressor using custom DecisionTreeRegression.
    Only supports squared-error loss.

    Key idea:
    -----------
    1. Initialize predictions with mean(y)
    2. Iteratively fit trees to residuals (y - F(x))
    3. Update model: F_new = F_old + learning_rate * tree_prediction
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, verbose=False):
        """
        Parameters:
        -----------
        n_estimators : int
            Number of boosting iterations (trees).

        learning_rate : float
            Shrinkage parameter η to control step size.

        max_depth : int
            Maximum depth of each regression tree.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.verbose = verbose

        self.trees = []  # Stores fitted regression trees
        self.initial_prediction = None  # Initial F0

    def fit(self, X, y):
        """
        Train gradient boosting model.

        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        # -------------------------------------------------
        # STEP 0 — Initialize F0
        # -------------------------------------------------
        # For squared error, best constant prediction = mean(y)
        self.initial_prediction = np.mean(y)

        # Current model predictions F(x)
        F = np.full_like(y, self.initial_prediction, dtype=float)

        # ==========================================================
        # Boosting iterations
        # ==========================================================
        for m in range(self.n_estimators):

            # ------------------------------------------------------
            # STEP 1 — Compute negative gradient
            # ------------------------------------------------------
            # For squared loss:
            #
            #   L = 1/2 (y - F)^2
            #
            #   dL/dF = F - y
            #
            # Negative gradient:
            #
            #   -(F - y) = y - F
            #
            # Which equals residuals.
            #
            residuals = y - F

            # ------------------------------------------------------
            # STEP 2 — Fit regression tree to residuals
            # ------------------------------------------------------
            tree = DecisionTreeRegression(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # ------------------------------------------------------
            # STEP 3 — Predict correction
            # ------------------------------------------------------
            update = tree.predict(X)

            # ------------------------------------------------------
            # STEP 4 — Update model
            # ------------------------------------------------------
            # Functional gradient descent step:
            #
            #   F_new = F_old + η * h_m(x)
            #
            F += self.learning_rate * update

            # Store the trained tree
            self.trees.append(tree)

            # Optional: print training loss
            if self.verbose:
                mse = np.mean((y - F) ** 2)
                print(f"Iteration {m+1}, Training MSE: {mse:.4f}")

    def predict(self, X):
        """
        Make predictions using trained ensemble.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        """

        # Start with initial constant prediction
        F = np.full((X.shape[0],), self.initial_prediction, dtype=float)

        # Add contributions from each tree
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)

        return F
