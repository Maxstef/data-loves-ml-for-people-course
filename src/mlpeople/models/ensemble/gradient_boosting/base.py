import numpy as np
from mlpeople.models.tree import DecisionTreeRegression


# ============================================================
# General Gradient Boosting (any differentiable loss)
# ============================================================
class GeneralGradientBoosting:
    """
    General Gradient Boosting framework.

    Key idea:
    -------------------------
    1. We start with an initial model F0(x) (usually a constant).
    2. At each iteration, compute the negative gradient of the loss function
       with respect to the current model's predictions.
       This negative gradient is called the pseudo-residual = -dL/dF
    3. Fit a regression tree to these pseudo-residuals.
    4. Update the model by adding a scaled version of the tree's predictions: F_new = F_old + learning_rate * tree_prediction
    5. Repeat for a specified number of boosting iterations.

    Notes:
    - Works for regression and binary classification (with appropriate loss).
    - Multi-class not supported: requires one tree per class per iteration.
    """

    def __init__(
        self, loss, n_estimators=100, learning_rate=0.1, max_depth=3, verbose=False
    ):
        """
        Initialize Gradient Boosting model.

        Parameters:
        -----------
        loss : Loss instance
            Any differentiable loss function implementing .loss() and .gradient()

        n_estimators : int
            Number of boosting iterations (number of trees).

        learning_rate : float
            Shrinkage parameter η. Smaller values → slower learning, often better generalization.

        max_depth : int
            Maximum depth of each regression tree (weak learner).
        """
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.verbose = verbose

        # Store fitted regression trees
        self.trees = []

        # Initial prediction (F0)
        self.initial_prediction = None

    def fit(self, X, y):
        """
        Fit the gradient boosting model to training data.

        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """

        # =============================================
        # STEP 0 — Initialize model F0(x)
        # =============================================
        # In gradient boosting, we start with a constant model that minimizes
        # the loss across all training samples:
        #
        #   F0 = argmin_c Σ L(y_i, c)
        #
        # For squared error regression, F0 = mean(y)
        # For logistic loss, F0 = 0 (log-odds of 0.5 probability)
        self.initial_prediction = np.mean(y)

        # Current predictions for all samples (F(x_i))
        F = np.full_like(y, self.initial_prediction, dtype=float)

        # =============================================
        # STEP 1 — Boosting iterations
        # =============================================
        for m in range(self.n_estimators):

            # -------------------------------------------------
            # STEP 1a — Compute gradients of the loss
            # -------------------------------------------------
            # Compute derivative of loss w.r.t. model predictions:
            #   g_i = dL(y_i, F(x_i)) / dF(x_i)
            gradients = self.loss.gradient(y, F)

            # -------------------------------------------------
            # STEP 1b — Compute negative gradient (pseudo-residuals)
            # -------------------------------------------------
            # In functional gradient descent, we fit a model to:
            #   r_i = -g_i
            # For squared loss: r_i = y_i - F(x_i) (residuals)
            residuals = -gradients

            # -------------------------------------------------
            # STEP 2 — Fit regression tree to pseudo-residuals
            # -------------------------------------------------
            # Each regression tree h_m(x) approximates the negative gradient
            # This is the "weak learner" in boosting
            tree = DecisionTreeRegression(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # -------------------------------------------------
            # STEP 3 — Predict correction for all samples
            # -------------------------------------------------
            # Use the tree to compute the update for all training samples
            update = tree.predict(X)

            # -------------------------------------------------
            # STEP 4 — Update the model (functional gradient descent step)
            # -------------------------------------------------
            # Scale the update by learning_rate for stability
            # Functional gradient descent:
            #   F_new = F_old + η * h_m(x)
            F += self.learning_rate * update

            # Store the fitted tree for future predictions
            self.trees.append(tree)

            # -------------------------------------------------
            # Optional — monitor training progress
            # -------------------------------------------------
            # Compute average loss over training data to see convergence
            if self.verbose:
                avg_loss = np.mean(self.loss.loss(y, F))
                print(f"Iteration {m+1}, Avg Loss: {avg_loss:.6f}")

    def predict(self, X):
        """
        Predict target values for X using the trained boosting ensemble.

        Steps:
        1. Start with initial prediction F0
        2. Add contributions from all fitted trees
        3. Return final predictions
        """
        # Start with initial constant prediction
        F = np.full((X.shape[0],), self.initial_prediction, dtype=float)

        # Add predictions of all trees scaled by learning_rate
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)

        return F
