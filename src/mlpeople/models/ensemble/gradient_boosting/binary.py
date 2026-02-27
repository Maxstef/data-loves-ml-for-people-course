import numpy as np
from mlpeople.models.tree import DecisionTreeRegression
from .loss import LogisticLoss


class BinaryGradientBoosting:
    """
    Gradient Boosting for binary classification using LogisticLoss.

    y must be encoded as {-1, +1}.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, verbose=False):
        self.loss = LogisticLoss()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.verbose = verbose

        self.trees = []
        self.loss_history = []
        self.initial_prediction = 0  # log-odds = 0 → p=0.5

    def fit(self, X, y):
        F = np.full_like(y, self.initial_prediction, dtype=float)

        for m in range(self.n_estimators):
            residuals = -self.loss.gradient(y, F)
            tree = DecisionTreeRegression(max_depth=self.max_depth)
            tree.fit(X, residuals)
            update = tree.predict(X)
            F += self.learning_rate * update
            self.trees.append(tree)

            self.loss_history.append(np.mean(self.loss.loss(y, F)))

            if self.verbose:
                avg_loss = np.mean(self.loss.loss(y, F))
                print(f"Iteration {m+1}, Avg Logistic Loss: {avg_loss:.6f}")

    def predict_scores(self, X):
        """
        Return raw scores (logits).
        """
        F = np.full((X.shape[0],), self.initial_prediction, dtype=float)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X):
        """
        Return probabilities of class 1.
        """
        logits = self.predict_scores(X)
        probs = 1 / (1 + np.exp(-logits))
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        """
        Return predicted class labels {-1, +1}.
        """
        probs = self.predict_proba(X)[:, 1]
        return np.where(probs > 0.5, 1, -1)
