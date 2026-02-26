import numpy as np


class DecisionStump:
    """
    A decision stump is a one-level decision tree.

    It splits the data based on:
        - One feature
        - One threshold
        - One polarity (direction of inequality)

    In AdaBoost, it serves as the weak learner.
    """

    def __init__(self):
        # Direction of inequality:
        # polarity = 1  ->  predict -1 if x < threshold
        # polarity = -1 ->  predict -1 if x > threshold
        self.polarity = 1

        # Index of the feature used for splitting
        self.feature_index = None

        # Threshold value used for splitting
        self.threshold = None

        # Weight of this weak learner (computed later in AdaBoost)
        self.alpha = None

    def fit(self, X, y, sample_weights):
        """
        Train the stump using weighted error minimization.

        Parameters:
            X : feature matrix (n_samples, n_features)
            y : labels (-1 or +1)
            sample_weights : importance weight of each sample

        Returns:
            min_error : the minimum weighted classification error found
        """

        n_samples, n_features = X.shape
        min_error = float("inf")

        # Try every feature
        for feature_i in range(n_features):

            # Take one feature column
            X_column = X[:, feature_i]

            # Possible split points (unique values of feature)
            thresholds = np.unique(X_column)

            # Try every possible threshold
            for threshold in thresholds:

                # Try both inequality directions
                for polarity in [1, -1]:

                    # Start by predicting all samples as +1
                    predictions = np.ones(n_samples)

                    # Apply split rule:
                    # If condition is satisfied → predict -1
                    predictions[polarity * X_column < polarity * threshold] = -1

                    # Compute weighted classification error
                    # Only count weights of misclassified points
                    error = np.sum(sample_weights[y != predictions])

                    # Keep track of best split
                    if error < min_error:
                        min_error = error
                        self.polarity = polarity
                        self.threshold = threshold
                        self.feature_index = feature_i

        return min_error

    def predict(self, X):
        """
        Predict labels using the learned feature, threshold and polarity.
        """

        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]

        predictions = np.ones(n_samples)

        # Apply learned decision rule
        predictions[self.polarity * X_column < self.polarity * self.threshold] = -1

        return predictions


class AdaBoost:
    """
    AdaBoost implementation for binary classification (-1, +1).

    Builds an ensemble of weak learners (DecisionStump).
    Each learner is assigned a weight (alpha).
    Final prediction is weighted vote of all learners.
    """

    def __init__(self, n_clf=5, early_stopping=False, tol=1e-6):
        # Number of weak classifiers (boosting rounds)
        self.n_clf = n_clf

        # List to store trained weak learners
        self.clfs = []

        # early stopping props
        self.early_stopping = early_stopping
        self.tol = tol

    def fit(self, X, y):
        """
        Train AdaBoost on dataset.

        Steps:
        1. Initialize uniform sample weights
        2. Train weak learner
        3. Compute weighted error
        4. Compute alpha (learner importance)
        5. Update sample weights
        6. Repeat
        """

        n_samples, _ = X.shape

        # Step 1: Initialize weights uniformly
        # Every sample starts equally important
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []

        F = np.zeros(n_samples)  # cumulative margin for early stopping

        for t in range(self.n_clf):

            # Step 2: Train weak learner using current weights
            clf = DecisionStump()
            error = clf.fit(X, y, w)

            # Avoid division by zero
            error = max(error, 1e-10)

            # Step 3: Compute alpha (importance of this classifier)
            # If error small → alpha large (strong learner)
            # If error close to 0.5 → alpha small (weak learner)
            clf.alpha = 0.5 * np.log((1 - error) / error)

            # Get predictions from this weak learner
            predictions = clf.predict(X)

            # Step 4: Update sample weights
            #
            # If sample correctly classified:
            #   y * prediction = +1
            #   weight decreases
            #
            # If misclassified:
            #   y * prediction = -1
            #   weight increases
            #
            w *= np.exp(-clf.alpha * y * predictions)

            # Normalize weights so they sum to 1
            w /= np.sum(w)

            # Store trained classifier
            self.clfs.append(clf)

            F += clf.alpha * predictions

            # Early stopping check
            if self.early_stopping:
                ensemble_error = np.mean(np.sign(F) != y)
                if ensemble_error <= self.tol:
                    print(
                        f"Early stopping at stage {t+1}, ensemble error={ensemble_error:.6f}"
                    )
                    break

    def predict(self, X):
        """
        Final prediction is sign of weighted sum of weak learners.

        F(x) = sign( sum(alpha_t * h_t(x)) )
        """

        # Multiply each classifier's prediction by its alpha
        clf_preds = np.array([clf.alpha * clf.predict(X) for clf in self.clfs])

        # Sum along classifiers
        y_pred = np.sum(clf_preds, axis=0)

        # Return sign of total vote
        return np.sign(y_pred)
