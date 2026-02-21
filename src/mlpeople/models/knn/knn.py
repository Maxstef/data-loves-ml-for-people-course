import numpy as np
from collections import Counter


class KNN:
    """
    K-Nearest Neighbors classifier with configurable Minkowski distance.

    Parameters:
    -----------
    k : int
        Number of neighbors to use.

    p : int or float
        Minkowski distance parameter.

        p = 1 → Manhattan distance (L1)
        p = 2 → Euclidean distance (L2)
        p > 2 → More emphasis on large coordinate differences

    weighted : bool
        If True → use distance-weighted voting
        If False → use simple majority voting
    """

    def __init__(self, k=5, p=2, weighted=False):
        # Number of neighbors to use for prediction
        self.k = k

        # Minkowski distance parameter (p = 2 → Euclidean distance)
        self.p = p

        # Whether to use weighted voting (closer neighbors have more influence)
        self.weighted = weighted

        # These will be assigned during fitting
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store training data.

        Unlike many models, KNN does NOT learn parameters.
        It simply memorizes the dataset.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix.
        y : array-like, shape (n_samples,)
            Target labels.
        """
        # Convert to NumPy arrays (ensures consistent math operations)
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _minkowski_distance(self, x1, x2):
        """
        Compute Minkowski distance between two vectors.

        Formula:
        (sum(|x1 - x2|^p))^(1/p)

        Special cases:
        - p = 1 → Manhattan distance
        - p = 2 → Euclidean distance
        """
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)

    def _euclidean_distance(self, x1, x2):
        """
        Compute Euclidean distance between two points.

        Formula:
        sqrt(sum((x1 - x2)^2))

        This measures straight-line distance in feature space.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _get_neighbors(self, x):
        """
        For a single test sample x:
        1. Compute distance to every training sample
        2. Sort by distance
        3. Return indices of k nearest samples
        """

        distances = []

        # Compute distance from x to every training sample
        for x_train in self.X_train:
            distance = self._minkowski_distance(x, x_train)
            distances.append(distance)

        # Convert list to NumPy array
        distances = np.array(distances)

        # argsort() returns indices that would sort the array
        # We take first k smallest distances
        k_indices = np.argsort(distances)[: self.k]

        # Return both indices and their distances
        return k_indices, distances[k_indices]

    def predict(self, X):
        """
        Predict class labels for multiple samples.

        Steps for each test sample:
        1. Find k nearest neighbors
        2. Collect their labels
        3. Perform voting
        """

        X = np.array(X)
        predictions = []

        for x in X:

            # Find nearest neighbors
            k_indices, k_distances = self._get_neighbors(x)

            # Get labels of nearest neighbors
            k_labels = self.y_train[k_indices]

            if self.weighted:
                """
                Weighted voting:
                Each neighbor contributes weight = 1 / distance
                Closer neighbors influence prediction more.
                """

                # Add small epsilon to avoid division by zero
                weights = 1 / (k_distances + 1e-10)

                label_weights = {}

                # Sum weights per class
                for label, weight in zip(k_labels, weights):
                    label_weights[label] = label_weights.get(label, 0) + weight

                # Choose label with highest total weight
                predicted_label = max(label_weights, key=label_weights.get)

            else:
                """
                Simple majority voting:
                The most common class among neighbors wins.
                """
                predicted_label = Counter(k_labels).most_common(1)[0][0]

            predictions.append(predicted_label)

        return np.array(predictions)

    def predict_proba(self, X):
        """
        Estimate class probabilities for KNN (works for multi-class).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Probability distribution over classes for each sample.
            Each row sums to 1.
        """
        X = np.array(X)
        classes = np.unique(self.y_train)
        n_classes = len(classes)
        proba = []

        for x in X:
            k_indices, _ = self._get_neighbors(x)
            k_labels = self.y_train[k_indices]

            # Count occurrences of each class
            counts = np.array([np.sum(k_labels == c) for c in classes])
            # Divide by k to get probability
            prob = counts / self.k
            proba.append(prob)

        return np.array(proba)
