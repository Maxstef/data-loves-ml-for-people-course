import numpy as np

from .base import BaseDecisionTree


class DecisionTreeRegression(BaseDecisionTree):
    """
    CART Regression Tree.

    Uses Mean Squared Error (variance reduction).
    """

    def _leaf_value(self, y):
        return np.mean(y)

    # Stats stored as:
    # {
    #   "n": sample_count,
    #   "sum": sum_y,
    #   "sum_sq": sum_y_squared
    # }

    def _node_stats(self, y):
        return {"n": len(y), "sum": np.sum(y), "sum_sq": np.sum(y**2)}

    def _init_split_stats(self, y_sorted):

        right = {
            "n": len(y_sorted),
            "sum": np.sum(y_sorted),
            "sum_sq": np.sum(y_sorted**2),
        }

        left = {"n": 0, "sum": 0.0, "sum_sq": 0.0}

        return left, right

    def _update_split_stats(self, left, right, value):

        left["n"] += 1
        left["sum"] += value
        left["sum_sq"] += value**2

        right["n"] -= 1
        right["sum"] -= value
        right["sum_sq"] -= value**2

    def _merge_stats(self, left, right):

        return {
            "n": left["n"] + right["n"],
            "sum": left["sum"] + right["sum"],
            "sum_sq": left["sum_sq"] + right["sum_sq"],
        }

    def _impurity_from_stats(self, stats):

        n = stats["n"]
        if n == 0:
            return 0

        mean = stats["sum"] / n

        # Variance formula:
        # Var = (sum(y^2)/n) - mean^2
        variance = (stats["sum_sq"] / n) - mean**2

        return variance

    def _collapse_node(self, node):

        mean_value = node["stats"]["sum"] / node["stats"]["n"]

        return {"value": mean_value, "samples": node["samples"], "stats": node["stats"]}
