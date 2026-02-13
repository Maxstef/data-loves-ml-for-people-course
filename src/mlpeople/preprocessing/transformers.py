from sklearn.base import BaseEstimator, TransformerMixin
import operator
import pandas as pd

OPS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
}


class NumericBinaryFlagTransformer(BaseEstimator, TransformerMixin):
    """
    Create binary flags from numeric columns based on thresholds or threshold functions.
    """

    def __init__(self, flag_config: dict):
        """
        flag_config example:

        {
            "Age": [
                {
                    "flag_name": "age_gt_60",
                    "threshold": 60,
                    "threshold_sign": ">",
                    "drop_original": False
                },
                {
                    "flag_name": "age_above_mean",
                    "threshold_func": np.mean,
                    "threshold_sign": ">=",
                    "drop_original": True
                }
            ]
        }
        """
        self.flag_config = flag_config
        self.thresholds_ = {}  # Will store (setup, computed_threshold, operator_func)

    def fit(self, X, y=None):
        X = X.copy()

        for col, setups in self.flag_config.items():
            self.thresholds_[col] = []

            for setup in setups:
                # Validate operator
                sign = setup.get("threshold_sign", ">")
                if sign not in OPS:
                    raise ValueError(
                        f"Invalid threshold_sign '{sign}'. Must be one of {list(OPS.keys())}"
                    )
                op_func = OPS[sign]

                # Compute threshold
                if "threshold_func" in setup:
                    threshold = setup["threshold_func"](X[col])
                else:
                    threshold = setup["threshold"]

                self.thresholds_[col].append((setup, threshold, op_func))

        return self

    def transform(self, X):
        X = X.copy()

        for col, setups in self.thresholds_.items():
            for setup, threshold, op_func in setups:
                flag_name = setup.get("flag_name", col + "_flag")
                X[flag_name] = op_func(X[col], threshold).astype(int)

                if setup.get("drop_original", False):
                    X = X.drop(columns=col, errors="ignore")

        return X


class CategoricalBinaryFlagTransformer(BaseEstimator, TransformerMixin):
    """
    Creates binary indicator columns from categorical features,
    with optional removal of the original column.
    """

    def __init__(self, flag_config: dict):
        """
        Example config:

        {
            "Geography": [
                {
                    "flag_name": "is_germany",
                    "value": "Germany",
                    "drop_original": True
                }
            ]
        }
        """
        self.flag_config = flag_config
        self.columns_to_drop_ = set()

    def fit(self, X, y=None):
        # Decide what to drop during fit → deterministic pipelines
        for col, setups in self.flag_config.items():
            for setup in setups:
                if setup.get("drop_original", False):
                    self.columns_to_drop_.add(col)

        return self

    def transform(self, X):
        X = X.copy()

        for col, setups in self.flag_config.items():
            for setup in setups:

                X[setup["flag_name"]] = (X[col] == setup["value"]).astype(int)

        if self.columns_to_drop_:
            X = X.drop(columns=list(self.columns_to_drop_), errors="ignore")

        return X


class TopNCategoricalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, config: dict):
        """
        config = {"Surname": 10, "City": 5}
        """
        self.config = config
        self.top_values_ = {}

    def fit(self, X, y=None):
        for col, n in self.config.items():
            self.top_values_[col] = set(X[col].value_counts().nlargest(n).index)
        return self

    def transform(self, X):
        X = X.copy()

        for col, allowed in self.top_values_.items():
            X[col] = X[col].where(X[col].isin(allowed))

        return X


class NumericBinner(BaseEstimator, TransformerMixin):
    def __init__(self, mapping):
        """
        mapping = {
            "EstimatedSalary": {
                "bins": [0, 50000, 120000, float("inf")],
                "labels": ["low", "medium", "high"],
                "new_col": "SalaryScore",
                "drop_original": True
            }
        }
        """
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols_to_drop = []

        for col, cfg in self.mapping.items():

            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

            new_col = cfg.get("new_col", f"{col}_binned")
            drop_original = cfg.get("drop_original", False)

            X[new_col] = pd.cut(
                X[col], bins=cfg["bins"], labels=cfg["labels"], ordered=True
            )

            if drop_original:
                cols_to_drop.append(col)

        if cols_to_drop:
            X = X.drop(columns=cols_to_drop, errors="ignore")

        return X
