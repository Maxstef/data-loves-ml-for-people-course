from sklearn.tree import DecisionTreeClassifier
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def evaluate_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dynamic_params: dict,
    static_params: dict | None = None,
) -> dict:
    """
    Fit a DecisionTreeClassifier with given parameters and compute training and validation errors.

    Args:
        X_train (pd.DataFrame): Training input features.
        y_train (pd.Series): Training target values.
        X_test (pd.DataFrame): Validation/test input features.
        y_test (pd.Series): Validation/test target values.
        dynamic_params (dict): Parameters to vary in experiments (logged in results).
        static_params (dict | None): Fixed parameters for the tree.

    Returns:
        dict: Dictionary containing:
            - "Training Error": float
            - "Validation Error": float
            - keys from dynamic_params and their values
    """
    # Merge params safely
    params = {}
    if static_params:
        params.update(static_params)
    if dynamic_params:
        params.update(dynamic_params)

    model = DecisionTreeClassifier(random_state=42, **params)
    model.fit(X_train, y_train)

    train_error = 1 - model.score(X_train, y_train)
    test_error = 1 - model.score(X_test, y_test)

    result = {
        "Training Error": train_error,
        "Validation Error": test_error,
    }

    # Only log dynamic params
    result.update(dynamic_params)

    return result


def plot_errors(errors_df: pd.DataFrame, params: str | list[str]):
    """
    Plot training and validation errors for 1 or 2 hyperparameters.

    1 parameter → line plot with best validation error highlighted.
    2 parameters → heatmap with best point highlighted.

    Args:
        errors_df (pd.DataFrame): DataFrame containing columns "Training Error", "Validation Error",
            and the parameter(s) to visualize.
        params (str | list[str]): Name(s) of 1 or 2 parameter columns to plot.

    Raises:
        ValueError: If params contains more than 2 parameters.
    """

    if isinstance(params, str):
        params = [params]

    # -------- 1 PARAMETER (Line Plot) --------
    if len(params) == 1:
        param = params[0]
        df = errors_df.sort_values(param)

        x = df[param]
        train_err = df["Training Error"]
        val_err = df["Validation Error"]

        best_idx = val_err.idxmin()
        best_x = df.loc[best_idx, param]
        best_val = df.loc[best_idx, "Validation Error"]

        plt.figure(figsize=(8, 5))

        plt.plot(x, train_err, marker="o")
        plt.plot(x, val_err, marker="o")

        plt.scatter(best_x, best_val, s=120)
        plt.axvline(best_x, linestyle="--")

        plt.xticks(x)

        plt.xlabel(param)
        plt.ylabel("Error")
        plt.title(f"Training vs Validation Error vs {param}")
        plt.legend(["Training Error", "Validation Error", "Best Validation"])
        plt.grid(True)
        plt.show()

        print(f"Best {param}: {best_x}")
        print(f"Lowest Validation Error: {best_val:.4f}")

    # -------- 2 PARAMETERS (Heatmap) --------
    elif len(params) == 2:
        p1, p2 = params

        pivot = errors_df.pivot(index=p1, columns=p2, values="Validation Error")

        plt.figure(figsize=(8, 6))

        plt.imshow(pivot, aspect="auto")
        plt.colorbar(label="Validation Error")

        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)

        plt.xlabel(p2)
        plt.ylabel(p1)
        plt.title("Validation Error Heatmap")

        # Highlight best point
        best_idx = errors_df["Validation Error"].idxmin()
        best_p1 = errors_df.loc[best_idx, p1]
        best_p2 = errors_df.loc[best_idx, p2]

        y_pos = list(pivot.index).index(best_p1)
        x_pos = list(pivot.columns).index(best_p2)

        plt.scatter(x_pos, y_pos, s=150)

        plt.show()

        print(f"Best {p1}: {best_p1}")
        print(f"Best {p2}: {best_p2}")
        print(
            f"Lowest Validation Error: {errors_df.loc[best_idx, 'Validation Error']:.4f}"
        )

    else:
        raise ValueError("Only 1 or 2 parameters are supported.")


def run_decision_tree_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param_grid: dict,
    static_params: dict | None = None,
    plot_params: list[str] | None = None,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Run a grid search over dynamic hyperparameters for a DecisionTreeClassifier,
    compute training and validation errors, and optionally plot results.

    Args:
        X_train (pd.DataFrame): Training input features.
        y_train (pd.Series): Training target values.
        X_test (pd.DataFrame): Test/validation input features.
        y_test (pd.Series): Test/validation target values.
        param_grid (dict): Dictionary of parameter names and lists/ranges to explore.
            Example: {"max_depth": range(1, 11), "min_samples_split": [2, 5, 10]}
        static_params (dict | None): Fixed parameters applied to all models.
        plot_params (list[str] | None): Parameter(s) to visualize (1 or 2 names).
        plot (bool): Whether to plot results.

    Returns:
        pd.DataFrame: DataFrame containing all parameter combinations with
        "Training Error" and "Validation Error".
    """

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    results = []

    for combination in product(*param_values):
        dynamic_params = dict(zip(param_names, combination))

        result = evaluate_decision_tree(
            X_train,
            y_train,
            X_test,
            y_test,
            dynamic_params=dynamic_params,
            static_params=static_params,
        )

        results.append(result)

    errors_df = pd.DataFrame(results)

    # -------- Optional Plotting --------
    if plot and plot_params is not None:
        if len(plot_params) not in [1, 2]:
            raise ValueError("plot_params must contain 1 or 2 parameters.")
        plot_errors(errors_df, plot_params)

    return errors_df
