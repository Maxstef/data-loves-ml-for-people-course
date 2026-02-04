import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from .model import estimate_linear, predict, fit_ols
from .metrics import r2_score, rmse, mae


def plot_1d_predictions(
    X,
    y,
    w,
    b=0.0,
    predict_fn=None,
    xlabel="Feature",
    ylabel="Target",
    title=None,
    ax=None,
):
    """
    Plot actual vs predicted values for a 1D feature.

    Parameters
    ----------
    X : array-like (n_samples,)
        Feature values
    y : array-like (n_samples,)
        Target values
    w : scalar
        Model weight
    b : float
        Bias term
    predict_fn : callable, optional
        Prediction function (defaults to linear_predict)
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 1:
        raise ValueError("plot_1d_predictions expects a single feature (1D input)")

    if predict_fn is None:
        from .model import linear_predict

        predict_fn = linear_predict

    predictions = predict_fn(X, w, b)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(X, predictions, "r", alpha=0.9)
    ax.scatter(X, y, s=8, alpha=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(["Estimate", "Actual"])

    if title:
        ax.set_title(title)

    return ax


def try_parameters(
    df,
    w,
    b=0.0,
    x=None,
    y="target",
):
    """
    Visualize actual vs predicted values for a single feature in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing features and target.
    w : float
        Weight for the feature.
    b : float, default 0.0
        Bias/intercept term.
    x : str, default "age"
        Column name for the feature.
    y : str, default "charges"
        Column name for the target.

    Returns
    -------
    matplotlib.axes.Axes
        Axes object with scatter and prediction line plotted.
    """
    return plot_1d_predictions(
        X=df[x],
        y=df[y],
        w=w,
        b=b,
        xlabel=x.capitalize(),
        ylabel=y.capitalize(),
    )


def plot_features_vs_target(
    X: np.ndarray, y: np.ndarray, beta_hat: np.ndarray | None = None
):
    """
    Plot each feature Xi against the target y.

    Optionally overlay a fitted regression line if beta_hat is provided.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix (without intercept column).
    y : ndarray of shape (n_samples,)
        Target vector.
    beta_hat : ndarray of shape (n_features + 1,), optional
        Estimated coefficients including intercept.
        If provided, a regression line for each feature will be drawn
        using the coefficient corresponding to that feature.
    """

    n_features = X.shape[1]

    for i in range(n_features):
        plt.figure(figsize=(6, 4))
        plt.scatter(X[:, i], y, alpha=0.7, label="Data")

        # Draw regression line if coefficients are provided
        if beta_hat is not None:
            intercept = beta_hat[0]
            coef_i = beta_hat[i + 1]  # +1 because beta_hat[0] is intercept
            x_vals = np.array([X[:, i].min(), X[:, i].max()])
            y_vals = intercept + coef_i * x_vals
            plt.plot(x_vals, y_vals, color="red", label="OLS fit", lw=2)

        plt.xlabel(f"X{i}")
        plt.ylabel("y")
        plt.title(f"Feature X{i} vs Target y")
        if beta_hat is not None:
            plt.legend()
        plt.tight_layout()
        plt.show()


def plot_ols_predictions_with_error(
    X, y, beta_hat, fit_intercept=True, with_error="MAE", show_error_lines=False
):
    """
    Plot target y vs feature X (1D) with OLS line, predicted y, optional error bands,
    and optional lines from each actual y to predicted y with error labels.

    Parameters
    ----------
    X : ndarray of shape (n_samples, 1)
        Feature matrix
    y : ndarray of shape (n_samples,)
        True target
    beta_hat : ndarray of shape (2,)
        Estimated coefficients including intercept
    fit_intercept : bool
        Whether beta_hat includes intercept
    with_error : str or None, default "MAE"
        Type of error band to show: "MAE", "RMSE", or None
    show_error_lines : bool, default False
        If True, draw vertical lines from each y to y_pred with error labels.
    """
    if X.shape[1] != 1:
        raise ValueError("This plot only supports a single feature.")

    # Predicted values
    y_pred = predict(X, beta_hat, fit_intercept=fit_intercept)

    # Compute error metric
    error_val = None
    error_label = ""
    if with_error is not None:
        if with_error.upper() == "MAE":
            error_val = mae(y, y_pred)
            error_label = f"MAE ±{error_val:.2f}"
        elif with_error.upper() == "RMSE":
            error_val = rmse(y, y_pred)
            error_label = f"RMSE ±{error_val:.2f}"
        else:
            raise ValueError("with_error must be 'MAE', 'RMSE', or None")

    # Scatter plot of true target
    plt.scatter(X[:, 0], y, color="blue", label="Target y")

    # OLS line (dotted)
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = predict(x_range, beta_hat, fit_intercept=fit_intercept)
    plt.plot(x_range, y_line, color="red", linestyle=":", label="OLS line")

    # Predicted points
    plt.scatter(X[:, 0], y_pred, color="green", marker="x", label="Predicted y")

    # Error band (MAE or RMSE)
    if error_val is not None:
        plt.fill_between(
            x_range[:, 0],
            y_line - error_val,
            y_line + error_val,
            color="orange",
            alpha=0.1,
            label=error_label,
        )

    # Optional error lines from actual y to predicted y
    if show_error_lines:
        for xi, yi, ypi in zip(X[:, 0], y, y_pred):
            # Draw vertical line
            plt.plot([xi, xi], [yi, ypi], color="gray", linestyle="--", linewidth=0.8)
            # Show error label above the line
            err = abs(yi - ypi)
            plt.text(
                xi,
                max(yi, ypi) + 0.02 * (y.max() - y.min()),
                f"{err:.2f}",
                fontsize=8,
                color="gray",
                ha="center",
            )

    # Title includes R²
    r2_val = r2_score(y, y_pred)
    plt.title(f"OLS Fit | R² = {r2_val:.3f}")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_mae_rmse_rmsle(y_true, y_pred):
    """
    Visualize MAE, RMSE, and RMSLE intuitions using aligned subplots.

    Each subplot shows per-point contribution to the metric.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("RMSLE requires non-negative targets.")

    # Per-point contributions
    abs_errors = np.abs(y_true - y_pred)
    sq_errors = (y_true - y_pred) ** 2
    log_errors = np.abs(np.log1p(y_true) - np.log1p(y_pred))

    # Aggregate metrics
    mae_val = abs_errors.mean()
    rmse_val = np.sqrt(sq_errors.mean())
    rmsle_val = np.sqrt((log_errors**2).mean())

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    # MAE
    axes[0].scatter(y_true, abs_errors, color="blue")
    axes[0].set_title(f"MAE intuition | MAE = {mae_val:.2f}")
    axes[0].set_ylabel("|y - ŷ|")
    axes[0].grid(True)

    # RMSE
    axes[1].scatter(y_true, sq_errors, color="orange")
    axes[1].set_title(f"RMSE intuition | RMSE = {rmse_val:.2f}")
    axes[1].set_ylabel("(y - ŷ)²")
    axes[1].grid(True)

    # RMSLE
    axes[2].scatter(y_true, log_errors, color="green")
    axes[2].set_title(f"RMSLE intuition | RMSLE = {rmsle_val:.3f}")
    axes[2].set_ylabel("|log(y+1) - log(ŷ+1)|")
    axes[2].set_xlabel("True y")
    axes[2].grid(True)

    axes[2].set_xscale("log")

    plt.tight_layout()
    plt.show()


def plot_residuals(
    X,
    y,
    y_pred,
    plot_residuals_vs_features: bool = False,
    feature_names=None,
):
    """
    Plot standard residual diagnostics for linear regression.

    Diagnostics included:
    1. Residuals vs Predicted
    2. Residuals vs Features (optional)
    3. Histogram of residuals
    4. Q–Q plot (normality)
    5. Scale–Location plot (variance check)

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        True target values
    y_pred : ndarray of shape (n_samples,)
        Predicted target values
    plot_residuals_vs_features : bool, default False
        If True, plot residuals vs each feature X_i
    feature_names : list of str or None
        Optional feature names for labeling plots
    """
    residuals = y - y_pred
    n_samples, n_features = X.shape

    # -------------------------------
    # 1️⃣ Residuals vs Predicted
    # -------------------------------
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals)
    plt.axhline(0, linestyle="--", color="red")
    plt.xlabel("Predicted y")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    plt.grid(True)
    plt.show()

    # --------------------------------
    # 2️⃣ Residuals vs Features (optional)
    # --------------------------------
    if plot_residuals_vs_features:
        for i in range(n_features):
            plt.figure(figsize=(6, 4))
            plt.scatter(X[:, i], residuals)
            plt.axhline(0, linestyle="--", color="red")

            label = feature_names[i] if feature_names is not None else f"X[{i}]"

            plt.xlabel(label)
            plt.ylabel("Residuals")
            plt.title(f"Residuals vs {label}")
            plt.grid(True)
            plt.show()

    # -------------------------------
    # 3️⃣ Histogram of residuals
    # -------------------------------
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=20, edgecolor="black")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residuals")
    plt.grid(True)
    plt.show()

    # -------------------------------
    # 4️⃣ Q–Q plot (Normality)
    # -------------------------------
    plt.figure(figsize=(6, 4))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q–Q Plot of Residuals")
    plt.grid(True)
    plt.show()

    # --------------------------------
    # 5️⃣ Scale–Location plot
    # --------------------------------
    standardized_residuals = residuals / np.std(residuals)
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, np.sqrt(np.abs(standardized_residuals)))
    plt.xlabel("Predicted y")
    plt.ylabel("√|Standardized residuals|")
    plt.title("Scale–Location Plot")
    plt.grid(True)
    plt.show()


def plot_polynomial_fit_1d(xs: np.ndarray, ys: np.ndarray, degree: int = 2):
    """
    Fit a polynomial regression model of a chosen degree and visualize predictions.

    The function expands the feature matrix with powers of x:
        degree=1 → x
        degree=2 → x, x²
        degree=3 → x, x², x³
        ...

    Parameters
    ----------
    xs : np.ndarray of shape (n_samples, 1)
        Input feature column.

    ys : np.ndarray
        Target values.

    degree : int, default=2
        Polynomial degree. Must be >= 1.

    Returns
    -------
    beta_ols : np.ndarray
        Learned regression coefficients.
    """

    if degree < 1:
        raise ValueError("degree must be >= 1")

    # Ensure xs is 2D
    if xs.ndim == 1:
        xs = xs.reshape(-1, 1)

    # Build polynomial feature matrix: [x, x², ..., x^degree]
    xs_poly = np.hstack([xs**i for i in range(1, degree + 1)])

    beta_ols = fit_ols(xs_poly, ys, fit_intercept=True)
    # print(f'beta_ols: {beta_ols}')

    # -------- Smooth curve for prettier plotting --------
    x_dense = np.linspace(xs.min(), xs.max(), 300).reshape(-1, 1)
    x_dense_poly = np.hstack([x_dense**i for i in range(1, degree + 1)])

    y_dense_preds = predict(x_dense_poly, beta_ols, fit_intercept=True)

    plt.figure(figsize=(8, 5))

    plt.scatter(xs, ys, label="Data", alpha=0.6)
    plt.plot(
        x_dense, y_dense_preds, label=f"Polynomial Fit (degree={degree})", color="r"
    )

    plt.title(f"Polynomial Regression (degree={degree})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    return beta_ols
