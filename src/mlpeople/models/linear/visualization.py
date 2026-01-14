import matplotlib.pyplot as plt
from .model import estimate_linear

import numpy as np
import matplotlib.pyplot as plt


def plot_1d_predictions(
    X,
    y,
    w,
    b=0.0,
    predict_fn=None,
    xlabel="Feature",
    ylabel="Target",
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
