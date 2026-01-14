from typing import Callable, Optional, Literal

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from .calculus import derivative


def plot_function_and_numeric_derivative(
    f: Callable[[np.ndarray], np.ndarray],
    x_values: np.ndarray = np.linspace(0.01, 10.0, 100),
    *,
    title: str = "Function and numeric derivative",
    xlabel: str = "x",
    ylabel: str = "y",
    function_label: Optional[str] = None,
    derivative_label: Optional[str] = None,
    h: float = 1e-5,
    layout: Literal["overlay", "subplots"] = "overlay",
    show: bool = True,
):
    """
    Plot a function and its numeric (finite-difference) derivative.

    Parameters
    ----------
    f : callable
        Function mapping x -> f(x), must support NumPy arrays.
    x_values : np.ndarray
        Points at which to evaluate the function.
    title, xlabel, ylabel : str
        Plot labels.
    function_label : str, optional
        Label for the function curve.
    derivative_label : str, optional
        Label for the derivative curve.
    h : float
        Step size for finite difference derivative.
    layout:
        - "overlay": function and derivative on the same axes
        - "subplots": function and derivative on separate subplots
    show : bool
        Whether to call plt.show().
    """
    y = f(x_values)
    dy = derivative(f, x_values, h=h)

    function_label = "f(x)" if function_label is None else f"f(x) = {function_label}"
    derivative_label = (
        "f'(x)" if derivative_label is None else f"f'(x) = {derivative_label}"
    )

    if layout == "overlay":
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_values, y, label=function_label, color="blue")
        ax.plot(x_values, dy, linestyle="--", label=derivative_label, color="red")

        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    elif layout == "subplots":
        fig, (ax_f, ax_d) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 7))

        ax_f.plot(x_values, y, label=function_label, color="blue")
        ax_f.set_ylabel("f(x)")
        ax_f.legend()
        ax_f.grid(True)

        ax_d.plot(x_values, dy, linestyle="--", label=derivative_label, color="red")
        ax_d.set_xlabel(xlabel)
        ax_d.set_ylabel("f'(x)")
        ax_d.legend()
        ax_d.grid(True)

        ax = (ax_f, ax_d)

    else:
        raise ValueError(f"Unknown layout: {layout}")

    fig.suptitle(title)

    if show:
        plt.show()

    return ax


def plot_function_and_symbolic_derivative(
    f: sp.Expr,
    x: sp.Symbol,
    x_values: np.ndarray = np.linspace(0.01, 10.0, 100),
    *,
    title: str = "Function and symbolic derivative",
    xlabel: str = "x",
    ylabel: str = "y",
    layout: Literal["overlay", "subplots"] = "overlay",
    show: bool = True,
):
    """
    Plot a symbolic function and its exact derivative using SymPy.

    Parameters
    ----------
    f : sympy.Expr
        Symbolic function.
    x : sympy.Symbol
        Differentiation variable.
    x_values : np.ndarray
        Points at which to evaluate the function.
    title, xlabel, ylabel : str
        Plot labels.
    show : bool
        Whether to call plt.show().
    """
    df = sp.diff(f, x)

    f_fn = sp.lambdify(x, f, modules="numpy")
    df_fn = sp.lambdify(x, df, modules="numpy")

    y = f_fn(x_values)
    dy = df_fn(x_values)

    if layout == "overlay":
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_values, y, label=str(f), color="blue")
        ax.plot(x_values, dy, linestyle="--", label=str(df), color="red")

        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    elif layout == "subplots":
        fig, (ax_f, ax_d) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 7))

        ax_f.plot(x_values, y, label=str(f), color="blue")
        ax_f.set_ylabel("f(x)")
        ax_f.legend()
        ax_f.grid(True)

        ax_d.plot(x_values, dy, linestyle="--", label=str(df), color="red")
        ax_d.set_xlabel(xlabel)
        ax_d.set_ylabel("f'(x)")
        ax_d.legend()
        ax_d.grid(True)

        ax = (ax_f, ax_d)

    else:
        raise ValueError(f"Unknown layout: {layout}")

    fig.suptitle(title)

    if show:
        plt.show()

    return ax
