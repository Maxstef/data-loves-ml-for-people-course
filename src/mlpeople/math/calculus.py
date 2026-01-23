def derivative_function(f, h=1e-5, central=True):
    """
    Create and return a function that computes the numerical derivative of f.

    Parameters
    ----------
    f : callable
        Function of a single variable f(x).
    h : float, optional
        Small step size used for finite difference approximation.
        Default is 1e-5.
    central : bool, optional
        If True, use central difference.
        If False, use forward difference.

    Returns
    -------
    df : callable
        A function df(x) that approximates f'(x).
    """

    if central:
        # Central difference approximation (more accurate)
        def df(x):
            return (f(x + h) - f(x - h)) / (2 * h)

    else:
        # Forward difference approximation
        def df(x):
            return (f(x + h) - f(x)) / h

    return df


def derivative_point(f, x, h=1e-5, central=True):
    """
    Compute the numerical derivative of f at a single point x.

    Parameters
    ----------
    f : callable
        Function of a single variable f(x).
    x : float
        Point at which the derivative is evaluated.
    h : float, optional
        Small step size used for finite difference approximation.
        Default is 1e-5.
    central : bool, optional
        If True, use central difference.
        If False, use forward difference.

    Returns
    -------
    float
        Approximation of f'(x).
    """

    df = derivative_function(f, h=h, central=central)
    return df(x)


import numpy as np


def gradient_function(f, h=1e-5, central=True):
    """
    Create and return a function that computes the numerical derivative or gradient of f.

    Supports:
    - Scalar input  -> returns scalar derivative
    - Vector input  -> returns gradient vector

    Parameters
    ----------
    f : callable
        Function f(x) -> scalar OR f(w) -> scalar.
    h : float, optional
        Step size for finite difference approximation.
    central : bool, optional
        If True, use central differences (more accurate).
        If False, use forward differences.

    Returns
    -------
    grad_f : callable
        Function that computes derivative or gradient at a point.
    """

    def grad_f(x):
        x = np.asarray(x, dtype=float)

        # ---------- Scalar case ----------
        if x.ndim == 0:
            if central:
                return (f(x + h) - f(x - h)) / (2 * h)
            else:
                return (f(x + h) - f(x)) / h

        # ---------- Vector case ----------
        grad = np.zeros_like(x)

        for i in range(len(x)):
            e = np.zeros_like(x)
            e[i] = h

            if central:
                grad[i] = (f(x + e) - f(x - e)) / (2 * h)
            else:
                grad[i] = (f(x + e) - f(x)) / h

        return grad

    return grad_f


def gradient_point(f, x, h=1e-5, central=True):
    """
    Compute the numerical gradient of f at a specific point x.
    supports scalar and vector x
    """
    return gradient_function(f, h=h, central=central)(x)
