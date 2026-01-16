def is_linear_numeric(f, x_values=None, y_values=None, a_values=None, tol=1e-9):
    """
    Numerically check whether a pure Python function f(x) is LINEAR.

    A function is linear if it satisfies BOTH:
      1) Additivity:     f(x + y) = f(x) + f(y)
      2) Homogeneity:    f(a * x) = a * f(x)

    Since f is an arbitrary Python function, we cannot prove this symbolically.
    Instead, we test these properties on a finite set of numeric values.

    Parameters:
    - f        : callable, e.g. def f(x): return 3*x
    - x_values : list of x values to test (inputs)
    - y_values : list of y values to test (for additivity)
    - a_values : list of scalar multipliers to test (for homogeneity)
    - tol      : numerical tolerance for floating-point comparisons

    Returns:
    - True  → function behaves linearly on tested values
    - False → linearity violated for at least one test
    """
    if x_values is None:
        x_values = [0, 1, 2, -1, -2]
    if y_values is None:
        y_values = [0, 1, 2, -1, -2]
    if a_values is None:
        a_values = [0, 1, 2, -1, -2]

    # Additivity check
    for x in x_values:
        for y in y_values:
            if abs(f(x + y) - (f(x) + f(y))) > tol:
                return False

    # Homogeneity check
    for x in x_values:
        for a in a_values:
            if abs(f(a * x) - a * f(x)) > tol:
                return False

    return True


def is_affine_numeric(f, x_values=None, y_values=None, tol=1e-9):
    """
    Numerically check whether a pure Python function f(x) is AFFINE.

    A function is affine if it can be written as:
        f(x) = k*x + b

    Key idea:
    - Affine functions become linear after subtracting f(0)
    - Define g(x) = f(x) - f(0)
    - f is affine ⇔ g is linear

    Parameters:
    - f        : callable, e.g. def f(x): return 3*x + 2
    - x_values : list of x values to test
    - tol      : numerical tolerance

    Returns:
    - True  → function behaves affinely on tested values
    - False → affine structure violated
    """
    if x_values is None:
        x_values = [0, 1, 2, -1, -2]

    # Compute f(0)
    f0 = f(0)

    # Check if f(x) - f(0) is linear
    def g(x):
        return f(x) - f0

    return is_linear_numeric(g, x_values=x_values)
