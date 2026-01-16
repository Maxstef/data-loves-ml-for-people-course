import sympy as sp


def is_additive(f, var):
    """
    Check whether a SymPy function satisfies the additivity rule:
    f(x + y) = f(x) + f(y)

    Parameters:
    - f: sympy expression in one variable (e.g., x**2, x, sin(x))
    - var: sympy symbol (e.g., x)

    Returns:
    - True if additive, False otherwise
    """
    x = var
    y = sp.symbols("y", real=True)

    lhs = f.subs(x, x + y)
    rhs = f.subs(x, x) + f.subs(x, y)

    difference = sp.simplify(lhs - rhs)

    return difference == 0


def is_homogeneous(f, var):
    """
    Check homogeneity:
    f(a*x) = a*f(x)
    """
    x = var
    a = sp.symbols("a", real=True)

    lhs = f.subs(x, a * x)
    rhs = a * f.subs(x, x)

    difference = sp.simplify(lhs - rhs)

    return difference == 0


def is_linear(f, var):
    """
    Check linearity:
    additive + homogeneous
    """
    return is_additive(f, var) and is_homogeneous(f, var)


def is_affine(f, var):
    """
    Check if a function is affine: f(x) = k*x + b
    Allows constant offset.
    """
    x = var
    y, a = sp.symbols("y a", real=True)

    # Check if the difference f(x+y) - f(x) - f(y) is constant
    lhs = f.subs(x, x + y)
    rhs = f.subs(x, x) + f.subs(x, y)
    difference = sp.simplify(lhs - rhs)

    # If difference does not depend on x or y, it's affine
    # difference must be constant (no x or y)
    return difference.free_symbols == set()
