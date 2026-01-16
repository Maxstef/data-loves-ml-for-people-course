import sympy as sp


def evaluate_function(f, var, value, numeric=True):
    """
    Evaluate a SymPy function for a given argument.

    Parameters:
    - f: sympy expression (e.g., x**2, sin(x))
    - var: sympy symbol (e.g., x)
    - value: value to substitute (number or sympy expression)
    - numeric: if True, return numerical result; if False, return symbolic result

    Returns:
    - evaluated function result
    """
    result = f.subs(var, value)

    if numeric:
        return sp.N(result)

    return result
