import numpy as np


# ============================================================
# Loss Base Class (for arbitrary differentiable losses)
# ============================================================
class Loss:
    """
    Base class for differentiable loss functions.
    Subclasses must implement .loss() and .gradient().
    """

    def loss(self, y, F):
        raise NotImplementedError

    def gradient(self, y, F):
        """
        Return dL/dF (NOT negative).
        """
        raise NotImplementedError


# ============================================================
# Numeric Gradient Loss (generic, finite difference)
# ============================================================
class NumericLoss(Loss):
    """
    Computes loss and gradient numerically for arbitrary functions.
    """

    def __init__(self, func, h=1e-5):
        """
        Parameters:
        -----------
        func : callable
            Loss function L(y, F)
        h : float
            Step size for finite difference approximation
        """
        self.func = func
        self.h = h

    def loss(self, y, F):
        return self.func(y, F)

    def gradient(self, y, F):
        # Central difference approximation
        return (self.func(y, F + self.h) - self.func(y, F - self.h)) / (2 * self.h)


# ============================================================
# Squared Loss (analytical)
# ============================================================
class SquaredLoss(Loss):

    def loss(self, y, F):
        return 0.5 * (y - F) ** 2

    def gradient(self, y, F):
        return F - y


# ============================================================
# Logistic Loss (binary classification)
# ============================================================
class LogisticLoss(Loss):
    """
    Binary classification loss. y ∈ {-1, +1}.
    """

    def loss(self, y, F):
        return np.log(1 + np.exp(-y * F))

    def gradient(self, y, F):
        return -y / (1 + np.exp(y * F))
