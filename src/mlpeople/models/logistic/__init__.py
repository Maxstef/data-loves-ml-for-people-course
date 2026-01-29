# =========================
# Logistic Regression Module
# =========================

# Data generation
from .data import (
    generate_logistic_regression_data,
    generate_binary_classification_sklearn,
)

# Model functions
from .model import fit_logistic, predict, predict_proba

# Loss & gradient
from .loss import sigmoid, hypothesis, log_loss, compute_gradient

# Metrics
from .metrics import accuracy, precision, recall, f1_score

# Optional: define __all__ for cleaner "from ... import *" usage
__all__ = [
    # Data
    "generate_logistic_regression_data",
    "generate_binary_classification_sklearn",
    # Model
    "fit_logistic",
    "predict",
    "predict_proba",
    # Loss
    "sigmoid",
    "hypothesis",
    "log_loss",
    "compute_gradient",
    # Metrics
    "accuracy",
    "precision",
    "recall",
    "f1_score",
]
