import numpy as np


def accuracy(y_true, y_pred):
    """
    Fraction of correct predictions.
    """
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred, eps: float = 1e-15):
    """
    Precision = TP / (TP + FP)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    return tp / (tp + fp + eps)


def recall(y_true, y_pred, eps: float = 1e-15):
    """
    Recall = TP / (TP + FN)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp / (tp + fn + eps)


def f1_score(y_true, y_pred, eps: float = 1e-15):
    """
    F1 = harmonic mean of precision and recall.
    """
    p = precision(y_true, y_pred, eps)
    r = recall(y_true, y_pred, eps)

    return 2 * p * r / (p + r + eps)
