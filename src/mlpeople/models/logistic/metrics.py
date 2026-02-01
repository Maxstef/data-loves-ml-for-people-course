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


def confusion_matrix(y_true, y_pred, normalize=None, decimals=2):
    """
    Compute confusion matrix for binary classification.

    Parameters
    ----------
    y_true : array-like
        True labels (0 or 1)
    y_pred : array-like
        Predicted labels (0 or 1)
    normalize : {None, "true", "pred", "all"}, default=None
        Normalization mode:
        - None  : return raw counts
        - "true": normalize over true labels (rows)
        - "pred": normalize over predicted labels (columns)
        - "all" : normalize over all samples
    decimals : int or None, default=None
        Number of decimals to round the normalized values to.
        Ignored if `normalize=None`.

    Returns
    -------
    dict
        Dictionary with TP, FP, FN, TN (possibly normalized)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    cm = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}

    if normalize is None:
        return cm

    eps = 1e-15

    if normalize == "true":
        pos = tp + fn
        neg = fp + tn
        cm_norm = {
            "TP": tp / (pos + eps),
            "FN": fn / (pos + eps),
            "FP": fp / (neg + eps),
            "TN": tn / (neg + eps),
        }

    elif normalize == "pred":
        pos = tp + fp
        neg = fn + tn
        cm_norm = {
            "TP": tp / (pos + eps),
            "FP": fp / (pos + eps),
            "FN": fn / (neg + eps),
            "TN": tn / (neg + eps),
        }

    elif normalize == "all":
        total = tp + fp + fn + tn
        cm_norm = {k: v / (total + eps) for k, v in cm.items()}

    else:
        raise ValueError("normalize must be one of {None, 'true', 'pred', 'all'}")

    # round normalized values if decimals is specified
    if decimals is not None:
        cm_norm = {k: round(v, decimals) for k, v in cm_norm.items()}

    return cm_norm
