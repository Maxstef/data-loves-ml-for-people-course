import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# -------------------------
# Metric helpers
# -------------------------


def compute_roc(pred_probs, targets, pos_label=None):
    fpr, tpr, thresholds = roc_curve(targets, pred_probs, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


def compute_pr(pred_probs, targets, pos_label=None):
    precision, recall, thresholds = precision_recall_curve(
        targets, pred_probs, pos_label=pos_label
    )
    pr_auc = auc(recall, precision)
    return precision, recall, thresholds, pr_auc


def classify_outcomes(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    labels = np.empty(len(y_true), dtype=object)
    labels[(y_true == 1) & (y_pred == 1)] = "TP"
    labels[(y_true == 0) & (y_pred == 1)] = "FP"
    labels[(y_true == 1) & (y_pred == 0)] = "FN"
    labels[(y_true == 0) & (y_pred == 0)] = "TN"
    return labels


def fbeta_score_at_threshold(y_true, y_proba, threshold=0.5, beta=1.0, eps=1e-15):
    y_pred = (y_proba >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    beta2 = beta**2
    fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)
    return fbeta, precision, recall


def find_best_threshold(y_true, y_proba, beta=1.0, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 501)
    best = {"threshold": None, "fbeta": -np.inf, "precision": None, "recall": None}
    for t in thresholds:
        fbeta, p, r = fbeta_score_at_threshold(y_true, y_proba, t, beta)
        if fbeta > best["fbeta"]:
            best.update(threshold=t, fbeta=fbeta, precision=p, recall=r)
    return best


# -------------------------
# Unified plotting function
# -------------------------


def plot_roc(
    pred_probs,
    targets,
    name="",
    pos_label=None,
    show_pr_curve=False,
    show_probabilities=False,
    threshold=None,
    beta=1.0,
):
    """
    Plot ROC curve with optional Precision–Recall curve and probability row.
    Marks chosen or best threshold on PR curve and probability row.

    Parameters
    ----------
    pred_probs : array-like
        Predicted probabilities for positive class
    targets : array-like
        True binary labels
    show_pr_curve : bool
        Whether to plot Precision–Recall curve
    show_probabilities : bool
        Whether to plot probability row colored by TP/FP/TN/FN
    threshold : float or None
        Threshold to display. If None, auto-select best F-beta
    beta : float
        Beta for F-beta optimization if threshold is None
    """

    # --- ROC / PR metrics ---
    fpr, tpr, _, roc_auc = compute_roc(pred_probs, targets, pos_label)
    if show_pr_curve:
        precision, recall, _, pr_auc = compute_pr(pred_probs, targets, pos_label)

    # --- Determine threshold ---
    if threshold is None:
        best = find_best_threshold(targets, pred_probs, beta=beta)
        threshold = best["threshold"]
        threshold_precision = best["precision"]
        threshold_recall = best["recall"]
        threshold_fbeta = best["fbeta"]
    else:
        threshold_fbeta, threshold_precision, threshold_recall = (
            fbeta_score_at_threshold(
                targets, pred_probs, threshold=threshold, beta=beta
            )
        )

    # --- Layout ---
    if show_pr_curve and show_probabilities:
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 2, height_ratios=[4, 1])
        ax_roc = fig.add_subplot(gs[0, 0])
        ax_pr = fig.add_subplot(gs[0, 1])
        ax_prob = fig.add_subplot(gs[1, :])
    elif show_pr_curve:
        fig = plt.figure(figsize=(10, 4))
        gs = fig.add_gridspec(1, 2)
        ax_roc = fig.add_subplot(gs[0, 0])
        ax_pr = fig.add_subplot(gs[0, 1])
        ax_prob = None
    elif show_probabilities:
        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
        ax_roc = fig.add_subplot(gs[0, 0])
        ax_pr = None
        ax_prob = fig.add_subplot(gs[1, 0])
    else:
        fig, ax_roc = plt.subplots(figsize=(6, 6))
        ax_pr = None
        ax_prob = None

    # --- ROC ---
    ax_roc.plot(fpr, tpr, lw=2, label=f"ROC (AUROC = {roc_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", lw=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC {f'— {name}' if name else ''}")
    ax_roc.legend()
    ax_roc.grid(alpha=0.3)
    ax_roc.set_aspect("equal", adjustable="box")

    # --- Precision–Recall ---
    if show_pr_curve:
        ax_pr.plot(recall, precision, lw=2, label=f"PR (AUPRC = {pr_auc:.3f})")
        baseline = np.mean(targets)
        ax_pr.hlines(baseline, 0, 1, linestyle="--", lw=1, label="Baseline")

        # --- mark threshold ---
        ax_pr.scatter(
            threshold_recall,
            threshold_precision,
            s=100,
            color="black",
            zorder=6,
            label=f"Threshold = {threshold:.2f}"
            + (f", F{beta}={threshold_fbeta:.2f}" if threshold_fbeta else ""),
        )
        ax_pr.annotate(
            f"({threshold_recall:.2f}, {threshold_precision:.2f})",
            (threshold_recall, threshold_precision),
            textcoords="offset points",
            xytext=(5, 5),
        )

        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision–Recall")
        ax_pr.legend()
        ax_pr.grid(alpha=0.3)
        ax_pr.set_aspect("equal", adjustable="box")

    # --- Probability row ---
    if show_probabilities:
        outcomes = classify_outcomes(targets, pred_probs, threshold)
        colors = {"TP": "green", "TN": "blue", "FP": "red", "FN": "orange"}

        for outcome, color in colors.items():
            mask = outcomes == outcome
            ax_prob.scatter(
                pred_probs[mask],
                np.zeros(np.sum(mask)),
                c=color,
                s=40,
                alpha=0.5,
                label=outcome,
            )

        # mark threshold
        ax_prob.axvline(threshold, linestyle="--", color="black", lw=1)
        ax_prob.set_xlim(0, 1)
        ax_prob.set_yticks([])
        ax_prob.set_xlabel("Predicted Probability")
        ax_prob.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.35))

    plt.tight_layout()
    plt.show()

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc if show_pr_curve else None,
        "best_threshold": threshold,
        "threshold_precision": threshold_precision,
        "threshold_recall": threshold_recall,
        "threshold_fbeta": threshold_fbeta,
    }
