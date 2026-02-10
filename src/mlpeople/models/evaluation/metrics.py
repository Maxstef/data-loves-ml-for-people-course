from sklearn.metrics import roc_curve, auc


def predict_and_score_auc(model, inputs, targets, name="", threshold=0.5, verbose=True):
    """
    Predict target labels and probabilities, then calculate AUROC.

    Args:
        model: fitted classifier with `predict_proba` method
        inputs: array-like or DataFrame of features
        targets: array-like of true binary labels (0 or 1)
        name: optional string to identify dataset (e.g., 'Training'/'Validation')
        threshold: float, probability threshold to convert proba -> binary predictions
        verbose: bool, if True prints AUROC score

    Returns:
        preds: binary predictions (0/1)
        y_pred_proba: predicted probabilities for class 1
        roc_auc: AUROC score
    """
    # Predict probabilities
    y_pred_proba = model.predict_proba(inputs)[:, 1]

    # predict target based on threshold, calculate f1 and confusion matrix
    preds = (y_pred_proba >= threshold).astype(int)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(targets, y_pred_proba, pos_label=1)

    # Compute AUROC
    roc_auc = auc(fpr, tpr)
    if verbose:
        print(f"AUROC for {name}: {roc_auc:.4f}")

    return preds, y_pred_proba, roc_auc
