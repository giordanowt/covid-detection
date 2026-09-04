import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def per_class_report(y_true, y_pred, y_prob, labels):
    indices = list(range(len(labels)))
    matrix = confusion_matrix(y_true, y_pred, labels=indices)
    report = {}
    for i, label in enumerate(labels):
        tp = matrix[i, i]
        fn = matrix[i, :].sum() - tp
        fp = matrix[:, i].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        one_vs_rest = (np.asarray(y_true) == i).astype(int)
        auroc = roc_auc_score(one_vs_rest, y_prob[:, i]) if 0 < one_vs_rest.sum() < len(one_vs_rest) else None
        report[label] = {
            "support": int(tp + fn),
            "sensitivity": _safe_ratio(tp, tp + fn),
            "specificity": _safe_ratio(tn, tn + fp),
            "precision": _safe_ratio(tp, tp + fp),
            "auroc": auroc,
        }
    return report


def evaluate_predictions(y_true, y_prob, labels):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(axis=1)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    return {
        "labels": list(labels),
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class_report(y_true, y_pred, y_prob, labels),
    }
