from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def compute_all_metrics(y_true, y_pred, y_proba=None):
    res = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            res["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            res["roc_auc"] = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    res["confusion_matrix"] = cm.tolist()
    return res
