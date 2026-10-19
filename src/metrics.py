"""Metric helpers and the single definition of the support-tier thresholds."""
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)

# Support tiers are defined on the final grade G3 (0-20 scale).
HIGH_SUPPORT_BELOW = 10   # G3 < 10   -> High Support  (below the pass mark)
MEDIUM_SUPPORT_BELOW = 14  # 10 <= G3 < 14 -> Medium Support; G3 >= 14 -> Low Support
TIER_LABELS = ["High Support", "Low Support", "Medium Support"]  # sorted order used by models


def support_level_from_grade(grade: float) -> str:
    """Map a 0-20 grade to a support tier."""
    if grade < HIGH_SUPPORT_BELOW:
        return "High Support"
    if grade < MEDIUM_SUPPORT_BELOW:
        return "Medium Support"
    return "Low Support"


def support_level_from_score_pct(score_pct: float) -> str:
    """Map a 0-100 percentage score (grade x 5) to a support tier."""
    return support_level_from_grade(score_pct / 5.0)


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """MAE, RMSE and R2 (unrounded)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred, labels: List[str],
                           proba: Optional[np.ndarray] = None) -> Dict:
    """Accuracy, macro precision/recall/F1, per-class recall, confusion matrix, ROC-AUC (OvR)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision_Macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "Recall_Macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "F1_Macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "Recall_Per_Class": {lab: float(r) for lab, r in zip(labels, per_class_recall)},
        "Confusion_Matrix": cm.tolist(),
        "ROC_AUC": None,
    }
    if proba is not None and proba.shape[1] == len(labels):
        try:
            y_bin = np.array([[1 if t == lab else 0 for lab in labels] for t in y_true])
            out["ROC_AUC"] = float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro"))
        except ValueError:
            out["ROC_AUC"] = None
    return out
