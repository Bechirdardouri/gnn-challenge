from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def macro_f1(y_true, y_pred) -> float:
    y_true_arr = np.asarray(y_true, dtype=float).astype(int)
    y_pred_arr = np.rint(np.asarray(y_pred, dtype=float)).astype(int)
    return float(f1_score(y_true_arr, y_pred_arr, average="macro"))


def binary_auc(y_true, y_pred) -> float:
    return float(roc_auc_score(y_true, y_pred))
