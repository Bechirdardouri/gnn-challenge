from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from .metrics import binary_auc, macro_f1
except ImportError:  # pragma: no cover - script execution path
    from metrics import binary_auc, macro_f1


SUBMISSION_ID_COLUMNS = ("node_id", "id", "filename")
SUBMISSION_PRED_COLUMNS = ("target", "prediction", "y_pred")
LABEL_ID_COLUMNS = ("node_id", "id", "filename")
LABEL_TARGET_COLUMNS = ("target", "label", "y_true", "prediction")


def _pick_column(df: pd.DataFrame, candidates: Iterable[str], kind: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"Missing {kind} column. Supported: {', '.join(candidates)}")


def _is_integer_like(series: pd.Series, tol: float = 1e-9) -> bool:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.isnan(arr).any():
        return False
    rounded = np.round(arr)
    return bool(np.all(np.abs(arr - rounded) <= tol))


def _coerce_id_column(df: pd.DataFrame, id_col: str) -> pd.Series:
    raw = df[id_col]
    if pd.api.types.is_numeric_dtype(raw):
        # Normalize 1 and 1.0 to the same canonical id string.
        return pd.to_numeric(raw, errors="coerce").astype("Int64").astype(str)
    return raw.astype(str)


def normalize_submission_columns(df: pd.DataFrame) -> pd.DataFrame:
    id_col = _pick_column(df, SUBMISSION_ID_COLUMNS, "submission id")
    pred_col = _pick_column(df, SUBMISSION_PRED_COLUMNS, "prediction")

    out = df[[id_col, pred_col]].rename(columns={id_col: "id", pred_col: "pred"}).copy()
    out["id"] = _coerce_id_column(out, "id")
    out["pred"] = pd.to_numeric(out["pred"], errors="coerce")
    return out


def normalize_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    id_col = _pick_column(df, LABEL_ID_COLUMNS, "label id")
    target_col = _pick_column(df, LABEL_TARGET_COLUMNS, "label target")

    out = df[[id_col, target_col]].rename(columns={id_col: "id", target_col: "true"}).copy()
    out["id"] = _coerce_id_column(out, "id")
    out["true"] = pd.to_numeric(out["true"], errors="coerce")
    return out


def normalize_test_nodes(df: pd.DataFrame) -> pd.DataFrame:
    id_col = _pick_column(df, SUBMISSION_ID_COLUMNS, "test-node id")
    out = df[[id_col]].rename(columns={id_col: "id"}).copy()
    out["id"] = _coerce_id_column(out, "id")
    return out


def infer_prediction_mode(preds: pd.DataFrame) -> str:
    if _is_integer_like(preds["pred"]):
        return "class_label"
    return "probability"


def read_submission_csv(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return normalize_submission_columns(raw)


def read_label_csv(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return normalize_label_columns(raw)


def validate_submission_df(preds: pd.DataFrame, test_nodes: pd.DataFrame) -> None:
    if preds["id"].duplicated().any():
        raise ValueError("Duplicate IDs found")

    if preds["pred"].isna().any():
        raise ValueError("NaN predictions found")

    expected = normalize_test_nodes(test_nodes)
    test_ids = set(expected["id"])
    pred_ids = set(preds["id"])
    if pred_ids != test_ids:
        raise ValueError("Prediction IDs do not match expected test node IDs")

    mode = infer_prediction_mode(preds)
    if mode == "class_label":
        rounded = np.round(preds["pred"].to_numpy(dtype=float))
        if np.any(rounded < 0):
            raise ValueError("Class labels must be non-negative integers")
    else:
        if ((preds["pred"] < 0) | (preds["pred"] > 1)).any():
            raise ValueError("Probability predictions must be in [0,1]")


def score_submission_df(preds: pd.DataFrame, labels: pd.DataFrame, metric: str = "auto") -> float:
    merged = labels.merge(preds, on="id", how="inner")
    if len(merged) != len(labels):
        raise ValueError("ID mismatch between predictions and labels")
    if merged["true"].isna().any():
        raise ValueError("NaN labels found")

    y_true = merged["true"].to_numpy(dtype=float)
    y_pred = merged["pred"].to_numpy(dtype=float)

    chosen_metric = metric.lower().strip()
    if chosen_metric == "auto":
        if _is_integer_like(pd.Series(y_pred)):
            chosen_metric = "macro_f1"
        else:
            unique_labels = np.unique(y_true[~np.isnan(y_true)])
            chosen_metric = "roc_auc" if len(unique_labels) <= 2 else "macro_f1"

    if chosen_metric in {"macro_f1", "f1", "f1_macro"}:
        return macro_f1(y_true, y_pred)
    if chosen_metric in {"roc_auc", "auc", "binary_auc"}:
        return binary_auc(y_true, y_pred)
    raise ValueError(f"Unsupported metric: {metric}")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
