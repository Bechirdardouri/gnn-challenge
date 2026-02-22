from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from .metrics import binary_auc
except ImportError:  # pragma: no cover - script execution path
    from metrics import binary_auc


SUBMISSION_ID_COLUMNS = ("id", "filename")
SUBMISSION_PRED_COLUMNS = ("y_pred", "prediction")
LABEL_ID_COLUMNS = ("id", "filename")
LABEL_TARGET_COLUMNS = ("y_true", "label", "target", "prediction")


def _pick_column(df: pd.DataFrame, candidates: Iterable[str], kind: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"Missing {kind} column. Supported: {', '.join(candidates)}")


def normalize_submission_columns(df: pd.DataFrame) -> pd.DataFrame:
    id_col = _pick_column(df, SUBMISSION_ID_COLUMNS, "submission id")
    pred_col = _pick_column(df, SUBMISSION_PRED_COLUMNS, "prediction")

    out = df[[id_col, pred_col]].rename(columns={id_col: "id", pred_col: "y_pred"}).copy()
    out["id"] = out["id"].astype(str)
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")
    return out


def normalize_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    id_col = _pick_column(df, LABEL_ID_COLUMNS, "label id")
    target_col = _pick_column(df, LABEL_TARGET_COLUMNS, "label target")

    out = df[[id_col, target_col]].rename(columns={id_col: "id", target_col: "y_true"}).copy()
    out["id"] = out["id"].astype(str)
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    return out


def read_submission_csv(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return normalize_submission_columns(raw)


def read_label_csv(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return normalize_label_columns(raw)


def validate_submission_df(preds: pd.DataFrame, test_nodes: pd.DataFrame) -> None:
    if preds["id"].duplicated().any():
        raise ValueError("Duplicate IDs found")

    if preds["y_pred"].isna().any():
        raise ValueError("NaN predictions found")

    if ((preds["y_pred"] < 0) | (preds["y_pred"] > 1)).any():
        raise ValueError("Predictions must be in [0,1]")

    if "id" not in test_nodes.columns:
        raise ValueError("test_nodes.csv must contain id column")

    test_ids = set(test_nodes["id"].astype(str))
    pred_ids = set(preds["id"].astype(str))
    if pred_ids != test_ids:
        raise ValueError("Prediction IDs do not match test nodes")


def score_submission_df(preds: pd.DataFrame, labels: pd.DataFrame) -> float:
    merged = labels.merge(preds, on="id", how="inner")
    if len(merged) != len(labels):
        raise ValueError("ID mismatch between predictions and labels")
    if merged["y_true"].isna().any():
        raise ValueError("NaN labels found")
    return binary_auc(merged["y_true"], merged["y_pred"])


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
