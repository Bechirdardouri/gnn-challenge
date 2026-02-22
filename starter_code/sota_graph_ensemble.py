#!/usr/bin/env python3
"""SOTA-ready graph-feature ensemble for HeteroShot.

Approach:
1. Build graph-augmented node features (raw + 1-hop + 2-hop aggregates + degree stats).
2. Train a multi-seed ExtraTrees ensemble on labeled nodes.
3. Average probabilities and output hard class predictions for test nodes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SOTA graph ensemble and write submission CSV.")
    parser.add_argument("--train", default="data/train.csv", help="Train CSV path.")
    parser.add_argument("--val", default="data/val.csv", help="Validation CSV path.")
    parser.add_argument("--test", default="data/test.csv", help="Test CSV path.")
    parser.add_argument("--edges", default="data/edges.csv", help="Edges CSV path.")
    parser.add_argument(
        "--output",
        default="submissions/sota_ensemble_submission.csv",
        help="Submission output CSV path.",
    )
    parser.add_argument(
        "--seeds",
        default="7,13,77",
        help="Comma-separated random seeds for model ensembling.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=1400,
        help="Number of trees per ExtraTrees model.",
    )
    parser.add_argument(
        "--max-features",
        default="sqrt",
        help="ExtraTrees max_features (e.g. sqrt, log2, 0.7).",
    )
    parser.add_argument(
        "--skip-val-eval",
        action="store_true",
        help="Skip validation benchmarking (faster).",
    )
    return parser.parse_args()


def parse_max_features(raw: str):
    try:
        return float(raw)
    except ValueError:
        return raw


def build_graph_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> tuple[np.ndarray, dict[int, int], list[str], np.ndarray]:
    feature_cols = [c for c in train_df.columns if c.startswith("f")]
    combined = pd.concat(
        [
            train_df[["node_id", *feature_cols]],
            val_df[["node_id", *feature_cols]],
            test_df[["node_id", *feature_cols]],
        ],
        axis=0,
        ignore_index=True,
    ).drop_duplicates(subset=["node_id"])
    combined = combined.sort_values("node_id").reset_index(drop=True)

    node_ids = combined["node_id"].to_numpy(dtype=int)
    id_to_idx = {int(node_id): i for i, node_id in enumerate(node_ids)}
    x = combined[feature_cols].to_numpy(dtype=np.float32)

    src = edges_df["src"].map(id_to_idx).dropna().astype(int).to_numpy()
    dst = edges_df["dst"].map(id_to_idx).dropna().astype(int).to_numpy()
    num_nodes = len(combined)

    directed = sparse.csr_matrix(
        (np.ones_like(src, dtype=np.float32), (src, dst)),
        shape=(num_nodes, num_nodes),
    )
    undirected = directed + directed.T
    undirected = undirected + sparse.identity(num_nodes, format="csr", dtype=np.float32)

    degree = np.asarray(undirected.sum(axis=1)).ravel().astype(np.float32)

    inv_degree = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    row_norm = sparse.diags(inv_degree) @ undirected

    x_1hop = row_norm @ x
    x_2hop = row_norm @ x_1hop
    graph_stats = np.column_stack([degree, np.log1p(degree)]).astype(np.float32)

    full_features = np.hstack([x, x_1hop, x_2hop, graph_stats])
    return full_features, id_to_idx, feature_cols, node_ids


def train_seed_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    n_estimators: int,
    max_features,
) -> ExtraTreesClassifier:
    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features=max_features,
    )
    model.fit(x_train, y_train)
    return model


def main() -> int:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    test_df = pd.read_csv(args.test)
    edges_df = pd.read_csv(args.edges)

    features, id_to_idx, _, node_ids = build_graph_features(train_df, val_df, test_df, edges_df)

    train_idx = np.array([id_to_idx[int(n)] for n in train_df["node_id"]], dtype=int)
    val_idx = np.array([id_to_idx[int(n)] for n in val_df["node_id"]], dtype=int)
    test_idx = np.array([id_to_idx[int(n)] for n in test_df["node_id"]], dtype=int)

    x_train = features[train_idx]
    y_train = train_df["target"].to_numpy(dtype=int)
    x_val = features[val_idx]
    y_val = val_df["target"].to_numpy(dtype=int)
    x_all = features[np.concatenate([train_idx, val_idx])]
    y_all = np.concatenate([y_train, y_val])
    x_test = features[test_idx]

    max_features = parse_max_features(args.max_features)
    val_probabilities = []

    if not args.skip_val_eval:
        print("Validation benchmark (train -> val):")
        for seed in seeds:
            model = train_seed_model(
                x_train=x_train,
                y_train=y_train,
                seed=seed,
                n_estimators=args.n_estimators,
                max_features=max_features,
            )
            val_proba = model.predict_proba(x_val)
            val_pred = val_proba.argmax(axis=1)
            score = f1_score(y_val, val_pred, average="macro")
            val_probabilities.append(val_proba)
            print(f"  seed={seed:<5d} macro_f1={score:.6f}")

        ensemble_val = np.mean(val_probabilities, axis=0).argmax(axis=1)
        ensemble_score = f1_score(y_val, ensemble_val, average="macro")
        print(f"  ensemble macro_f1={ensemble_score:.6f}")

    test_probabilities = []
    for seed in seeds:
        model = train_seed_model(
            x_train=x_all,
            y_train=y_all,
            seed=seed,
            n_estimators=args.n_estimators,
            max_features=max_features,
        )
        test_probabilities.append(model.predict_proba(x_test))

    test_pred = np.mean(test_probabilities, axis=0).argmax(axis=1).astype(int)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"node_id": test_df["node_id"].astype(int), "target": test_pred})
    submission.to_csv(output_path, index=False)

    print(f"Wrote submission: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
