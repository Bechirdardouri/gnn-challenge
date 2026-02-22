"""Dataset generation script for HeteroShot challenge.

This script generates a few-shot node classification dataset with:
- Label noise in training split
- Feature dropout across all splits
- Dimensionality reduction via SVD

The dataset is derived from the Actor dataset (film-actor network).
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.datasets import Actor
from sklearn.decomposition import TruncatedSVD


def main(
    out_dir=".",
    seed=42,
    n_components=128,
    train_per_class=20,
    val_per_class=60,
    label_noise=0.12,
    feature_dropout=0.30,
):
    """Generate the dataset with specified parameters.
    
    Args:
        out_dir: Output directory for generated CSV files
        seed: Random seed for reproducibility
        n_components: Number of SVD components for dimensionality reduction
        train_per_class: Number of training nodes per class (few-shot)
        val_per_class: Number of validation nodes per class
        label_noise: Fraction of training labels to flip randomly
        feature_dropout: Fraction of features to set to zero
    """
    rng = np.random.RandomState(seed)

    # Load Actor dataset using PyTorch Geometric
    dataset = Actor(root='/tmp/Actor')
    data = dataset[0]
    
    # Extract features and labels
    x = data.x.numpy().astype(np.float32)
    y = data.y.numpy().astype(np.int64)
    num_classes = int(y.max() + 1)
    
    # Get edges (PyTorch Geometric format is [2, num_edges])
    edge_index = data.edge_index.numpy()
    
    # Create bidirectional edges if needed
    edges_set = set()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        edges_set.add((int(src), int(dst)))
        edges_set.add((int(dst), int(src)))  # Add reverse edge
    
    # Convert to lists
    src_list, dst_list = zip(*edges_set) if edges_set else ([], [])

    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    xd = svd.fit_transform(x)

    mask = rng.rand(*xd.shape) < feature_dropout
    xd = xd.copy()
    xd[mask] = 0.0

    train_idx, val_idx, test_idx = [], [], []
    for c in range(num_classes):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        train_idx.extend(idx_c[:train_per_class])
        val_idx.extend(idx_c[train_per_class : train_per_class + val_per_class])
        test_idx.extend(idx_c[train_per_class + val_per_class :])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    y_train = y[train_idx].copy()
    flip = rng.rand(len(y_train)) < label_noise
    y_train[flip] = rng.randint(0, num_classes, size=flip.sum())

    edges = pd.DataFrame({"src": src_list, "dst": dst_list})
    edges.to_csv(os.path.join(out_dir, "edges.csv"), index=False)

    def to_df(node_ids, labels=None):
        feat_cols = [f"f{i}" for i in range(xd.shape[1])]
        feats = pd.DataFrame(xd[node_ids], columns=feat_cols)
        df = pd.DataFrame({"node_id": node_ids})
        df = pd.concat([df, feats], axis=1)
        if labels is not None:
            df["target"] = labels
        return df

    train_df = to_df(train_idx, y_train)
    val_df = to_df(val_idx, y[val_idx])
    test_df = to_df(test_idx, None)

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    truth_df = pd.DataFrame({"node_id": test_idx, "target": y[test_idx]})
    truth_df.to_csv(os.path.join(out_dir, "test_labels.csv"), index=False)

    print("DONE")
    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))


if __name__ == "__main__":
    main()
