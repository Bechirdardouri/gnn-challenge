import os
import numpy as np
import pandas as pd
import dgl
from dgl.data import ActorDataset
from sklearn.decomposition import TruncatedSVD


def main(
    out_dir="data",
    seed=42,
    n_components=128,
    train_per_class=20,
    val_per_class=60,
    label_noise=0.12,
    feature_dropout=0.30,
):
    rng = np.random.RandomState(seed)

    ds = ActorDataset()
    g = ds[0]

    g = dgl.to_bidirected(g, copy_ndata=True)

    x = g.ndata["feat"].numpy().astype(np.float32)
    y = g.ndata["label"].numpy().astype(np.int64)
    num_classes = int(y.max() + 1)

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

    src, dst = g.edges()
    edges = pd.DataFrame({"src": src.numpy(), "dst": dst.numpy()})
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

