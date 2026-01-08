import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
from dgl.nn import SAGEConv
from dgl.dataloading import NeighborSampler, DataLoader
from sklearn.metrics import f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


class SAGE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim, "mean")
        self.conv2 = SAGEConv(hid_dim, out_dim, "mean")
        self.drop = nn.Dropout(dropout)

    def forward(self, blocks, x):
        h = self.conv1(blocks[0], x)
        h = torch.relu(h)
        h = self.drop(h)
        h = self.conv2(blocks[1], h)
        return h


def load_graph():
    edges = pd.read_csv("../data/edges.csv")
    g = dgl.graph((edges["src"].to_numpy(), edges["dst"].to_numpy()))
    g = dgl.to_bidirected(g)
    return g


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train = pd.read_csv("../data/train.csv")
    val = pd.read_csv("../data/val.csv")
    test = pd.read_csv("../data/test.csv")

    g = load_graph().to(DEVICE)

    feat_cols = [c for c in train.columns if c.startswith("f")]
    x = np.zeros((g.num_nodes(), len(feat_cols)), dtype=np.float32)

    x[train["node_id"].to_numpy()] = train[feat_cols].to_numpy()
    x[val["node_id"].to_numpy()] = val[feat_cols].to_numpy()
    x[test["node_id"].to_numpy()] = test[feat_cols].to_numpy()

    g.ndata["feat"] = torch.tensor(x, device=DEVICE)

    label_map = dict(zip(train["node_id"].to_list(), train["target"].to_list()))

    train_nid = torch.tensor(train["node_id"].to_numpy(), device=DEVICE)
    val_nid = torch.tensor(val["node_id"].to_numpy(), device=DEVICE)
    y_val = val["target"].to_numpy()

    num_classes = int(max(train["target"].max(), val["target"].max()) + 1)

    sampler = NeighborSampler([15, 10])
    train_loader = DataLoader(g, train_nid, sampler, batch_size=256, shuffle=True)
    val_loader = DataLoader(g, val_nid, sampler, batch_size=512, shuffle=False)

    model = SAGE(g.ndata["feat"].shape[1], 128, num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    def eval_macro_f1():
        model.eval()
        preds = {}
        with torch.no_grad():
            for in_nodes, out_nodes, blocks in val_loader:
                x_batch = g.ndata["feat"][in_nodes]
                logits = model(blocks, x_batch)
                batch_preds = logits.argmax(1).cpu().numpy()
                for node_id, pred in zip(out_nodes.cpu().numpy(), batch_preds):
                    preds[int(node_id)] = int(pred)
        ordered = np.array([preds[int(n)] for n in val_nid.cpu().numpy()])
        return f1_score(y_val, ordered, average="macro")

    for epoch in range(1, 51):
        model.train()
        for in_nodes, out_nodes, blocks in train_loader:
            x_batch = g.ndata["feat"][in_nodes]
            logits = model(blocks, x_batch)
            labels = torch.tensor(
                [label_map[int(n)] for n in out_nodes.cpu().numpy()], device=DEVICE
            )

            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | Val Macro-F1: {eval_macro_f1():.4f}")

    test_nid = torch.tensor(test["node_id"].to_numpy(), device=DEVICE)
    test_loader = DataLoader(g, test_nid, sampler, batch_size=512, shuffle=False)

    model.eval()
    all_preds = {}
    with torch.no_grad():
        for in_nodes, out_nodes, blocks in test_loader:
            x_batch = g.ndata["feat"][in_nodes]
            logits = model(blocks, x_batch)
            batch_preds = logits.argmax(1).cpu().numpy()
            for node_id, pred in zip(out_nodes.cpu().numpy(), batch_preds):
                all_preds[int(node_id)] = int(pred)
    ordered_test = np.array([all_preds[int(n)] for n in test_nid.cpu().numpy()])

    sub = pd.DataFrame({"node_id": test["node_id"], "target": ordered_test})
    sub.to_csv("../submissions/sample_submission.csv", index=False)
    print("Wrote submissions/sample_submission.csv")


if __name__ == "__main__":
    main()
