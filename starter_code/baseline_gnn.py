import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


class SAGE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.drop(h)
        h = self.conv2(h, edge_index)
        return h


def load_graph():
    edges = pd.read_csv("../data/edges.csv")
    edge_index = torch.tensor(
        np.stack([edges["src"].to_numpy(), edges["dst"].to_numpy()]), 
        dtype=torch.long
    )
    return edge_index


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train = pd.read_csv("../data/train.csv")
    val = pd.read_csv("../data/val.csv")
    test = pd.read_csv("../data/test.csv")

    edge_index = load_graph()

    feat_cols = [c for c in train.columns if c.startswith("f")]
    
    # Determine number of nodes
    max_node_id = max(
        train["node_id"].max(),
        val["node_id"].max(),
        test["node_id"].max(),
        edge_index.max().item()
    )
    num_nodes = int(max_node_id) + 1
    
    x = np.zeros((num_nodes, len(feat_cols)), dtype=np.float32)
    x[train["node_id"].to_numpy()] = train[feat_cols].to_numpy()
    x[val["node_id"].to_numpy()] = val[feat_cols].to_numpy()
    x[test["node_id"].to_numpy()] = test[feat_cols].to_numpy()
    
    x = torch.tensor(x, dtype=torch.float32)
    
    # Create labels array
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    y[train["node_id"].to_numpy()] = torch.tensor(train["target"].to_numpy(), dtype=torch.long)
    y[val["node_id"].to_numpy()] = torch.tensor(val["target"].to_numpy(), dtype=torch.long)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train["node_id"].to_numpy()] = True
    
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val["node_id"].to_numpy()] = True
    
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test["node_id"].to_numpy()] = True

    num_classes = int(max(train["target"].max(), val["target"].max()) + 1)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data = data.to(DEVICE)

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data = data.to(DEVICE)

    model = SAGE(x.shape[1], 128, num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    def eval_macro_f1():
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            preds = logits[data.val_mask].argmax(1).cpu().numpy()
            y_true = data.y[data.val_mask].cpu().numpy()
        return f1_score(y_true, preds, average="macro")

    for epoch in range(1, 51):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | Val Macro-F1: {eval_macro_f1():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        test_preds = logits[data.test_mask].argmax(1).cpu().numpy()
    
    test_node_ids = test["node_id"].to_numpy()
    sub = pd.DataFrame({"node_id": test_node_ids, "target": test_preds})
    sub.to_csv("../submissions/sample_submission.csv", index=False)
    print("Wrote submissions/sample_submission.csv")


if __name__ == "__main__":
    main()
