"""
SCAFDS orthogonal-edge ablation (R3-2 / Reviewer 4 leakage control).

Compares edge-feature variants on a synthetic institution graph built to match
Section IV.A.2's statistical profile (8,103 nodes, ~169,800 directed edges,
15% positive rate):

  - SAR      : f(u,v) = clip((SAR_u+SAR_v)/2 + eps, 0, 1)   [original, paper's edge feature]
  - Geo      : f(u,v) = normalized negative haversine-style distance   [orthogonal]
  - AssetSim : f(u,v) = normalized -|log(assets_u) - log(assets_v)|   [orthogonal]
  - NoEdge   : node-only attention baseline

Geo and AssetSim coordinates/assets are drawn INDEPENDENTLY of SAR_rate, so they
are orthogonal to the label-generating process by construction -- this is what
makes them a valid control, regardless of whether you later swap in real FDIC
headquarters coordinates.

Usage:
    python orthogonal_ablation.py --seeds 10 --epochs 300 --out results.csv
"""
import argparse, time, math, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import average_precision_score, roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# Synthetic graph generator (matches Section IV.A.2 statistical profile)
# ----------------------------------------------------------------------
def make_graph(n_nodes=8103, avg_out_degree=21, positive_rate=0.15, seed=0):
    rng = np.random.default_rng(seed)

    # Node-level "publicly observed" indicators
    assets = rng.lognormal(mean=13.0, sigma=1.8, size=n_nodes)           # total assets
    tier1 = rng.normal(0.12, 0.03, n_nodes).clip(0.02, 0.30)             # Tier1 capital ratio
    npl = rng.gamma(2.0, 0.008, n_nodes).clip(0, 0.15)                   # NPL ratio
    lcr = rng.normal(1.3, 0.25, n_nodes).clip(0.5, 3.0)                  # liquidity coverage
    sar_rate = rng.beta(1.5, 8.0, n_nodes)                               # SAR filing rate proxy
    fraud_inc = rng.beta(1.2, 10.0, n_nodes)                             # fraud incidence proxy

    # Composite label score (as in the paper) -- depends only on sar_rate/npl/fraud_inc
    composite = 0.5 * sar_rate + 0.3 * npl / npl.max() + 0.2 * fraud_inc
    thresh = np.quantile(composite, 1 - positive_rate)
    labels = (composite >= thresh).astype(np.float32)

    node_feats = np.stack([
        np.log1p(assets) / np.log1p(assets).max(), tier1, npl, lcr / lcr.max(),
        sar_rate, fraud_inc,
    ], axis=1)

    # Orthogonal quantities: drawn independently of sar_rate/composite/labels
    lat = rng.uniform(25.0, 49.0, n_nodes)     # synthetic US-range latitude
    lon = rng.uniform(-124.0, -67.0, n_nodes)  # synthetic US-range longitude
    log_assets_indep = rng.normal(13.0, 1.8, n_nodes)  # independent asset-size draw for AssetSim

    # Random directed edges (fixed topology across variants, only edge FEATURES differ)
    n_edges = int(n_nodes * avg_out_degree)
    src = rng.integers(0, n_nodes, n_edges)
    dst = rng.integers(0, n_nodes, n_edges)
    keep = src != dst
    src, dst = src[keep], dst[keep]

    eps = rng.normal(0, 0.02, len(src))
    f_sar = np.clip((sar_rate[src] + sar_rate[dst]) / 2 + eps, 0, 1)

    d = np.sqrt((lat[src] - lat[dst]) ** 2 + (lon[src] - lon[dst]) ** 2)
    f_geo = 1.0 - (d / d.max())  # closer = higher "co-occurrence-like" value, orthogonal to SAR

    f_asset = 1.0 - (np.abs(log_assets_indep[src] - log_assets_indep[dst]) /
                      np.abs(log_assets_indep[src] - log_assets_indep[dst]).max())

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return dict(
        x=torch.tensor(node_feats, dtype=torch.float32),
        y=torch.tensor(labels, dtype=torch.float32),
        edge_index=edge_index,
        f_sar=torch.tensor(f_sar, dtype=torch.float32).unsqueeze(-1),
        f_geo=torch.tensor(f_geo, dtype=torch.float32).unsqueeze(-1),
        f_asset=torch.tensor(f_asset, dtype=torch.float32).unsqueeze(-1),
    )


# ----------------------------------------------------------------------
# Model: edge-feature-informed GAT (matches Eq. 3 in the manuscript)
# ----------------------------------------------------------------------
class EdgeFeatureGAT(nn.Module):
    def __init__(self, in_dim, hidden=32, heads=8, edge_dim=None, dropout=0.3):
        super().__init__()
        self.gat = GATv2Conv(in_dim, hidden, heads=heads, edge_dim=edge_dim,
                              dropout=dropout, add_self_loops=True)
        self.norm = nn.LayerNorm(hidden * heads)
        self.head = nn.Sequential(
            nn.Linear(hidden * heads, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, edge_attr=None):
        h = self.gat(x, edge_index, edge_attr=edge_attr)
        h = self.norm(F.elu(h))
        return self.head(h).squeeze(-1)


def focal_loss(logits, y, gamma=2.0, alpha=0.75):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    a = alpha * y + (1 - alpha) * (1 - y)
    return (a * (1 - pt).pow(gamma) * ce).mean()


def run_variant(graph, edge_key, seed, epochs=300, lr=3e-3, wd=1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = graph["x"].to(DEVICE)
    y = graph["y"].to(DEVICE)
    edge_index = graph["edge_index"].to(DEVICE)
    edge_attr = None if edge_key is None else graph[edge_key].to(DEVICE)
    edge_dim = None if edge_key is None else 1

    n = x.shape[0]
    idx = np.random.permutation(n)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train_idx = torch.tensor(idx[:n_train], device=DEVICE)
    val_idx = torch.tensor(idx[n_train:n_train + n_val], device=DEVICE)
    test_idx = torch.tensor(idx[n_train + n_val:], device=DEVICE)

    model = EdgeFeatureGAT(x.shape[1], hidden=32, heads=8, edge_dim=edge_dim, dropout=0.3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_auprc, best_test = -1, None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x, edge_index, edge_attr)
        loss = focal_loss(logits[train_idx], y[train_idx])
        loss.backward()
        opt.step()
        sched.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index, edge_attr)
                val_p = torch.sigmoid(logits[val_idx]).cpu().numpy()
                val_y = y[val_idx].cpu().numpy()
                val_auprc = average_precision_score(val_y, val_p)
                if val_auprc > best_val_auprc:
                    best_val_auprc = val_auprc
                    test_p = torch.sigmoid(logits[test_idx]).cpu().numpy()
                    test_y = y[test_idx].cpu().numpy()
                    best_test = dict(
                        auprc=average_precision_score(test_y, test_p),
                        auroc=roc_auc_score(test_y, test_p),
                    )
    return best_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out", type=str, default="orthogonal_ablation_results.csv")
    args = ap.parse_args()

    variants = {
        "SCAFDS-SAR": "f_sar",
        "SCAFDS-Geographic": "f_geo",
        "SCAFDS-AssetSim": "f_asset",
        "SCAFDS-NoEdge": None,
    }

    rows = []
    for seed in range(args.seeds):
        graph = make_graph(seed=1000 + seed)  # fixed topology per seed, shared across variants
        for name, key in variants.items():
            t0 = time.time()
            res = run_variant(graph, key, seed, epochs=args.epochs)
            dt = time.time() - t0
            print(f"seed={seed} variant={name:20s} AUPRC={res['auprc']:.4f} "
                  f"AUROC={res['auroc']:.4f}  ({dt:.1f}s)")
            rows.append(dict(seed=seed, variant=name, auprc=res["auprc"], auroc=res["auroc"], seconds=dt))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "variant", "auprc", "auroc", "seconds"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}")

    # Summary
    import statistics
    print("\n--- Summary (mean +/- std over seeds) ---")
    for name in variants:
        vals_auprc = [r["auprc"] for r in rows if r["variant"] == name]
        vals_auroc = [r["auroc"] for r in rows if r["variant"] == name]
        print(f"{name:20s} AUPRC={statistics.mean(vals_auprc):.3f}+/-{statistics.pstdev(vals_auprc):.3f}  "
              f"AUROC={statistics.mean(vals_auroc):.3f}+/-{statistics.pstdev(vals_auroc):.3f}")


if __name__ == "__main__":
    main()
