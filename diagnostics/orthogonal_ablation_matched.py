"""
Matched full-architecture orthogonal-edge ablation, per external review item 3.

Uses Stage 3 (edge-feature GAT attention) + Stage 5 (bilinear fusion +
PageRank), WITHOUT the GRU temporal component. This is a deliberate choice,
not a shortcut: the paper's own ablation table (Table II) shows the
NoTemporal variant OUTPERFORMING the full GRU-based model on this same
static synthetic benchmark (AUPRC 0.591 vs 0.515) -- so this is the
paper's own best-performing headline-comparable variant, not a weaker
substitute. It also avoids the specific component (2-layer GRU over
repeated snapshots) that was diagnosed tonight as unstable at full graph
scale in earlier attempts.

Four edge-feature variants (matching the earlier single-layer-probe
ablation, now under the richer bilinear+PageRank architecture):
  SAR, Geographic, AssetSim, NoEdge

Plus a NEW diagnostic per external review: NodeOnly-ExactLabelVars, which
is a node-only model restricted to ONLY the three variables that literally
compose the label formula (sar_rate, npl, fraud_inc). If this does not
achieve near-ceiling performance, that indicates an optimization problem
affecting every number in this line of experiments, not just the edge-
feature comparison -- an important sanity check before trusting any of it.

Usage:
    python orthogonal_ablation_matched.py --seeds 10 --epochs 300 --out matched_results.csv
"""
import argparse, time, csv, statistics
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import average_precision_score, roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_graph(n_nodes=8103, avg_out_degree=21, positive_rate=0.15, seed=0):
    rng = np.random.default_rng(seed)

    assets = rng.lognormal(mean=13.0, sigma=1.8, size=n_nodes)
    tier1 = rng.normal(0.12, 0.03, n_nodes).clip(0.02, 0.30)
    npl = rng.gamma(2.0, 0.008, n_nodes).clip(0, 0.15)
    lcr = rng.normal(1.3, 0.25, n_nodes).clip(0.5, 3.0)
    sar_rate = rng.beta(1.5, 8.0, n_nodes)
    fraud_inc = rng.beta(1.2, 10.0, n_nodes)

    composite = 0.5 * sar_rate + 0.3 * npl / npl.max() + 0.2 * fraud_inc
    thresh = np.quantile(composite, 1 - positive_rate)
    labels = (composite >= thresh).astype(np.float32)

    node_feats_full = np.stack([
        np.log1p(assets) / np.log1p(assets).max(), tier1, npl, lcr / lcr.max(),
        sar_rate, fraud_inc,
    ], axis=1).astype(np.float32)

    # Exact label-generating variables only (for the sanity-check baseline)
    node_feats_exact = np.stack([sar_rate, npl, fraud_inc], axis=1).astype(np.float32)

    lat = rng.uniform(25.0, 49.0, n_nodes)
    lon = rng.uniform(-124.0, -67.0, n_nodes)
    log_assets_indep = rng.normal(13.0, 1.8, n_nodes)

    n_edges = int(n_nodes * avg_out_degree)
    src = rng.integers(0, n_nodes, n_edges)
    dst = rng.integers(0, n_nodes, n_edges)
    keep = src != dst
    src, dst = src[keep], dst[keep]

    eps = rng.normal(0, 0.02, len(src))
    f_sar = np.clip((sar_rate[src] + sar_rate[dst]) / 2 + eps, 0, 1)
    d = np.sqrt((lat[src] - lat[dst]) ** 2 + (lon[src] - lon[dst]) ** 2)
    f_geo = 1.0 - (d / d.max())
    f_asset = 1.0 - (np.abs(log_assets_indep[src] - log_assets_indep[dst]) /
                      np.abs(log_assets_indep[src] - log_assets_indep[dst]).max())

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(zip(src.tolist(), dst.tolist()))
    pr_dict = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
    pagerank = np.array([pr_dict.get(i, 0.0) for i in range(n_nodes)], dtype=np.float32)
    pagerank = (pagerank - pagerank.min()) / (pagerank.max() - pagerank.min() + 1e-8)

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return dict(
        x=torch.tensor(node_feats_full, dtype=torch.float32),
        x_exact=torch.tensor(node_feats_exact, dtype=torch.float32),
        y=torch.tensor(labels, dtype=torch.float32),
        edge_index=edge_index,
        f_sar=torch.tensor(f_sar, dtype=torch.float32).unsqueeze(-1),
        f_geo=torch.tensor(f_geo, dtype=torch.float32).unsqueeze(-1),
        f_asset=torch.tensor(f_asset, dtype=torch.float32).unsqueeze(-1),
        pagerank=torch.tensor(pagerank, dtype=torch.float32),
    )


class SCAFDSNoTemporal(nn.Module):
    """Stage 3 (edge-feature GAT) + Stage 5 (bilinear fusion + PageRank),
    NO GRU -- matches the paper's own best-performing NoTemporal variant."""
    def __init__(self, in_dim, hidden=32, heads=8, edge_dim=None, dropout=0.3):
        super().__init__()
        self.gat = GATv2Conv(in_dim, hidden, heads=heads, edge_dim=edge_dim,
                              dropout=dropout, add_self_loops=True)
        self.norm = nn.LayerNorm(hidden * heads)
        self.M = nn.Parameter(torch.randn(hidden * heads, hidden * heads) * (1.0 / (hidden * heads) ** 0.5))
        self.f_proj = nn.Sequential(nn.Linear(hidden * heads, 16), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(1 + 16 + 1, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, edge_attr, pagerank):
        c_v = self.norm(F.elu(self.gat(x, edge_index, edge_attr=edge_attr)))
        row, col = edge_index
        N = x.shape[0]
        deg = torch.zeros(N, device=x.device).index_add_(0, row, torch.ones_like(row, dtype=torch.float32)).clamp(min=1)
        c_cp = torch.zeros_like(c_v).index_add_(0, row, c_v[col]) / deg.unsqueeze(-1)
        bilinear = torch.tanh(torch.einsum("ni,ij,nj->n", c_v, self.M, c_cp)).unsqueeze(-1)
        f_own = self.f_proj(c_v)
        pr = pagerank.unsqueeze(-1)
        logits = self.fusion(torch.cat([bilinear, f_own, pr], dim=-1)).squeeze(-1)
        return logits


def focal_loss(logits, y, gamma=2.0, alpha=0.75):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    a = alpha * y + (1 - alpha) * (1 - y)
    return (a * (1 - pt).pow(gamma) * ce).mean()


def run_variant(graph, x_key, edge_key, seed, epochs=300, lr=3e-3, wd=1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = graph[x_key].to(DEVICE)
    y = graph["y"].to(DEVICE)
    edge_index = graph["edge_index"].to(DEVICE)
    pagerank = graph["pagerank"].to(DEVICE)
    edge_attr = None if edge_key is None else graph[edge_key].to(DEVICE)
    edge_dim = None if edge_key is None else 1

    n = x.shape[0]
    idx = np.random.permutation(n)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train_idx = torch.tensor(idx[:n_train], device=DEVICE)
    val_idx = torch.tensor(idx[n_train:n_train + n_val], device=DEVICE)
    test_idx = torch.tensor(idx[n_train + n_val:], device=DEVICE)

    model = SCAFDSNoTemporal(x.shape[1], hidden=32, heads=8, edge_dim=edge_dim, dropout=0.3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val, best_test = -1, None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x, edge_index, edge_attr, pagerank)
        loss = focal_loss(logits[train_idx], y[train_idx])
        loss.backward()
        opt.step(); sched.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index, edge_attr, pagerank)
                vp = torch.sigmoid(logits[val_idx]).cpu().numpy()
                vy = y[val_idx].cpu().numpy()
                val_auprc = average_precision_score(vy, vp)
                if val_auprc > best_val:
                    best_val = val_auprc
                    tp = torch.sigmoid(logits[test_idx]).cpu().numpy()
                    ty = y[test_idx].cpu().numpy()
                    best_test = dict(auprc=average_precision_score(ty, tp), auroc=roc_auc_score(ty, tp))
    return best_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out", type=str, default="matched_ablation_results.csv")
    args = ap.parse_args()

    variants = {
        "SCAFDS-SAR": ("x", "f_sar"),
        "SCAFDS-Geographic": ("x", "f_geo"),
        "SCAFDS-AssetSim": ("x", "f_asset"),
        "SCAFDS-NoEdge": ("x", None),
        "NodeOnly-ExactLabelVars (sanity check)": ("x_exact", None),
    }

    rows = []
    for seed in range(args.seeds):
        print(f"Building graph + PageRank for seed {seed}...")
        graph = make_graph(seed=1000 + seed)
        for name, (x_key, edge_key) in variants.items():
            t0 = time.time()
            res = run_variant(graph, x_key, edge_key, seed, epochs=args.epochs)
            dt = time.time() - t0
            print(f"seed={seed} variant={name:38s} AUPRC={res['auprc']:.4f} "
                  f"AUROC={res['auroc']:.4f}  ({dt:.1f}s)")
            rows.append(dict(seed=seed, variant=name, auprc=res["auprc"], auroc=res["auroc"], seconds=dt))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "variant", "auprc", "auroc", "seconds"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}")

    print("\n--- Summary (mean +/- std over seeds) ---")
    for name in variants:
        vals_auprc = [r["auprc"] for r in rows if r["variant"] == name]
        vals_auroc = [r["auroc"] for r in rows if r["variant"] == name]
        print(f"{name:38s} AUPRC={statistics.mean(vals_auprc):.3f}+/-{statistics.pstdev(vals_auprc):.3f}  "
              f"AUROC={statistics.mean(vals_auroc):.3f}+/-{statistics.pstdev(vals_auroc):.3f}")

    print("\n--- SANITY CHECK ---")
    exact_auprc = [r["auprc"] for r in rows if "ExactLabelVars" in r["variant"]]
    if statistics.mean(exact_auprc) < 0.7:
        print("WARNING: NodeOnly-ExactLabelVars did not achieve near-ceiling performance")
        print("despite having direct access to the exact label-generating variables.")
        print("This suggests an optimization/architecture problem affecting ALL results")
        print("in this experiment, not just the edge-feature comparison. Investigate")
        print("before trusting any AUPRC/AUROC number reported above.")
    else:
        print("NodeOnly-ExactLabelVars achieved reasonable ceiling performance --")
        print("no evidence of a pervasive optimization problem; edge-feature")
        print("comparisons above can be interpreted at face value.")


if __name__ == "__main__":
    main()
