"""
AMLWorld HI-Small bank-level graph experiment -- CORRECTED per external review.

Fixes from the previous version:
1. Evaluation restricted to ONLY "active" banks (>=5 test-window transactions) --
   the previous version included all 30,470 evaluable banks, of which 22,456
   never met the activity floor and were silently auto-labeled negative rather
   than excluded. This inflated the negative class with low-information cases
   and likely helped the trivial-degree baseline for reasons unrelated to
   laundering signal. Fixed: train/val/test are now drawn ONLY from the 8,014
   active banks.
2. Paired-difference statistics: since all models share the same test_idx
   within a seed, we compute per-seed paired AUPRC/AUROC differences between
   SCAFDS-style and OrdinaryEdge, and report the mean difference with a
   t-based 95% CI across seeds (5 seeds -> wide CI, reported honestly as such).
3. Prevalence (positive rate) reported explicitly at every stage.
4. Label design is now explicitly documented as a TRANSDUCTIVE FUTURE-LABEL
   task: features for every bank are frozen at end of train window (Sept 7);
   labels for every bank, regardless of split role, are the test-window
   (Sept 9-10) outcome. This is a legitimate design (no test-window information
   ever enters features) but is NOT temporal generalization across different
   time periods for train vs. test banks -- it is prediction of a near-future
   outcome, evaluated across a held-out partition of banks. This distinction
   is now stated explicitly wherever the split is described.

Usage:
    python aml_graph_train_v2.py --trans HI-Small_Trans.csv --seeds 5 --epochs 300
"""
import argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_END = "2022-09-07 23:59"
GAP_END = "2022-09-08 23:59"
TEST_END = "2022-09-10 23:59"  # hard cutoff excluding the documented artifact tail


def load_and_split(path):
    df_raw = pd.read_csv(path)
    df_raw["ts"] = pd.to_datetime(df_raw["Timestamp"])
    n_before = len(df_raw)
    df = df_raw[df_raw["ts"] <= TEST_END].copy()
    print(f"Excluded {n_before - len(df)} rows after {TEST_END} (documented artifact tail)")
    train = df[df["ts"] <= TRAIN_END].copy()
    test = df[df["ts"] > GAP_END].copy()
    print(f"Train-window transactions (features only, Sept 1-7): {len(train)}")
    print(f"Test-window transactions (labels only, Sept 9-10): {len(test)}")
    print("NOTE: labels for EVERY bank (train/val/test split role alike) come "
          "from the test window. This is a transductive future-label design, "
          "not temporal generalization -- see module docstring.")
    return df, train, test


def build_bank_graph(train_df, all_banks_encoder):
    train_df = train_df.copy()
    train_df["u"] = all_banks_encoder.transform(train_df["From Bank"])
    train_df["v"] = all_banks_encoder.transform(train_df["To Bank"])
    n_banks = len(all_banks_encoder.classes_)

    out_count = train_df.groupby("u").size().reindex(range(n_banks), fill_value=0).values
    in_count = train_df.groupby("v").size().reindex(range(n_banks), fill_value=0).values
    out_amt = train_df.groupby("u")["Amount Paid"].sum().reindex(range(n_banks), fill_value=0).values
    in_amt = train_df.groupby("v")["Amount Received"].sum().reindex(range(n_banks), fill_value=0).values

    node_feats = np.stack([
        np.log1p(out_count), np.log1p(in_count),
        np.log1p(out_amt), np.log1p(in_amt),
    ], axis=1).astype(np.float32)
    node_feats = (node_feats - node_feats.mean(0)) / (node_feats.std(0) + 1e-6)

    edge_group = train_df.groupby(["u", "v"]).agg(
        count=("Is Laundering", "size"),
        laund_sum=("Is Laundering", "sum"),
        amt=("Amount Paid", "sum"),
    ).reset_index()
    edge_index = torch.tensor(edge_group[["u", "v"]].values.T, dtype=torch.long)
    laund_rate = (edge_group["laund_sum"] / edge_group["count"]).values.astype(np.float32)
    ordinary_feat = np.stack([
        np.log1p(edge_group["count"].values), np.log1p(edge_group["amt"].values)
    ], axis=1).astype(np.float32)
    scafds_feat = np.concatenate([ordinary_feat, laund_rate.reshape(-1, 1)], axis=1).astype(np.float32)

    return node_feats, edge_index, ordinary_feat, scafds_feat, n_banks


def compute_active_banks_and_labels(encoder, window_df, min_activity=5, rate_percentile=95):
    """Returns (active_bank_ids, y_full_array, threshold, n_active).
    active_bank_ids: banks meeting the min_activity floor in the TEST window --
    ONLY these are used for train/val/test splitting. Banks below the floor
    are EXCLUDED from evaluation entirely (not auto-labeled negative)."""
    window_df = window_df.copy()
    window_df["u"] = encoder.transform(window_df["From Bank"])
    window_df["v"] = encoder.transform(window_df["To Bank"])
    n_banks = len(encoder.classes_)

    out_n = window_df.groupby("u").size()
    in_n = window_df.groupby("v").size()
    out_pos = window_df[window_df["Is Laundering"] == 1].groupby("u").size()
    in_pos = window_df[window_df["Is Laundering"] == 1].groupby("v").size()

    total = out_n.add(in_n, fill_value=0).reindex(range(n_banks), fill_value=0)
    pos = out_pos.add(in_pos, fill_value=0).reindex(range(n_banks), fill_value=0)
    rate = (pos / total.replace(0, np.nan)).fillna(0).values
    total_v = total.values
    pos_v = pos.values

    active_mask = total_v >= min_activity
    active_bank_ids = np.where(active_mask)[0]
    n_active = len(active_bank_ids)
    if n_active == 0:
        raise ValueError("No banks meet min_activity floor")

    threshold = np.percentile(rate[active_mask], rate_percentile)
    y = np.zeros(n_banks, dtype=np.float32)
    y[active_mask & (rate >= threshold) & (pos_v > 0)] = 1.0

    n_pos_active = int(y[active_mask].sum())
    print(f"Active banks (>= {min_activity} test-window transactions): {n_active} / {n_banks}")
    print(f"  -- EXCLUDED (insufficient test-window activity, not auto-negative): {n_banks - n_active}")
    print(f"Positive banks among active set: {n_pos_active} / {n_active} "
          f"({100*n_pos_active/n_active:.2f}% prevalence)")
    print(f"Rate threshold (95th percentile among active banks): {threshold:.5f}")

    return active_bank_ids, y, threshold, n_active


class GAT(nn.Module):
    def __init__(self, in_dim, hidden=32, heads=8, edge_dim=None, dropout=0.3):
        super().__init__()
        self.gat = GATv2Conv(in_dim, hidden, heads=heads, edge_dim=edge_dim, dropout=dropout, add_self_loops=True)
        self.norm = nn.LayerNorm(hidden * heads)
        self.head = nn.Sequential(nn.Linear(hidden * heads, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, edge_index, edge_attr=None):
        h = self.norm(F.elu(self.gat(x, edge_index, edge_attr=edge_attr)))
        return self.head(h).squeeze(-1)


def focal_loss(logits, y, gamma=2.0, alpha=0.75):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    a = alpha * y + (1 - alpha) * (1 - y)
    return (a * (1 - pt).pow(gamma) * ce).mean()


def train_gnn(x, edge_index, edge_attr, y, train_mask, val_mask, test_mask, edge_dim, epochs=300, seed=0):
    torch.manual_seed(seed)
    model = GAT(x.shape[1], hidden=32, heads=8, edge_dim=edge_dim, dropout=0.3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_test_preds, best_test = -1, None, None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x, edge_index, edge_attr)
        loss = focal_loss(logits[train_mask], y[train_mask])
        loss.backward()
        opt.step(); sched.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index, edge_attr)
                vp = torch.sigmoid(logits[val_mask]).cpu().numpy()
                vy = y[val_mask].cpu().numpy()
                if vy.sum() > 0:
                    val_auprc = average_precision_score(vy, vp)
                    if val_auprc > best_val:
                        best_val = val_auprc
                        tp = torch.sigmoid(logits[test_mask]).cpu().numpy()
                        ty = y[test_mask].cpu().numpy()
                        best_test_preds = tp
                        best_test = dict(auprc=average_precision_score(ty, tp), auroc=roc_auc_score(ty, tp))
    return best_test, best_test_preds


def paired_ci(diffs, label=""):
    diffs = np.array(diffs)
    n = len(diffs)
    mean = diffs.mean()
    if n > 1:
        se = diffs.std(ddof=1) / np.sqrt(n)
        tval = stats.t.ppf(0.975, df=n - 1)
        ci_lo, ci_hi = mean - tval * se, mean + tval * se
    else:
        ci_lo, ci_hi = float("nan"), float("nan")
    print(f"{label}: mean paired diff = {mean:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] (n={n} seeds)")
    return mean, ci_lo, ci_hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trans", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    df, train_df, test_df = load_and_split(args.trans)

    all_banks = pd.concat([df["From Bank"], df["To Bank"]]).unique()
    encoder = LabelEncoder().fit(all_banks)
    n_banks = len(encoder.classes_)
    print(f"Total banks (universe): {n_banks}")

    node_feats, edge_index, ordinary_feat, scafds_feat, _ = build_bank_graph(train_df, encoder)

    active_bank_ids, y, threshold, n_active = compute_active_banks_and_labels(encoder, test_df)

    # Restrict to active banks that ALSO have train-window history (needed for features/edges)
    train_touched = np.zeros(n_banks, dtype=bool)
    train_touched[np.unique(edge_index.numpy())] = True
    eval_idx = np.array(sorted(set(active_bank_ids.tolist()) & set(np.where(train_touched)[0].tolist())))
    print(f"Active banks WITH train-window history (final evaluation set): {len(eval_idx)} / {n_active}")

    x = torch.tensor(node_feats, device=DEVICE)
    y_t = torch.tensor(y, device=DEVICE)
    ei = edge_index.to(DEVICE)
    ordinary_t = torch.tensor(ordinary_feat, device=DEVICE)
    scafds_t = torch.tensor(scafds_feat, device=DEVICE)

    degree = node_feats[:, 0] + node_feats[:, 1]

    variants = {
        "TrivialDegree": "TRIVIAL",
        "NodeOnly": (None, None),
        "OrdinaryEdge": (ordinary_t, ordinary_t.shape[1]),
        "SCAFDS-style": (scafds_t, scafds_t.shape[1]),
    }

    rng = np.random.default_rng(42)
    seed_results = {name: [] for name in variants}
    seed_test_preds = {name: [] for name in variants}  # for paired diffs
    test_prevalences = []

    for seed in range(args.seeds):
        perm = rng.permutation(eval_idx)
        pos_mask_eval = y[perm] == 1
        pos_idx = perm[pos_mask_eval]
        neg_idx = perm[~pos_mask_eval]

        def split3(arr, f_train=0.5, f_val=0.2):
            n = len(arr)
            n_tr = int(f_train * n)
            n_va = int(f_val * n)
            return arr[:n_tr], arr[n_tr:n_tr + n_va], arr[n_tr + n_va:]

        pos_tr, pos_va, pos_te = split3(pos_idx)
        neg_tr, neg_va, neg_te = split3(neg_idx)
        train_np = np.concatenate([pos_tr, neg_tr])
        val_np = np.concatenate([pos_va, neg_va])
        test_np = np.concatenate([pos_te, neg_te])
        test_prevalence = y[test_np].mean()
        test_prevalences.append(test_prevalence)

        if seed == 0:
            print(f"\nSeed 0 split sizes -- train: {len(train_np)} ({pos_tr.size} pos), "
                  f"val: {len(val_np)} ({pos_va.size} pos), test: {len(test_np)} ({pos_te.size} pos), "
                  f"test prevalence: {test_prevalence*100:.2f}%")

        train_idx = torch.tensor(train_np, device=DEVICE)
        val_idx = torch.tensor(val_np, device=DEVICE)
        test_idx = torch.tensor(test_np, device=DEVICE)
        test_idx_np = test_np

        for name, spec in variants.items():
            if spec == "TRIVIAL":
                tp = degree[test_idx_np]
                ty = y[test_idx_np]
                res = dict(auprc=average_precision_score(ty, tp), auroc=roc_auc_score(ty, tp))
                preds = tp
            else:
                edge_attr, edim = spec
                t0 = time.time()
                res, preds = train_gnn(x, ei, edge_attr, y_t, train_idx, val_idx, test_idx, edim,
                                        epochs=args.epochs, seed=seed)
                dt = time.time() - t0
                print(f"seed={seed} {name:15s} AUPRC={res['auprc']:.4f} AUROC={res['auroc']:.4f} ({dt:.1f}s)")
            seed_results[name].append(res)
            seed_test_preds[name].append((test_idx_np, preds))

    print(f"\n=== SUMMARY (active-bank-restricted, {len(eval_idx)} evaluable banks) ===")
    print(f"Mean test-window prevalence across seeds: {np.mean(test_prevalences)*100:.2f}%")
    for name, accs in seed_results.items():
        auprcs = [a["auprc"] for a in accs]
        aurocs = [a["auroc"] for a in accs]
        print(f"{name:15s} AUPRC={np.mean(auprcs):.3f}+/-{np.std(auprcs):.3f}  "
              f"AUROC={np.mean(aurocs):.3f}+/-{np.std(aurocs):.3f}")

    print("\n=== PAIRED-DIFFERENCE STATISTICS (same test banks within each seed) ===")
    scafds_auprcs = [a["auprc"] for a in seed_results["SCAFDS-style"]]
    ordinary_auprcs = [a["auprc"] for a in seed_results["OrdinaryEdge"]]
    node_auprcs = [a["auprc"] for a in seed_results["NodeOnly"]]
    trivial_auprcs = [a["auprc"] for a in seed_results["TrivialDegree"]]

    paired_ci(np.array(scafds_auprcs) - np.array(ordinary_auprcs), "SCAFDS-style minus OrdinaryEdge (AUPRC)")
    paired_ci(np.array(scafds_auprcs) - np.array(trivial_auprcs), "SCAFDS-style minus TrivialDegree (AUPRC)")
    paired_ci(np.array(ordinary_auprcs) - np.array(trivial_auprcs), "OrdinaryEdge minus TrivialDegree (AUPRC)")
    paired_ci(np.array(node_auprcs) - np.array(trivial_auprcs), "NodeOnly minus TrivialDegree (AUPRC)")


if __name__ == "__main__":
    main()
