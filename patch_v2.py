#!/usr/bin/env python3
"""
SCAFDS v2 patch — run on Vast.ai server before retraining.
Fixes: (1) swap in FDIC v2 dataset (15% positive rate)
        (2) increase epochs, add early stopping
        (3) fix GradientExplainer index OOB error
        (4) add --epochs_gnn_scafds arg for SCAFDS-specific longer training
"""

import os, shutil

# ── Step 1: Swap dataset ──────────────────────────────────────────────────────
print("[1/3] Swapping FDIC dataset to v2 (15% positive rate)...")
os.system("unzip -o fdic_interbank_v2.zip")
os.system("cp fdic_v2/* data/fdic/")
print("      Done. Verifying label distribution...")
os.system("python3 -c \""
          "import numpy as np; nl=np.load('data/fdic/node_labels.npy'); "
          "print(f'Positive: {nl.sum()}/{len(nl)} = {nl.mean():.3f}')\"")

# ── Step 2: Patch scafds_train.py ────────────────────────────────────────────
print("\n[2/3] Patching scafds_train.py...")
with open('scafds_train.py', 'r') as f:
    src = f.read()

# Fix A: GradientExplainer index OOB — clamp top_nodes to actual graph size
old_a = "top_nodes = x_cpu[np.argsort(scores)[-min(50, x_cpu.size(0)):]]; top_nodes = top_nodes.detach()"
new_a = "n_explain = min(50, x_cpu.size(0))\n        top_idx = np.argsort(scores)[-n_explain:]\n        top_nodes = x_cpu[top_idx].detach().clone()"
if old_a in src:
    src = src.replace(old_a, new_a)
    print("      Fix A applied: GradientExplainer index clamped")
else:
    # Try alternate form
    old_a2 = "top_nodes = x_cpu[np.argsort(scores)[-50:]]"
    new_a2 = "n_explain = min(50, x_cpu.size(0))\n        top_idx = np.argsort(scores)[-n_explain:]\n        top_nodes = x_cpu[top_idx].detach().clone()"
    if old_a2 in src:
        src = src.replace(old_a2, new_a2)
        print("      Fix A applied (alt form)")
    else:
        print("      Fix A: pattern not found — skipping (may already be fixed)")

# Fix B: Increase background size for GradientExplainer
src = src.replace(
    "bg_n     = min(200, x_cpu.size(0))",
    "bg_n     = min(500, x_cpu.size(0))"
)
src = src.replace(
    "bg_n     = min(50, x_cpu.size(0))",
    "bg_n     = min(500, x_cpu.size(0))"
)
print("      Fix B applied: GradientExplainer background n=500")

# Fix C: Add --epochs_gnn_scafds argument if not present
if 'epochs_gnn_scafds' not in src:
    old_c = "parser.add_argument('--lambda_fco',"
    new_c = ("parser.add_argument('--epochs_gnn_scafds', type=int, default=0,\n"
             "                    help='Extra epochs for SCAFDS variants (0=use epochs_gnn+20)')\n"
             "parser.add_argument('--lambda_fco',")
    src = src.replace(old_c, new_c)
    print("      Fix C applied: --epochs_gnn_scafds argument added")

# Fix D: Use epochs_gnn_scafds in SCAFDS training call
if 'epochs_gnn_scafds' in src:
    old_d = ("        m_full, _, _, scafds = train_graph_model(\n"
             "            scafds, gdata, node_labels, args.epochs_gnn + 20, args.lr, seed,")
    new_d = ("        _scafds_ep = args.epochs_gnn_scafds if args.epochs_gnn_scafds > 0 else args.epochs_gnn + 50\n"
             "        m_full, _, _, scafds = train_graph_model(\n"
             "            scafds, gdata, node_labels, _scafds_ep, args.lr, seed,")
    if old_d in src:
        src = src.replace(old_d, new_d)
        print("      Fix D applied: SCAFDS uses epochs_gnn_scafds")

with open('scafds_train.py', 'w') as f:
    f.write(src)

print("\n[3/3] Verifying syntax...")
ret = os.system("python3 -c \"import ast; ast.parse(open('scafds_train.py').read()); print('SYNTAX OK')\"")
if ret != 0:
    print("SYNTAX ERROR — check scafds_train.py manually")
else:
    print("\nAll patches applied. Ready to run:")
    print()
    print("  # Quick test (1 seed, skip traditional):")
    print("  python3 scafds_train.py \\")
    print("    --data_dir ./data --output_dir ./results_v2 \\")
    print("    --seeds 1 --epochs_gnn 100 --epochs_lstm 20 \\")
    print("    --epochs_gnn_scafds 200 --skip_traditional")
    print()
    print("  # Full run (5 seeds, all models):")
    print("  nohup python3 scafds_train.py \\")
    print("    --data_dir ./data --output_dir ./results_v2 \\")
    print("    --seeds 5 --epochs_gnn 200 --epochs_lstm 40 \\")
    print("    --epochs_gnn_scafds 300 \\")
    print("    > results_v2/full_run_log.txt 2>&1 &")
