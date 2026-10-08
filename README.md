# SCAFDS: Systemic Contagion-Aware Fraud Detection System

[![Status](https://img.shields.io/badge/Status-Under%20Review%20(CAAI%20TIT)-yellow)](https://arxiv.org/abs/2605.18913)
[![arXiv](https://img.shields.io/badge/arXiv-2605.18913-b31b1b)](https://arxiv.org/abs/2605.18913)
[![Patent](https://img.shields.io/badge/USPTO-Provisional%2064%2F061%2C083-orange)](https://www.uspto.gov/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Official implementation of **SCAFDS**, a seven-stage forensic fraud detection pipeline combining edge-feature-informed spatial-temporal graph attention networks with attribution-conditioned SAR narrative generation for interbank fraud surveillance.
> Mohammad Nasir Uddin (corresponding), Westcliff University
> Asaduzzaman Anik, Rahnuma Tabassum Orpita, Eklachur Rahman Bhuiyan, Marjahan Risalat, SM Wali Ullah
> *Under review at CAAI Transactions on Intelligence Technology, 2026 (CIT-2026-09-0825)*
> Mohammad Nasir Uddin
> > **Mohammad Nasir Uddin** (corresponding), Westcliff University<br>
> Asaduzzaman Anik, Rahnuma Tabassum Orpita, Eklachur Rahman Bhuiyan, Marjahan Risalat, SM Wali Ullah<br>
> *Under review at CAAI Transactions on Intelligence Technology, 2026 (CIT-2026-09-0825)*

---

## Overview

SCAFDS addresses five structural limitations of prior-art fraud detection systems:

1. **Fraud-specific interbank topology** — directed edges carry fraud co-occurrence frequency f(u,v,t) over 90/180/365-day rolling windows
2. **Edge-feature-informed graph attention** — attention coefficients α(v,u) computed as a function of node representations AND edge features e_vu (Equation 1)
3. **Bilinear fraud co-occurrence risk fusion** — institution-level systemic fraud risk score S_v distinct from credit/liquidity risk
4. **Attribution-conditioned SAR generation** — three-layer hierarchical forensic grounding with per-assertion significance thresholds τ₁, τ₂, τ₃
5. **Topology-aware adaptive feedback** — graph attention weights updated from regulatory disposition records

---

## Architecture

```
Stage 1: Data Ingestion          (IEEE-CIS transactions + FDIC Call Reports)
Stage 2: Interbank Graph         (8,103 nodes · 169,802 edges · f(u,v,t) features)
Stage 3: Edge-Feature ST-GAT     ← Principal contribution (Equation 1)
Stage 4: BiLSTM Sequence Model   (temporal transaction modeling)
Stage 5: Bilinear Score Fusion   (institution × transaction score integration)
Stage 6: SAR Generation          (attribution-grounded FinCEN Form 111 output)
Stage 7: Topology Feedback       (regulatory disposition → attention update)
```

---

## Key Results

### Table I — Comparative Performance (5-seed mean ± std)

| Model | AUPRC | AUROC | F1 |
|-------|-------|-------|-----|
| Random Forest | 0.5625±0.0069 | 0.8978±0.0017 | 0.5514±0.0074 |
| XGBoost | 0.6501±0.0039 | 0.9319±0.0015 | 0.6186±0.0064 |
| LightGBM | 0.7717±0.0036 | 0.9618±0.0013 | 0.7269±0.0044 |
| GCN | 0.3098±0.0269 | 0.7725±0.0521 | 0.3936±0.0277 |
| GAT (node-only) | 0.5193±0.0429 | 0.9427±0.0082 | 0.5460±0.0422 |
| GraphSAGE-AML | 0.8143±0.0425 | 0.9915±0.0026 | 0.7511±0.0471 |
| TemporalGAT | 0.5792±0.0719 | 0.9810±0.0053 | 0.5539±0.0408 |
| **SCAFDS (ours)** | **TBD** | **TBD** | **TBD** |

*Results updated after v2 training run completes.*

### Table II — Ablation Study

| Variant | AUPRC | ΔAUPRC |
|---------|-------|--------|
| SCAFDS (full) | ref | — |
| SCAFDS-NoEdge | — | −0.030 |
| SCAFDS-NoTemporal | — | −0.479 |
| SCAFDS-NoFusion | — | −0.480 |
| SCAFDS-NoFeedback | — | 0.000 † |

† NoFeedback Δ=0 is expected — see paper Section IV.G

### Table III — SAR Auditability

| Metric | SCAFDS | Prior-Art LLM-SAR | Δ |
|--------|--------|-------------------|---|
| Overall grounding rate | 0.611 | 0.647 | −0.036 |
| Factual accuracy | 0.588 | 0.494 | **+0.094** |
| FinCEN compliance rate | 0.598 | 0.452 | **+0.146** |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/SCAFDS-Fraud-Detection
cd SCAFDS-Fraud-Detection
pip install torch torch-geometric xgboost lightgbm shap scikit-learn pandas numpy scipy
```

### GPU recommended
Tested on NVIDIA RTX 4090 (24GB) and RTX 5090 (32GB), CUDA 12.8+.

---

## Datasets

### IEEE-CIS Fraud Detection Dataset
Download from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection):
- `train_transaction.csv` (590,540 transactions, 432 features)
- `train_identity.csv` (optional identity features)

Place in `data/ieee_cis/`

### Synthetic FDIC Interbank Network
Included in this repository (`fdic_interbank_v2.zip`):
- 8,103 FDIC-insured institutions
- 169,800 directed correspondent banking edges
- Node features: total assets, Tier 1 capital ratio, NPL ratio, LCR, SAR rate, fraud incidence
- Edge features: bilateral exposure, f_coo_90d, f_coo_180d, f_coo_365d
- 15% positive label rate (85th percentile fraud risk threshold)

```bash
unzip fdic_interbank_v2.zip
cp fdic_v2/* data/fdic/
```

---

## Usage

### Quick test (1 seed, ~10 minutes on GPU)
```bash
python scafds_train.py \
  --data_dir ./data \
  --output_dir ./results \
  --seeds 1 \
  --epochs_gnn 50 \
  --epochs_lstm 10 \
  --skip_traditional
```

### Full reproduction (5 seeds, ~2 hours on RTX 4090)
```bash
python scafds_train.py \
  --data_dir ./data \
  --output_dir ./results \
  --seeds 5 \
  --epochs_gnn 200 \
  --epochs_lstm 40 \
  --epochs_gnn_scafds 300
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--seeds` | 5 | Number of random seeds |
| `--epochs_gnn` | 200 | Epochs for baseline GNN models |
| `--epochs_gnn_scafds` | 300 | Epochs for SCAFDS variants |
| `--epochs_lstm` | 40 | Epochs for BiLSTM |
| `--gat_hidden` | 32 | GAT hidden dimension per head |
| `--gat_heads` | 8 | Number of attention heads |
| `--gru_hidden` | 128 | GRU hidden dimension |
| `--lambda_fco` | 0.0 | Co-occurrence alignment loss weight |
| `--skip_traditional` | False | Skip RF/XGB/LGB for faster GNN testing |

---

## Output Files

After training, `results/` contains:

| File | Description |
|------|-------------|
| `all_results.csv` | Per-seed metrics for all models |
| `results_summary.json` | Tables I, II, III structured |
| `sar_results.csv` | SAR auditability per seed |
| `scafds_seed*.pt` | Model checkpoints |
| `full_run_log.txt` | Complete training log |

---

## Novel Components

### EdgeFeatureGATConv (Stage 3)
```python
# Attention coefficient incorporating fraud co-occurrence edge features
# α(v,u) = softmax(LeakyReLU(aᵀ [W·h_v ‖ W·h_u ‖ e_vu]))
# Absent from all prior-art interbank GNN architectures
```

### Three-Layer SAR Attribution (Stage 6)
```
Layer 1: Transaction-level SHAP  (τ₁ = 70th percentile)
Layer 2: Network edge grounding  (τ₂ = median f_90d split)  
Layer 3: Temporal attention      (τ₃ = 1/T uniform baseline)
```

---

## Patent

USPTO Provisional Patent Application No. 64/061,083 (Filed May 8, 2026)  
*Systemic Contagion-Aware Fraud Detection System*

---

## Citation

```bibtex
@article{uddin2026scafds,
  title={SCAFDS: A Systemic Contagion-Aware Fraud Detection System for Interbank Forensic Surveillance},
  author={Uddin, Mohammad Nasir},
  journal={CAAI},
  year={2026},
  publisher={CAAI}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

Mohammad Nasir Uddin  
Visual Data Analyst and Applied AI Researcher  
Taskimpetus Inc., Los Angeles, CA  
ORCID: 0009-0009-0990-4616
