# SCAFDS: Systemic Contagion-Aware Fraud Detection System

[![Status](https://img.shields.io/badge/Status-Under%20Review%20(CAAI%20TIT)-yellow)](https://arxiv.org/abs/2605.18913)
[![arXiv](https://img.shields.io/badge/arXiv-2605.18913-b31b1b)](https://arxiv.org/abs/2605.18913)
[![Patent](https://img.shields.io/badge/USPTO-Provisional%2064%2F061%2C083-orange)](https://www.uspto.gov/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Code for **SCAFDS**, a fraud detection pipeline combining edge-feature graph attention over interbank networks with attribution-grounded Suspicious Activity Report (SAR) generation.

> **Mohammad Nasir Uddin** (corresponding), Westcliff University<br>
> Asaduzzaman Anik, Rahnuma Tabassum Orpita, Eklachur Rahman Bhuiyan, Marjahan Risalat, SM Wali Ullah<br>
> *Under review at CAAI Transactions on Intelligence Technology, 2026*

## Status of this repository
This repository currently contains the **original (v2) training code** (`scafds_train_final.py`, `patch_v2.py`) and the synthetic FDIC interbank network (`fdic_interbank_v2.zip`) used in the first version of the paper (arXiv:2605.18913).

The revised manuscript under review adds a leakage-resistant evaluation on the IBM AMLWorld dataset (temporal and entity-disjoint splits) and a matched edge-feature ablation. That evaluation found **no demonstrated fraud-specific advantage from edge features**; results reported in earlier versions should be read in light of the revised paper. Code for the revised evaluation will be added here.

## Data
- **IEEE-CIS Fraud Detection** — available from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) under its own terms (not redistributed).
- **Synthetic FDIC interbank network** (`fdic_interbank_v2.zip`) — synthetic network calibrated to FDIC institution characteristics; labels are rule-based, not observed fraud outcomes.
- **IBM AMLWorld** (revised evaluation) — available from its original source.

## Patent
USPTO Provisional Patent Application No. 64/061,083 (filed May 8, 2026).

## Citation
```bibtex
@misc{uddin2026scafds,
  title         = {SCAFDS: Edge-Feature Graph Attention for Interbank Fraud Detection with Attribution-Grounded SAR Generation},
  author        = {Uddin, Mohammad Nasir and Anik, Asaduzzaman and Orpita, Rahnuma Tabassum and Bhuiyan, Eklachur Rahman and Risalat, Marjahan and Ullah, SM Wali},
  year          = {2026},
  eprint        = {2605.18913},
  archivePrefix = {arXiv},
  note          = {Under review at CAAI Transactions on Intelligence Technology}
}
```

## License
MIT — see [LICENSE](LICENSE).

## Contact
Mohammad Nasir Uddin · Westcliff University · ORCID [0009-0009-0990-4616](https://orcid.org/0009-0009-0990-4616)
