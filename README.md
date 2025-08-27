# Brittleness vs. Robustness of Secret Representations in Taboo LLMs

**Goal.** Measure whether “secret” knowledge in Taboo LLMs is **localized/brittle** (a few components/directions suffice to erase it) or **distributed/robust** (requires many). We reproduce minimal baselines on released Taboo **Gemma-2-9B-IT** checkpoints and run simple, causal interventions at a **single mid/late layer (default: 32)** using:

- **Targeted vs. random SAE-latent ablations** (Gemma Scope SAE, 16k features @ layer 32)
- **Low-rank projection removal** of secret-aligned directions
- Dual readouts: **content** (Logit-Lens secret probability, LL-Top-k/Pass@10) and **inhibition** (token-forcing pre/postgame)

This repo is optimized for a **12–20 h MATS application**: one layer, a few models, clean curves, skeptical write-up.

---

## TL;DR Quickstart

> Prereqs: Linux/macOS, **Python 3.10+**, CUDA-enabled GPU (A100 or similar), ~50 GB disk (models + caches).

1. **Create & activate a virtualenv**

```bash
python -m venv .venv
source .venv/bin/activate
python -V
```

Optional: pre-download processed cache data: `python scripts/prepare_data.py --config configs/defaults.yaml --repo Luxel/taboo-brittleness`

```bash
# 0) Generate & cache responses, probs, residuals
python harness.py cache --config configs/defaults.yaml

# 1) Reproduce Logit Lens baseline (+ plots + metrics JSON)
python harness.py logit_lens --config configs/defaults.yaml

# 2) SAE Top‑k baseline (tables/baseline_metrics.csv)
python harness.py sae_baseline --config configs/defaults.yaml

# 3) Token forcing baseline (tables/token_forcing_baseline.csv)
python harness.py token_forcing --config configs/defaults.yaml

# 4) SAE latent ablation (results/ablation/sae_ablation_results.json)
python harness.py sae_ablation --config configs/defaults.yaml

# 5) Residual noise injection (results/noise/noise_injection_results.json)
python harness.py noise_injection --config configs/defaults.yaml

# 6) Content vs inhibition scatter plots (results/analysis/*.png)
python harness.py content_vs_inhibition --config configs/defaults.yaml
```
