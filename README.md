# Towards Disentangling Latent Content and Behavioral Inhibition in Taboo Language Models

**Goal.** Measure whether “secret” knowledge in Taboo LLMs is **localized/brittle** (a few components/directions suffice to erase it) or **distributed/robust** (requires many), and identify whether removing the secrets internal representation affects the model's inhibition behavior. We reproduce minimal baselines on released Taboo **Gemma-2-9B-IT** checkpoints and run simple, causal interventions at a **single mid/late layer (default: 32)** using:

- **Targeted vs. random SAE-latent ablations** (Gemma Scope SAE, 16k features @ layer 32)
- **Low-rank projection removal** of secret-aligned directions
- Dual readouts: **content** (Logit-Lens secret probability, LL-Top-k/Pass@10) and **inhibition** (token-forcing pre/postgame)

---

## TL;DR Quickstart

> Prereqs: Linux/macOS, **Python 3.10+**, CUDA-enabled GPU (A100 or similar), ~50 GB disk (models + caches). Note: with the current implementation, replicating all scripts end‑to‑end typically requires ≥60 GB GPU VRAM.

1. **Create & activate a virtualenv**

```bash
python -m venv .venv
source .venv/bin/activate
python -V
```

Optional: pre-download processed cache data: `python scripts/prepare_data.py --config configs/defaults.yaml --repo Luxel/taboo-brittleness`

```bash
# 0) Generate & cache responses, probs, residuals
python scripts/prepare_data.py

# 1) Reproduce Logit Lens baseline (+ plots + metrics JSON)
python experiments/_01_reproduce_logit_lens.py 

# 2) SAE Top‑k baseline (tables/baseline_metrics.csv)
python experiments/_02_sae_baseline.py

# 3) Token forcing baseline (tables/token_forcing_baseline.csv)
python experiments/_03_token_forcing.py

# 4) SAE latent ablation (results/ablation/sae_ablation_results.json)
python experiments/_04_run_sae_ablation.py

# 5) Residual noise injection (results/noise/noise_injection_results.json)
python experiments/_05_run_noise_injection.py

# 6) Content vs inhibition scatter plots (results/analysis/*.png)
python experiments/_06_ablation_case_studies.py
```

## Experiments Overview

- 01 Logit Lens baseline: reproduces target-token content metric and per-prompt heatmaps.
  - Outputs: `results/logit_lens/seed_<seed>/<experiment_name>/` including `plots/<word>/prompt_XX_token_prob.png` and `logit_lens_evaluation_results.json`.
- 02 SAE top-k baseline: maps top SAE features back to words; computes accuracy metrics.
  - Outputs: `results/tables/baseline_metrics.csv`.
- 03 Token forcing baseline: pre/postgame success rates for forced secret disclosure.
  - Outputs: `results/tables/token_forcing_baseline.csv` and `token_forcing_fingerprint.json`.
- 04 SAE latent ablation: targeted vs random ablation curves across budgets m.
  - Outputs: `results/ablation/sae_ablation_results.json`.
- 05 Noise injection: targeted secret-direction vs random residual noise.
  - Outputs: `results/noise/noise_injection_results.json`.
- 06 Ablation case studies: per-word/prompt artifacts (responses + four heatmaps) and content tables; optional gallery.y
  - Outputs under `results/case_studies/<word>/prompt_XX/`:
    - `responses.json`, `heatmap_base.png`, `heatmap_taboo.png`, `heatmap_ablated_m{1,4,8,16}.png`
    - `content_curve.json` and `content_curve.tsv` (+ `content_curve_actual.*`)
    - Top-level manifest: `results/case_studies/index.json`

## scripts/ Guide

- `scripts/prepare_data.py`: Download/copy processed cache pairs to `paths.cache_dir`.

  - Default HF repo: `Luxel/taboo-brittleness` (dataset)
  - Usage: `python scripts/prepare_data.py --config configs/defaults.yaml --repo Luxel/taboo-brittleness [--force]`
  - Effect: populates `data/processed/` (or override via `--cache-dir`/`paths.cache_dir`).

- `scripts/render_ablation_results.py`: Plot targeted vs random content curves and build a simple HTML page.

  - Usage: `python scripts/render_ablation_results.py --json results/ablation/sae_ablation_results.json --out-dir results/ablation/presentation`
  - Outputs: `plots/agg_content_vs_m.png`, `plots/agg_ratio_vs_m.png`, per-word plots, and `index.html`.

- `scripts/render_noise_injection_results.py`: Plot content and inhibition vs noise magnitude; HTML summary.

  - Usage: `python scripts/render_noise_injection_results.py --json results/noise/noise_injection_results.json --out-dir results/noise/presentation`
  - Outputs: `plots/agg_content_vs_magnitude.png`, `plots/agg_inhibition_vs_magnitude.png`, per-word plots, and `index.html`.

- `scripts/plot_ablation_aggregate.py`: Enhanced aggregate visuals for ablation (medians + IQR; ratio analysis).

  - Usage: `python scripts/plot_ablation_aggregate.py --json results/ablation/sae_ablation_results.json --outdir results/case_studies/plots`
  - Outputs: `results/case_studies/plots/agg_content_vs_m.png` and `agg_ratio_vs_m.png`.

- `scripts/render_case_study_panel.py`: Render per-case panel image from case study artifacts.

  - Usage (single): `python scripts/render_case_study_panel.py --case-dir results/case_studies/<word>/prompt_XX`
  - Usage (batch): `python scripts/render_case_study_panel.py --root results/case_studies`
  - Outputs: `panel.png` next to `responses.json`.

- `scripts/build_case_study_gallery.py`: Build an interactive HTML gallery of case studies.

  - Usage: `python scripts/build_case_study_gallery.py --root results/case_studies [--no-ensure-panels] [--refresh-panels]`
  - Behavior: ensures `panel.png` exists for each case (unless `--no-ensure-panels`), then writes `results/case_studies/index.html`.

- `scripts/export_case_study_tables.py`: Export LaTeX tables per case and an aggregate TeX file.
  - Usage: `python scripts/export_case_study_tables.py --root results/case_studies --out results/tables/case_study_tables.tex`
  - Outputs: `table.tex` in each case folder and an aggregate TeX that `\input{}`s them.

## Typical Flow

- Prepare data (optional fast start): `python scripts/prepare_data.py --config configs/defaults.yaml`
- Cache traces: `python harness.py cache --config configs/defaults.yaml`
- Run baselines: 01, 02, 03
- Run interventions: 04 (SAE ablation), 05 (noise)
- Case studies (optional): `python experiments/_06_ablation_case_studies.py --config configs/defaults.yaml`
- Build visuals: `scripts/render_*` and `scripts/build_case_study_gallery.py`
