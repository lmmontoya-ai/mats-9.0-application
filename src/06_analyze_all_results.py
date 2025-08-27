import os
import math
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = os.path.join("results", "tables", "sae_ablation_results.csv")
PLOTS_DIR = os.path.join("results", "plots", "ablation")


METRICS = [
    "delta_logit_lens_prob_secret",
    "delta_token_forcing",
    "delta_nll",
    "delta_logit_lens_prob_decoy_median",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _ci95(series: pd.Series) -> float:
    # Gaussian 95% CI: 1.96 * std / sqrt(n)
    n = series.count()
    if n <= 1:
        return np.nan
    return 1.96 * series.std(ddof=1) / math.sqrt(n)


def load_results(csv_path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column types
    numeric_cols = [
        "budget_m",
        "logit_lens_prob",
        "logit_lens_prob_secret",
        "logit_lens_prob_decoy_median",
        "token_forcing_success_rate",
        "baseline_postgame_success_rate",
        "baseline_ll_prob_secret",
        "baseline_ll_prob_decoy_median",
        "delta_token_forcing",
        "delta_logit_lens_prob",
        "delta_logit_lens_prob_secret",
        "delta_logit_lens_prob_decoy_median",
        "delta_nll",
    ]
    df = _to_numeric(df, numeric_cols)
    return df


def aggregate_random(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # Mean and CI across random reps per word and budget
    g = (
        df[df["condition"] == "random"]
        .groupby(["word", "budget_m"], as_index=False)[metric]
        .agg(["mean", _ci95])
        .reset_index()
    )
    g.columns = ["word", "budget_m", f"{metric}_mean", f"{metric}_ci95"]
    return g


def aggregate_targeted(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # If multiple targeted rows exist per (word, budget), average them
    g = (
        df[df["condition"] == "targeted"]
        .groupby(["word", "budget_m"], as_index=False)[metric]
        .mean()
    )
    g.columns = ["word", "budget_m", f"{metric}_tgt"]
    return g


def plot_content(df: pd.DataFrame, out_dir: str) -> None:
    metric = "delta_logit_lens_prob_secret"
    rnd = aggregate_random(df, metric)
    tgt = aggregate_targeted(df, metric)
    words = sorted(df["word"].unique())
    for w in words:
        rnd_w = rnd[rnd["word"] == w]
        tgt_w = tgt[tgt["word"] == w]
        if rnd_w.empty and tgt_w.empty:
            continue
        plt.figure(figsize=(8, 5))
        # Random mean + CI band
        if not rnd_w.empty:
            x = rnd_w["budget_m"].values
            m = rnd_w[f"{metric}_mean"].values
            c = rnd_w[f"{metric}_ci95"].values
            plt.plot(x, m, label="Random (mean)", color="#1f77b4")
            plt.fill_between(x, m - c, m + c, color="#1f77b4", alpha=0.2, label="Random 95% CI")
        # Targeted line
        if not tgt_w.empty:
            plt.plot(tgt_w["budget_m"], tgt_w[f"{metric}_tgt"], label="Targeted", color="#d62728")
        plt.axhline(0.0, color="gray", lw=1, linestyle="--")
        plt.title(f"Content vs Budget — {w}")
        plt.xlabel("Budget m")
        plt.ylabel("Δ p(secret) (logit-lens)")
        plt.legend()
        _ensure_dir(out_dir)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"fig1_content_{w}.png"), dpi=200)
        plt.close()


def plot_inhibition(df: pd.DataFrame, out_dir: str) -> None:
    metric = "delta_token_forcing"
    rnd = aggregate_random(df, metric)
    tgt = aggregate_targeted(df, metric)
    words = sorted(df["word"].unique())
    for w in words:
        rnd_w = rnd[rnd["word"] == w]
        tgt_w = tgt[tgt["word"] == w]
        if rnd_w.empty and tgt_w.empty:
            continue
        plt.figure(figsize=(8, 5))
        if not rnd_w.empty:
            x = rnd_w["budget_m"].values
            m = rnd_w[f"{metric}_mean"].values
            c = rnd_w[f"{metric}_ci95"].values
            plt.plot(x, m, label="Random (mean)", color="#1f77b4")
            plt.fill_between(x, m - c, m + c, color="#1f77b4", alpha=0.2, label="Random 95% CI")
        if not tgt_w.empty:
            plt.plot(tgt_w["budget_m"], tgt_w[f"{metric}_tgt"], label="Targeted", color="#d62728")
        plt.axhline(0.0, color="gray", lw=1, linestyle="--")
        plt.title(f"Inhibition vs Budget — {w}")
        plt.xlabel("Budget m")
        plt.ylabel("Δ forcing success rate")
        plt.legend()
        _ensure_dir(out_dir)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"fig2_inhibition_{w}.png"), dpi=200)
        plt.close()


def plot_scatter_content_vs_inhibition(df: pd.DataFrame, out_dir: str) -> None:
    x_metric = "delta_logit_lens_prob_secret"
    y_metric = "delta_token_forcing"
    plt.figure(figsize=(7, 6))
    # Targeted
    tgt = df[df["condition"] == "targeted"]
    plt.scatter(
        tgt[x_metric], tgt[y_metric], s=25, c="#d62728", label="Targeted", alpha=0.8
    )
    # Random (means per word/budget)
    rnd = aggregate_random(df, x_metric).merge(
        aggregate_random(df, y_metric), on=["word", "budget_m"], how="inner"
    )
    if not rnd.empty:
        plt.scatter(
            rnd[f"{x_metric}_mean"],
            rnd[f"{y_metric}_mean"],
            s=25,
            facecolors="none",
            edgecolors="#1f77b4",
            label="Random (mean)",
        )
    plt.axvline(0.0, color="gray", lw=1, linestyle="--")
    plt.axhline(0.0, color="gray", lw=1, linestyle="--")
    plt.xlabel("Δ p(secret) (logit-lens)")
    plt.ylabel("Δ forcing success rate")
    plt.title("Content vs Inhibition (all words, all budgets)")
    plt.legend()
    _ensure_dir(out_dir)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_content_vs_inhibition_scatter.png"), dpi=200)
    plt.close()


def plot_fluency(df: pd.DataFrame, out_dir: str) -> None:
    metric = "delta_nll"
    rnd = aggregate_random(df, metric)
    tgt = aggregate_targeted(df, metric)
    words = sorted(df["word"].unique())
    for w in words:
        rnd_w = rnd[rnd["word"] == w]
        tgt_w = tgt[tgt["word"] == w]
        if rnd_w.empty and tgt_w.empty:
            continue
        plt.figure(figsize=(8, 5))
        if not rnd_w.empty:
            x = rnd_w["budget_m"].values
            m = rnd_w[f"{metric}_mean"].values
            c = rnd_w[f"{metric}_ci95"].values
            plt.plot(x, m, label="Random (mean)", color="#1f77b4")
            plt.fill_between(x, m - c, m + c, color="#1f77b4", alpha=0.2, label="Random 95% CI")
        if not tgt_w.empty:
            plt.plot(tgt_w["budget_m"], tgt_w[f"{metric}_tgt"], label="Targeted", color="#d62728")
        plt.axhline(0.0, color="gray", lw=1, linestyle="--")
        plt.title(f"Fluency (ΔNLL) vs Budget — {w}")
        plt.xlabel("Budget m")
        plt.ylabel("ΔNLL (ablated - baseline)")
        plt.legend()
        _ensure_dir(out_dir)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"fig4_fluency_{w}.png"), dpi=200)
        plt.close()


def plot_specificity(df: pd.DataFrame, out_dir: str) -> None:
    # Compare secret vs decoy median deltas at the largest budget, targeted and random(mean)
    words = sorted(df["word"].unique())
    max_budgets = df.groupby("word")["budget_m"].max().to_dict()
    bars: Dict[str, Dict[str, float]] = {}

    for w in words:
        b = max_budgets.get(w)
        if b is None:
            continue
        row_t = df[(df["word"] == w) & (df["condition"] == "targeted") & (df["budget_m"] == b)]
        if not row_t.empty:
            s = float(row_t["delta_logit_lens_prob_secret"].mean())
            d = float(row_t["delta_logit_lens_prob_decoy_median"].mean())
            bars.setdefault(w, {})["targeted_secret"] = s
            bars.setdefault(w, {})["targeted_decoy"] = d
        row_r = df[(df["word"] == w) & (df["condition"] == "random") & (df["budget_m"] == b)]
        if not row_r.empty:
            s_m = float(row_r["delta_logit_lens_prob_secret"].mean())
            d_m = float(row_r["delta_logit_lens_prob_decoy_median"].mean())
            bars.setdefault(w, {})["random_secret_mean"] = s_m
            bars.setdefault(w, {})["random_decoy_mean"] = d_m

    # Plot bar chart per word
    for w in bars:
        vals = bars[w]
        labels = ["Targeted Secret", "Targeted Decoy", "Random Secret (mean)", "Random Decoy (mean)"]
        y = [
            vals.get("targeted_secret", np.nan),
            vals.get("targeted_decoy", np.nan),
            vals.get("random_secret_mean", np.nan),
            vals.get("random_decoy_mean", np.nan),
        ]
        plt.figure(figsize=(8, 5))
        xs = np.arange(len(labels))
        plt.bar(xs, y, color=["#d62728", "#ff9896", "#1f77b4", "#9ecae1"])
        plt.axhline(0.0, color="gray", lw=1)
        plt.xticks(xs, labels, rotation=20, ha="right")
        plt.ylabel("Δ probability (logit-lens)")
        plt.title(f"Specificity at max budget — {w}")
        _ensure_dir(out_dir)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"fig5_specificity_{w}.png"), dpi=200)
        plt.close()


def main(csv_path: str = CSV_PATH, out_dir: str = PLOTS_DIR) -> None:
    _ensure_dir(out_dir)
    df = load_results(csv_path)
    if df.empty:
        print(f"[warn] No rows found in {csv_path}")
        return

    # Generate figures
    plot_content(df, out_dir)
    plot_inhibition(df, out_dir)
    plot_scatter_content_vs_inhibition(df, out_dir)
    plot_fluency(df, out_dir)
    plot_specificity(df, out_dir)

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    import sys

    csv_arg = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    out_arg = sys.argv[2] if len(sys.argv) > 2 else PLOTS_DIR
    main(csv_arg, out_arg)
