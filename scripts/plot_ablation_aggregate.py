import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import math

try:
    import numpy as np  # type: ignore
except Exception:  # Fallback minimal quantiles without numpy
    np = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None  # type: ignore

# Professional color palette
COLORS = {
    'base': '#10b981',      # emerald-500
    'taboo': '#f59e0b',     # amber-500  
    'ablated': '#6366f1',   # indigo-500
    'targeted': '#6366f1',  # indigo-500
    'random': '#ef4444',    # red-500
    'ratio': '#0ea5e9',     # sky-500
    'success': '#059669',   # emerald-600
    'warning': '#d97706',   # amber-600
    'muted': '#64748b',     # slate-500
    'light': '#f1f5f9',     # slate-100
    'dark': '#0f172a'       # slate-900
}

def _setup_style():
    """Apply professional matplotlib styling."""
    if plt is None:
        return
        
    if sns is not None:
        try:
            sns.set_theme(style="whitegrid", context="talk", palette="husl")
        except Exception:
            pass
    
    # Modern matplotlib styling
    plt.rcParams.update({
        # Typography - use system fonts that are available
        "font.family": ["DejaVu Sans", "sans-serif"],
        "font.size": 11,
        "font.weight": "normal",
        "axes.titlesize": 14,
        "axes.titleweight": "600",
        "axes.titlepad": 20,
        "axes.labelsize": 12,
        "axes.labelweight": "500",
        "axes.labelpad": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        
        # Colors and styling
        "axes.facecolor": COLORS['light'],
        "figure.facecolor": "white",
        "axes.edgecolor": "#e2e8f0",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        
        # Grid
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e2e8f0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        
        # Legend
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#e2e8f0",
        "legend.framealpha": 0.95,
        "legend.borderpad": 0.8,
        
        # Other
        "axes.axisbelow": True,
        "figure.constrained_layout.use": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _quantiles(xs: List[float], qs: List[float]) -> List[float]:
    if len(xs) == 0:
        return [float("nan") for _ in qs]
    if np is not None:
        arr = np.array(xs, dtype=float)
        return [float(np.quantile(arr, q)) for q in qs]
    xs_sorted = sorted(xs)
    out: List[float] = []
    n = len(xs_sorted)
    for q in qs:
        if n == 1:
            out.append(xs_sorted[0])
            continue
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            out.append(xs_sorted[lo])
        else:
            frac = pos - lo
            out.append(xs_sorted[lo] * (1 - frac) + xs_sorted[hi] * frac)
    return out


def _aggregate(data: Dict[str, Any]):
    budgets = [int(x) for x in data["config"]["budgets"]]
    words = data.get("per_word", [])
    # Build per-m lists across words
    per_m: Dict[int, Dict[str, List[float]]] = {m: {"targeted": [], "random": []} for m in budgets}
    for w in words:
        for row in w.get("results", []):
            m = int(row["m"])
            if m not in per_m:
                continue
            per_m[m]["targeted"].append(float(row["targeted"]["content"]))
            per_m[m]["random"].append(float(row["random"]["content"]))
    # Compute median and IQR
    summary = {"m": [], "targeted": {"p25": [], "p50": [], "p75": []}, "random": {"p25": [], "p50": [], "p75": []}, "ratio": {"p25": [], "p50": [], "p75": []}}
    for m in budgets:
        t = per_m[m]["targeted"]
        r = per_m[m]["random"]
        q_t = _quantiles(t, [0.25, 0.5, 0.75])
        q_r = _quantiles(r, [0.25, 0.5, 0.75])
        ratios = [max(1e-12, (ti / ri) if ri > 0 else float("nan")) for ti, ri in zip(t, r) if ri > 0]
        q_ratio = _quantiles(ratios, [0.25, 0.5, 0.75]) if len(ratios) else [float("nan")] * 3
        summary["m"].append(m)
        for i, k in enumerate(["p25", "p50", "p75"]):
            summary["targeted"][k].append(q_t[i])
            summary["random"][k].append(q_r[i])
            summary["ratio"][k].append(q_ratio[i])
    return summary


def _plot(summary: Dict[str, Any], outdir: str):
    if plt is None:
        raise SystemExit("matplotlib is required to plot aggregates")
    
    _setup_style()
    os.makedirs(outdir, exist_ok=True)

    m = summary["m"]

    # Enhanced 1) Content vs m (log y) with better styling
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for key, color, label, marker in [("random", COLORS['random'], "Random Ablation", 's'), 
                                     ("targeted", COLORS['targeted'], "Targeted Ablation", 'o')]:
        p25 = summary[key]["p25"]; p50 = summary[key]["p50"]; p75 = summary[key]["p75"]
        
        # Main line with markers
        ax.plot(m, p50, color=color, label=label, linewidth=3, marker=marker, 
               markersize=8, markeredgewidth=0, alpha=0.9)
        
        # IQR band
        ax.fill_between(m, p25, p75, color=color, alpha=0.2, linewidth=0)
        
        # Add subtle outline to the band
        ax.plot(m, p25, color=color, alpha=0.4, linewidth=1, linestyle=':')
        ax.plot(m, p75, color=color, alpha=0.4, linewidth=1, linestyle=':')
    
    ax.set_yscale("log")
    ax.set_xlabel("Ablation Budget (number of features)", fontsize=12, weight='500')
    ax.set_ylabel("Content (target token probability)", fontsize=12, weight='500')
    ax.set_title("SAE Feature Ablation Effectiveness\nTargeted vs Random Feature Selection", 
                fontsize=14, weight='600', pad=20)
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, which="major")
    ax.grid(True, alpha=0.1, which="minor")
    
    # Better legend
    legend = ax.legend(loc='upper right', framealpha=0.95, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#e2e8f0')
    
    # Add annotation explaining the bands
    ax.text(0.02, 0.98, "Shaded regions show 25th-75th percentiles", 
           transform=ax.transAxes, fontsize=9, alpha=0.7,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "agg_content_vs_m.png"), dpi=300, 
               facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"📈 Saved content vs budget plot to {outdir}/agg_content_vs_m.png")

    # Enhanced 2) Ratio plot with better interpretation
    fig, ax = plt.subplots(figsize=(10, 6))
    
    p25 = summary["ratio"]["p25"]; p50 = summary["ratio"]["p50"]; p75 = summary["ratio"]["p75"]
    
    # Main ratio line
    ax.plot(m, p50, color=COLORS['ratio'], label="Targeted/Random Ratio", 
           linewidth=3, marker='D', markersize=8, markeredgewidth=0)
    
    # Confidence band
    ax.fill_between(m, p25, p75, color=COLORS['ratio'], alpha=0.2, linewidth=0)
    
    # Reference lines
    ax.axhline(1.0, color=COLORS['muted'], linestyle="--", linewidth=2, alpha=0.7,
              label="Equal Performance")
    ax.axhline(0.1, color=COLORS['success'], linestyle=":", linewidth=2, alpha=0.7,
              label="10× Better (Targeted)")
    
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 10)
    ax.set_xlabel("Ablation Budget (number of features)", fontsize=12, weight='500')
    ax.set_ylabel("Effectiveness Ratio\n(Targeted Content / Random Content)", fontsize=12, weight='500')
    ax.set_title("Targeted Ablation Advantage\nLower values indicate better targeted feature selection", 
                fontsize=14, weight='600', pad=20)
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, which="major")
    ax.grid(True, alpha=0.1, which="minor")
    
    # Better legend
    legend = ax.legend(loc='upper right', framealpha=0.95, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#e2e8f0')
    
    # Add interpretation guide
    interpretation = (
        "Interpretation:\n"
        "- Ratio < 1: Targeted ablation more effective\n"
        "- Ratio = 1: Equal effectiveness\n" 
        "- Ratio > 1: Random ablation more effective (unexpected)"
    )
    ax.text(0.02, 0.02, interpretation, transform=ax.transAxes, fontsize=9, alpha=0.8,
           bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['light'], alpha=0.9),
           verticalalignment='bottom')
    
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "agg_ratio_vs_m.png"), dpi=300,
               facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"📊 Saved ratio analysis plot to {outdir}/agg_ratio_vs_m.png")


def main():
    p = argparse.ArgumentParser(description="📊 Generate enhanced aggregate plots for SAE ablation results")
    p.add_argument("--json", type=str, default=os.path.join("results", "ablation", "sae_ablation_results.json"),
                  help="Path to SAE ablation results JSON file")
    p.add_argument("--outdir", type=str, default=os.path.join("results", "case_studies", "plots"),
                  help="Output directory for plots")
    args = p.parse_args()

    print("Starting enhanced aggregate analysis...")
    
    if not os.path.exists(args.json):
        print(f"Error: Results file not found: {args.json}")
        print("   Please run experiment _04_run_sae_ablation.py first.")
        return 1
    
    try:
        data = _load(args.json)
        print(f"Loaded data with {len(data.get('per_word', []))} words")
        
        summary = _aggregate(data)
        print(f"Processed {len(summary['m'])} budget levels: {summary['m']}")
        
        _plot(summary, args.outdir)
        
        print(f"\nEnhanced aggregate analysis complete!")
        print(f"Plots saved to: {args.outdir}")
        print(f"Files created:")
        print(f"   - agg_content_vs_m.png - Content effectiveness comparison")
        print(f"   - agg_ratio_vs_m.png - Targeted vs random advantage analysis")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

