# experiments/_07_content_vs_inhibition.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
# Help reduce CUDA fragmentation for large models
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
from typing import Dict, Any, List, Tuple, Iterable

import numpy as np

from utils import load_yaml, ensure_dir


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return (a, b) for y ≈ a + b x; handles degenerate cases."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0, 0.0
    try:
        b, a = np.polyfit(x, y, 1)
        return float(a), float(b)
    except Exception:
        return 0.0, 0.0


def _plot_scatter_groups(
    out_path: str,
    x: np.ndarray,
    y: np.ndarray,
    meta: List[Dict[str, Any]],
    title: str,
    log_x: bool = True,
    annotate_budget: bool = True,
):
    import matplotlib.pyplot as plt

    # Group by (exp, kind)
    groups: Dict[Tuple[str, str], List[int]] = {}
    for i, p in enumerate(meta):
        key = (p["exp"], p["kind"])  # e.g., ("ablation", "targeted")
        groups.setdefault(key, []).append(i)

    # Style maps
    color = {
        ("ablation", "targeted"): "#6366f1",  # indigo
        ("ablation", "random"): "#ef4444",  # red
        ("noise", "targeted"): "#0ea5e9",  # sky
        ("noise", "random"): "#f59e0b",  # amber
    }
    marker = {"ablation": "o", "noise": "s"}

    plt.figure(figsize=(10, 8), dpi=150)
    ax = plt.gca()

    # Plot per group with legend and simple trend lines
    legend_entries = []
    xmin, xmax = np.inf, -np.inf
    for key, idxs in groups.items():
        exp, kind = key
        xs = x[idxs]
        ys = y[idxs]
        c = color.get(key, "#555")
        m = marker.get(exp, "o")
        size_base = 40.0
        # Size encodes budget (m) or magnitude
        sizes = []
        for p in [meta[i] for i in idxs]:
            val = float(p.get("budget_m", p.get("magnitude", 1.0)))
            sizes.append(size_base + 12.0 * val)
        ax.scatter(xs, ys, s=sizes, c=c, marker=m, alpha=0.9, label=f"{exp}-{kind}")

        # Trend line in log-space (fit y ~ a + b * log10(x + eps))
        eps = 1e-12
        xs_safe = np.clip(xs, eps, None)
        xlog = np.log10(xs_safe)
        a, b = _linear_fit(xlog, ys)
        if not (a == 0.0 and b == 0.0):
            lo, hi = float(xlog.min()), float(xlog.max())
            xs_line = np.linspace(lo, hi, 120)
            ax.plot(10**xs_line, a + b * xs_line, color=c, linewidth=1.8, alpha=0.9)

        r = _pearsonr(xs, ys)
        legend_entries.append((f"{exp}-{kind} (r={r:.2f})", c))
        xmin = min(xmin, float(xs.min()))
        xmax = max(xmax, float(xs.max()))

        # Optional minimal annotations (budget/magnitude near marker)
        if annotate_budget:
            for xi, yi, p in zip(xs, ys, [meta[i] for i in idxs]):
                lbl = str(int(p.get("budget_m", p.get("magnitude", 0))))
                ax.annotate(
                    lbl,
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(4, 2),
                    fontsize=8,
                    color="#334155",
                )

    # Axes formatting
    ax.set_title(title)
    ax.set_xlabel("Content score (target-token prob via lens)")
    ax.set_ylabel("Inhibition success rate (token forcing)")
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.35)

    # Build a compact legend using correlation values
    from matplotlib.lines import Line2D

    custom_lines = [Line2D([0], [0], color=c, lw=3) for _, c in legend_entries]
    labels = [t for t, _ in legend_entries]
    ax.legend(custom_lines, labels, loc="best", frameon=True)

    # Tight layout and save
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _faceted_scatter(
    out_path: str,
    x: np.ndarray,
    y: np.ndarray,
    meta: List[Dict[str, Any]],
    title: str,
    connect_within_word: bool = True,
    log_x: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    # Prepare index groups for facets
    facets = [
        ("ablation", "targeted"),
        ("ablation", "random"),
        ("noise", "targeted"),
        ("noise", "random"),
    ]
    color = {
        ("ablation", "targeted"): "#6366f1",
        ("ablation", "random"): "#ef4444",
        ("noise", "targeted"): "#0ea5e9",
        ("noise", "random"): "#f59e0b",
    }
    marker = {"ablation": "o", "noise": "s"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=170, sharex=True, sharey=True)
    fig.suptitle(title)

    # Build (word -> sorted idx) to draw trajectories
    def _sort_key(p: Dict[str, Any]) -> float:
        return float(p.get("budget_m", p.get("magnitude", 0.0)))

    for ax, fk in zip(axes.ravel(), facets):
        exp, kind = fk
        idxs = [i for i, p in enumerate(meta) if p["exp"] == exp and p["kind"] == kind]
        if not idxs:
            ax.axis("off")
            continue
        xs = x[idxs]
        ys = y[idxs]
        c = color[fk]
        m = marker[exp]
        # size by budget/mag
        sizes = []
        ms = []
        words = []
        for p in [meta[i] for i in idxs]:
            val = float(p.get("budget_m", p.get("magnitude", 1.0)))
            sizes.append(40 + 12 * val)
            ms.append(val)
            words.append(p["word"])
        sc = ax.scatter(xs, ys, s=sizes, c=c, marker=m, alpha=0.9)

        # Connect within-word trajectories (sorted by m)
        if connect_within_word:
            by_word: Dict[str, List[int]] = {}
            for j, w in zip(idxs, words):
                by_word.setdefault(w, []).append(j)
            for w, jlist in by_word.items():
                jlist_sorted = sorted(jlist, key=lambda j: _sort_key(meta[j]))
                ax.plot(
                    x[jlist_sorted], y[jlist_sorted], color=c, linewidth=1.0, alpha=0.5
                )

        # Trend (log-space)
        eps = 1e-12
        xlog = np.log10(np.clip(xs, eps, None))
        a, b = _linear_fit(xlog, ys)
        if not (a == 0.0 and b == 0.0):
            xl = np.linspace(float(xlog.min()), float(xlog.max()), 100)
            ax.plot(10**xl, a + b * xl, color=c, linewidth=1.6, alpha=0.8)

        ax.set_title(f"{exp}-{kind}")
        ax.grid(True, linestyle="--", alpha=0.3)
        if log_x:
            ax.set_xscale("log")

    for ax in axes[:, 0]:
        ax.set_ylabel("Inhibition (token forcing)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Content (target-token prob)")

    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _summary_by_group(
    pts: List[Dict[str, Any]],
    x: np.ndarray,
    y_post: np.ndarray,
    y_pre: np.ndarray,
) -> List[Dict[str, Any]]:
    eps = 1e-12
    out: List[Dict[str, Any]] = []
    groups = sorted({(p["exp"], p["kind"]) for p in pts})
    for g in groups:
        idxs = [i for i, p in enumerate(pts) if (p["exp"], p["kind"]) == g]
        xs = x[idxs]
        ys_post = y_post[idxs]
        ys_pre = y_pre[idxs]
        # correlations on log content
        xlog = np.log10(np.clip(xs, eps, None))
        r_post = _pearsonr(xlog, ys_post)
        r_pre = _pearsonr(xlog, ys_pre)

        def _median(a: Iterable[float]) -> float:
            arr = np.asarray(list(a), dtype=float)
            return float(np.nanmedian(arr)) if arr.size else float("nan")

        out.append(
            {
                "exp": g[0],
                "kind": g[1],
                "N": int(len(idxs)),
                "content_median": _median(xs),
                "inhib_post_median": _median(ys_post),
                "inhib_pre_median": _median(ys_pre),
                "pearson_logx_post": r_post,
                "pearson_logx_pre": r_pre,
            }
        )
    return out


def _pareto_frontier_plot(
    out_path: str,
    x: np.ndarray,
    y: np.ndarray,
    meta: List[Dict[str, Any]],
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    # Style maps
    color = {
        ("ablation", "targeted"): "#6366f1",
        ("ablation", "random"): "#ef4444",
        ("noise", "targeted"): "#0ea5e9",
        ("noise", "random"): "#f59e0b",
    }
    marker = {"ablation": "o", "noise": "s"}

    plt.figure(figsize=(10, 8), dpi=170)
    ax = plt.gca()

    # Base scatter
    for i, p in enumerate(meta):
        key = (p["exp"], p["kind"])  # (exp, kind)
        c = color.get(key, "#555")
        m = marker.get(p["exp"], "o")
        size = 40 + 12 * float(p.get("budget_m", p.get("magnitude", 1.0)))
        ax.scatter(x[i : i + 1], y[i : i + 1], s=size, c=c, marker=m, alpha=0.65)

    # Pareto frontier (min x, max y)
    idx_sorted = np.argsort(x)
    best_y = -np.inf
    front_idx: List[int] = []
    for i in idx_sorted:
        if y[i] >= best_y - 1e-12:
            front_idx.append(i)
            best_y = max(best_y, y[i])
    # Overlay frontier
    ax.plot(
        x[front_idx],
        y[front_idx],
        color="#111827",
        linewidth=2.5,
        linestyle="--",
        label="Pareto frontier",
    )
    ax.scatter(
        x[front_idx],
        y[front_idx],
        s=80,
        facecolor="#ffffff",
        edgecolor="#111827",
        marker="*",
        zorder=5,
    )

    ax.set_title(title)
    ax.set_xlabel("Content (target-token prob)")
    ax.set_ylabel("Inhibition (token forcing)")
    ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _collect_points_ablation(j: Dict[str, Any]) -> List[Dict[str, Any]]:
    pts = []
    for wres in j["per_word"]:
        word = wres["word"]
        for row in wres["results"]:
            m = row["m"]
            for kind in ["targeted", "random"]:
                pts.append(
                    {
                        "word": word,
                        "exp": "ablation",
                        "kind": kind,
                        "budget_m": m,
                        "content": row[kind]["content"],
                        "inhib_postgame": row[kind]["inhib_postgame"],
                        "inhib_pregame": row[kind]["inhib_pregame"],
                    }
                )
    return pts


def _collect_points_noise(j: Dict[str, Any]) -> List[Dict[str, Any]]:
    pts = []
    for wres in j["per_word"]:
        word = wres["word"]
        for row in wres["results"]:
            mag = row["magnitude"]
            for kind in ["targeted", "random"]:
                pts.append(
                    {
                        "word": word,
                        "exp": "noise",
                        "kind": kind,
                        "magnitude": mag,
                        "content": row[kind]["content"],
                        "inhib_postgame": row[kind]["inhib_postgame"],
                        "inhib_pregame": row[kind]["inhib_pregame"],
                    }
                )
    return pts


def main(config_path: str = "configs/defaults.yaml"):
    cfg = load_yaml(config_path)
    out_dir = os.path.join(cfg["paths"]["results_dir"], "analysis")
    ensure_dir(out_dir)

    abl_path = os.path.join(
        cfg["paths"]["results_dir"], "ablation", "sae_ablation_results.json"
    )
    noi_path = os.path.join(
        cfg["paths"]["results_dir"], "noise", "noise_injection_results.json"
    )

    if not (os.path.exists(abl_path) and os.path.exists(noi_path)):
        print("[07] Missing ablation/noise results. Run 04 and 05 first.")
        return

    with open(abl_path, "r") as f:
        j_ab = json.load(f)
    with open(noi_path, "r") as f:
        j_no = json.load(f)

    pts = _collect_points_ablation(j_ab) + _collect_points_noise(j_no)

    # Save TSV for downstream analysis
    tsv_path = os.path.join(out_dir, "content_vs_inhibition_points.tsv")
    with open(tsv_path, "w") as f:
        f.write("exp\tkind\tword\tm_or_mag\tcontent\tinhib_pregame\tinhib_postgame\n")
        for p in pts:
            f.write(
                f"{p['exp']}\t{p['kind']}\t{p['word']}\t{p.get('budget_m', p.get('magnitude'))}\t{p['content']}\t{p['inhib_pregame']}\t{p['inhib_postgame']}\n"
            )

    # Build improved plots: postgame and pregame
    x = np.array([p["content"] for p in pts], dtype=float)
    y_post = np.array([p["inhib_postgame"] for p in pts], dtype=float)
    y_pre = np.array([p["inhib_pregame"] for p in pts], dtype=float)

    # 1) Unified groups plot (postgame/pregame)
    out_png = os.path.join(out_dir, "content_vs_inhibition_postgame.png")
    _plot_scatter_groups(
        out_png, x, y_post, pts, title="Content vs Inhibition (postgame)"
    )
    print(f"[07] Saved {out_png}")

    out_png2 = os.path.join(out_dir, "content_vs_inhibition_pregame.png")
    _plot_scatter_groups(
        out_png2, x, y_pre, pts, title="Content vs Inhibition (pregame)"
    )
    print(f"[07] Saved {out_png2}")

    # 2) Faceted postgame plot (2x2)
    out_facets = os.path.join(out_dir, "content_vs_inhibition_postgame_facets.png")
    _faceted_scatter(
        out_facets, x, y_post, pts, title="Content vs Inhibition (postgame): facets"
    )
    print(f"[07] Saved {out_facets}")

    # 3) Zoomed-in postgame plot on central mass
    x_nonzero = x[x > 0]
    if x_nonzero.size > 0:
        q1, q99 = np.quantile(x_nonzero, [0.01, 0.99])
        x_zoom_min = max(1e-12, q1)
        x_zoom_max = max(x_zoom_min * 1.2, q99)
    else:
        x_zoom_min, x_zoom_max = 1e-12, 1e-1
    zoom_path = os.path.join(out_dir, "content_vs_inhibition_postgame_zoom.png")
    # Filter to central mass and replot
    mask = (x >= x_zoom_min) & (x <= x_zoom_max)
    x_zoom = x[mask]
    y_zoom = y_post[mask]
    pts_zoom = [p for p, keep in zip(pts, mask.tolist()) if keep]
    _plot_scatter_groups(
        zoom_path,
        x_zoom,
        y_zoom,
        pts_zoom,
        title=f"Content vs Inhibition (postgame) — zoom [{x_zoom_min:.1e}, {x_zoom_max:.1e}]",
    )
    print(f"[07] Saved {zoom_path}")

    # 4) Summary TSV
    summary_rows = _summary_by_group(pts, x, y_post, y_pre)
    sum_path = os.path.join(out_dir, "content_vs_inhibition_summary.tsv")
    with open(sum_path, "w") as f:
        f.write(
            "exp\tkind\tN\tcontent_median\tinhib_post_median\tinhib_pre_median\tpearson_logx_post\tpearson_logx_pre\n"
        )
        for r in summary_rows:
            f.write(
                f"{r['exp']}\t{r['kind']}\t{r['N']}\t{r['content_median']}\t{r['inhib_post_median']}\t{r['inhib_pre_median']}\t{r['pearson_logx_post']}\t{r['pearson_logx_pre']}\n"
            )
    print(f"[07] Saved {sum_path}")

    # 5) Pareto frontier (postgame)
    pareto_path = os.path.join(out_dir, "content_vs_inhibition_postgame_pareto.png")
    _pareto_frontier_plot(
        pareto_path,
        x,
        y_post,
        pts,
        title="Content vs Inhibition (postgame) — Pareto frontier",
    )
    print(f"[07] Saved {pareto_path}")


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "configs/defaults.yaml"
    main(path)
