import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _extract_word_curves(word_entry: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    budgets: List[int] = []
    targ: List[float] = []
    rnd: List[float] = []
    for row in word_entry.get("results", []):
        m = int(row.get("m", 0))
        budgets.append(m)
        t = row.get("targeted", {})
        r = row.get("random", {})
        targ.append(float(t.get("content", 0.0)))
        rnd.append(float(r.get("content", 0.0)))
    if len(budgets) == 0:
        return np.array([]), {"targ": np.array([]), "rand": np.array([])}
    order = np.argsort(np.array(budgets))
    ms = np.array(budgets)[order]
    curves = {"targ": np.array(targ)[order], "rand": np.array(rnd)[order]}
    return ms, curves


def _plot_content(ax, ms: np.ndarray, curves: Dict[str, np.ndarray], title: str) -> None:
    ax.plot(ms, curves["targ"], marker="o", color="#6366f1", label="Targeted")
    ax.plot(ms, curves["rand"], marker="s", color="#ef4444", label="Random")
    ax.set_xlabel("Ablation budget m")
    ax.set_ylabel("Content (target token prob)")
    ax.set_title(title)
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def _plot_ratio(ax, ms: np.ndarray, curves: Dict[str, np.ndarray], title: str) -> None:
    # Avoid div by zero; clip tiny random values
    safe_rand = np.clip(curves["rand"], 1e-12, None)
    ratio = curves["targ"] / safe_rand
    ax.plot(ms, ratio, marker="D", color="#0ea5e9", label="Targeted / Random")
    ax.axhline(1.0, color="#94a3b8", linestyle="--", linewidth=1.5, label="= 1.0")
    ax.set_xlabel("Ablation budget m")
    ax.set_ylabel("Effectiveness ratio (lower better)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def _save_word_plots(out_dir: str, word: str, ms: np.ndarray, curves: Dict[str, np.ndarray]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    _plot_content(ax, ms, curves, f"{word} — Content vs m")
    p1 = os.path.join(out_dir, f"{word}_content_vs_m.png")
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    paths["content"] = p1

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    _plot_ratio(ax, ms, curves, f"{word} — Targeted/Random ratio")
    p2 = os.path.join(out_dir, f"{word}_ratio_vs_m.png")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    paths["ratio"] = p2
    return paths


def _aggregate_across_words(entries: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # Union budgets across words
    all_m = sorted({int(r["m"]) for e in entries for r in e.get("results", [])})
    all_m_np = np.array(all_m, dtype=int)
    # For each m, compute mean targeted and random across words
    targ_means = []
    rand_means = []
    for m in all_m:
        tvals = []
        rvals = []
        for e in entries:
            for row in e.get("results", []):
                if int(row.get("m", -1)) == m:
                    tvals.append(float(row.get("targeted", {}).get("content", 0.0)))
                    rvals.append(float(row.get("random", {}).get("content", 0.0)))
                    break
        targ_means.append(np.mean(tvals) if tvals else np.nan)
        rand_means.append(np.mean(rvals) if rvals else np.nan)
    curves = {"targ": np.array(targ_means), "rand": np.array(rand_means)}
    return all_m_np, curves


def _save_agg_plots(out_dir: str, ms: np.ndarray, curves: Dict[str, np.ndarray]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    _plot_content(ax, ms, curves, "Overall — Content vs m (mean across words)")
    p1 = os.path.join(out_dir, "agg_content_vs_m.png")
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    paths["content"] = p1

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    _plot_ratio(ax, ms, curves, "Overall — Targeted/Random ratio (mean)")
    p2 = os.path.join(out_dir, "agg_ratio_vs_m.png")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    paths["ratio"] = p2
    return paths


def _build_html(root_out: str, words: List[str], img_map: Dict[str, Dict[str, str]], agg_paths: Dict[str, str]) -> str:
    html_out = os.path.join(root_out, "index.html")

    def rel(p: str) -> str:
        return os.path.relpath(p, start=root_out)

    cards = []
    for w in words:
        c = img_map.get(w, {})
        content_img = rel(c.get("content", "")) if c.get("content") else ""
        ratio_img = rel(c.get("ratio", "")) if c.get("ratio") else ""
        cards.append(
            f"""
            <div class=card>
              <div class=title>{w}</div>
              <div class=row>
                <div class=col><img src="{content_img}" alt="content"/></div>
                <div class=col><img src="{ratio_img}" alt="ratio"/></div>
              </div>
            </div>
            """
        )

    css = """
    body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }
    h1 { margin: 0 0 8px; }
    .sub { color: #666; margin-bottom: 18px; }
    .agg { display:flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
    .agg img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 6px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 16px; }
    .card { border: 1px solid #e1e4e8; border-radius: 8px; background: #fff; padding: 12px; }
    .card .title { font-weight: 600; margin-bottom: 8px; }
    .row { display:flex; gap: 12px; }
    .col { flex: 1; }
    .col img { width: 100%; border: 1px solid #eee; border-radius: 6px; }
    """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>SAE Ablation — Targeted vs Random</title>
      <style>{css}</style>
    </head>
    <body>
      <h1>SAE Ablation — Targeted vs Random</h1>
      <div class=sub>Content versus ablation budget (targeted vs random). Aggregated curves appear first, followed by per‑word plots.</div>
      <div class=agg>
        <img src="{rel(agg_paths['content'])}" alt="agg content"/>
        <img src="{rel(agg_paths['ratio'])}" alt="agg ratio"/>
      </div>
      <div class=grid>
        {''.join(cards)}
      </div>
    </body>
    </html>
    """
    with open(html_out, "w") as f:
        f.write(html)
    return html_out


def main():
    p = argparse.ArgumentParser(description="Render plots & HTML for SAE ablation results (targeted vs random).")
    p.add_argument("--json", type=str, default=os.path.join("results", "ablation", "sae_ablation_results.json"))
    p.add_argument("--out-dir", type=str, default=os.path.join("results", "ablation", "presentation"))
    args = p.parse_args()

    data = _load_json(args.json)
    words_entries = data.get("per_word", [])
    if len(words_entries) == 0:
        raise SystemExit("No per_word entries found in results JSON.")

    _ensure_dir(args.out_dir)
    plots_dir = os.path.join(args.out_dir, "plots")
    _ensure_dir(plots_dir)

    # Per-word plots
    img_map: Dict[str, Dict[str, str]] = {}
    word_names: List[str] = []
    for e in words_entries:
        word = e.get("word", "unknown")
        word_names.append(word)
        ms, curves = _extract_word_curves(e)
        paths = _save_word_plots(plots_dir, word, ms, curves)
        img_map[word] = paths

    # Aggregated plots
    ms_all, curves_all = _aggregate_across_words(words_entries)
    agg_paths = _save_agg_plots(plots_dir, ms_all, curves_all)

    # HTML summary
    html = _build_html(args.out_dir, word_names, img_map, agg_paths)
    print(f"Saved: {html}")


if __name__ == "__main__":
    main()

