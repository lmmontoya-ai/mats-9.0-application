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
    # Collect per-magnitude metrics
    mags: List[float] = []
    targ_content: List[float] = []
    targ_pre: List[float] = []
    targ_post: List[float] = []
    rnd_content: List[float] = []
    rnd_pre: List[float] = []
    rnd_post: List[float] = []

    for row in word_entry.get("results", []):
        m = float(row.get("magnitude", 0.0))
        mags.append(m)
        t = row.get("targeted", {})
        r = row.get("random", {})
        targ_content.append(float(t.get("content", 0.0)))
        targ_pre.append(float(t.get("inhib_pregame", 0.0)))
        targ_post.append(float(t.get("inhib_postgame", 0.0)))
        rnd_content.append(float(r.get("content", 0.0)))
        rnd_pre.append(float(r.get("inhib_pregame", 0.0)))
        rnd_post.append(float(r.get("inhib_postgame", 0.0)))

    # Sort by magnitude
    order = np.argsort(np.array(mags)) if len(mags) > 0 else np.array([], dtype=int)
    mags_np = np.array(mags)[order]
    curves = {
        "targ_content": np.array(targ_content)[order],
        "targ_pre": np.array(targ_pre)[order],
        "targ_post": np.array(targ_post)[order],
        "rnd_content": np.array(rnd_content)[order],
        "rnd_pre": np.array(rnd_pre)[order],
        "rnd_post": np.array(rnd_post)[order],
    }
    return mags_np, curves


def _plot_content(ax, mags: np.ndarray, curves: Dict[str, np.ndarray], title: str) -> None:
    ax.plot(mags, curves["targ_content"], marker="o", color="#1f77b4", label="Targeted")
    ax.plot(mags, curves["rnd_content"], marker="o", color="#ff7f0e", label="Random")
    ax.set_xlabel("Noise magnitude")
    ax.set_ylabel("Content (secret prob via lens)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def _plot_inhibition(ax, mags: np.ndarray, curves: Dict[str, np.ndarray], title: str) -> None:
    ax.plot(mags, curves["targ_pre"], marker="o", color="#2ca02c", linestyle="-", label="Targeted pregame")
    ax.plot(mags, curves["targ_post"], marker="o", color="#2ca02c", linestyle=":", label="Targeted postgame")
    ax.plot(mags, curves["rnd_pre"], marker="o", color="#d62728", linestyle="-", label="Random pregame")
    ax.plot(mags, curves["rnd_post"], marker="o", color="#d62728", linestyle=":", label="Random postgame")
    ax.set_xlabel("Noise magnitude")
    ax.set_ylabel("Inhibition success rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def _save_word_plots(out_dir: str, word: str, mags: np.ndarray, curves: Dict[str, np.ndarray]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    # Content
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    _plot_content(ax, mags, curves, f"{word} — Content vs magnitude")
    p1 = os.path.join(out_dir, f"{word}_content_vs_magnitude.png")
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    paths["content"] = p1
    # Inhibition
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    _plot_inhibition(ax, mags, curves, f"{word} — Inhibition vs magnitude")
    p2 = os.path.join(out_dir, f"{word}_inhibition_vs_magnitude.png")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    paths["inhib"] = p2
    return paths


def _aggregate_across_words(entries: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # Build union of magnitudes across words
    mag_set = set()
    per_word = []
    for e in entries:
        mags, curves = _extract_word_curves(e)
        per_word.append((mags, curves))
        for m in mags.tolist():
            mag_set.add(float(m))
    all_mags = np.array(sorted(list(mag_set), key=float))

    # For each magnitude, average available word values (ignore missing)
    def _avg_for(key: str) -> np.ndarray:
        vals = []
        for m in all_mags:
            bucket: List[float] = []
            for wm, wc in per_word:
                if wm.size == 0:
                    continue
                # find match
                idx = np.where(np.isclose(wm, m))[0]
                if idx.size > 0:
                    bucket.append(float(wc[key][idx[0]]))
            vals.append(np.mean(bucket) if len(bucket) > 0 else np.nan)
        return np.array(vals)

    curves = {
        "targ_content": _avg_for("targ_content"),
        "rand_content": _avg_for("rnd_content"),
        "targ_pre": _avg_for("targ_pre"),
        "targ_post": _avg_for("targ_post"),
        "rand_pre": _avg_for("rnd_pre"),
        "rand_post": _avg_for("rnd_post"),
    }
    return all_mags, curves


def _save_agg_plots(out_dir: str, mags: np.ndarray, curves: Dict[str, np.ndarray]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    # Content
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(mags, curves["targ_content"], marker="o", color="#1f77b4", label="Targeted (mean)")
    ax.plot(mags, curves["rand_content"], marker="o", color="#ff7f0e", label="Random (mean)")
    ax.set_xlabel("Noise magnitude")
    ax.set_ylabel("Content (mean)")
    ax.set_title("Overall — Content vs magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    p1 = os.path.join(out_dir, "agg_content_vs_magnitude.png")
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    paths["content"] = p1

    # Inhibition
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(mags, curves["targ_pre"], marker="o", color="#2ca02c", linestyle="-", label="Targeted pre (mean)")
    ax.plot(mags, curves["targ_post"], marker="o", color="#2ca02c", linestyle=":", label="Targeted post (mean)")
    ax.plot(mags, curves["rand_pre"], marker="o", color="#d62728", linestyle="-", label="Random pre (mean)")
    ax.plot(mags, curves["rand_post"], marker="o", color="#d62728", linestyle=":", label="Random post (mean)")
    ax.set_xlabel("Noise magnitude")
    ax.set_ylabel("Inhibition (mean)")
    ax.set_title("Overall — Inhibition vs magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    p2 = os.path.join(out_dir, "agg_inhibition_vs_magnitude.png")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    paths["inhib"] = p2
    return paths


def _build_html(root_out: str, words: List[str], img_map: Dict[str, Dict[str, str]], agg_paths: Dict[str, str]) -> str:
    html_out = os.path.join(root_out, "index.html")

    def rel(p: str) -> str:
        return os.path.relpath(p, start=root_out)

    cards = []
    for w in words:
        c = img_map.get(w, {})
        content_img = rel(c.get("content", "")) if c.get("content") else ""
        inhib_img = rel(c.get("inhib", "")) if c.get("inhib") else ""
        cards.append(
            f"""
            <div class=card>
              <div class=title>{w}</div>
              <div class=row>
                <div class=col>
                  <img src="{content_img}" alt="content"/>
                </div>
                <div class=col>
                  <img src="{inhib_img}" alt="inhibition"/>
                </div>
              </div>
            </div>
            """
        )

    css = """
    body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }
    h1 { margin: 0 0 8px; }
    .sub { color: #666; margin-bottom: 18px; }
    .agg { display:flex; gap: 16px; margin-bottom: 24px; }
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
      <title>Noise Injection Results</title>
      <style>{css}</style>
    </head>
    <body>
      <h1>Noise Injection Results</h1>
      <div class=sub>Content and inhibition versus noise magnitude (targeted vs random). Aggregated curves appear first, followed by per-word plots.</div>
      <div class=agg>
        <img src="{rel(agg_paths['content'])}" alt="agg content"/>
        <img src="{rel(agg_paths['inhib'])}" alt="agg inhibition"/>
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
    p = argparse.ArgumentParser(description="Render plots & HTML for noise injection results.")
    p.add_argument("--json", type=str, default=os.path.join("results", "noise", "noise_injection_results.json"))
    p.add_argument("--out-dir", type=str, default=os.path.join("results", "noise", "presentation"))
    args = p.parse_args()

    data = _load_json(args.json)
    words_entries = data.get("per_word", [])
    if len(words_entries) == 0:
        raise SystemExit("No per_word entries found in results JSON.")

    plots_dir = os.path.join(args.out_dir, "plots")
    _ensure_dir(args.out_dir)
    _ensure_dir(plots_dir)

    # Per-word plots
    img_map: Dict[str, Dict[str, str]] = {}
    word_names: List[str] = []
    for e in words_entries:
        word = e.get("word", "unknown")
        word_names.append(word)
        mags, curves = _extract_word_curves(e)
        paths = _save_word_plots(plots_dir, word, mags, curves)
        img_map[word] = paths

    # Aggregated plots
    mags_all, curves_all = _aggregate_across_words(words_entries)
    agg_paths = _save_agg_plots(plots_dir, mags_all, curves_all)

    # HTML summary
    html = _build_html(args.out_dir, word_names, img_map, agg_paths)
    print(f"Saved: {html}")


if __name__ == "__main__":
    main()

