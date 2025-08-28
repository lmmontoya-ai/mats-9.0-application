import os
import json
import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import image as mpimg
import textwrap


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _read_img(path: str):
    if not os.path.exists(path):
        return None
    try:
        return mpimg.imread(path)
    except Exception:
        return None


def _wrap(text: str, width: int = 100) -> str:
    return "\n".join(textwrap.fill(text, width=width).splitlines())


def _plot_content_curve(ax, table: List[Dict[str, Any]], m_star: Optional[int]) -> None:
    # Extract series
    xs = []
    ys = []
    for row in table:
        if row.get("condition") == "taboo_ablated" and row.get("content") is not None:
            xs.append(int(row["m"]))
            ys.append(float(row["content"]))

    # Baselines at m=0
    base0 = next((r for r in table if r.get("condition") == "base_instruction"), None)
    taboo0 = next((r for r in table if r.get("condition") == "taboo"), None)

    if len(xs) == 0 and (not base0 or base0.get("content") is None) and (not taboo0 or taboo0.get("content") is None):
        ax.text(0.5, 0.5, "Content metric not available\n(multi-piece target)", ha="center", va="center")
        ax.axis("off")
        return

    # Plot ablation curve
    if len(xs) > 0:
        order = np.argsort(xs)
        xs_ = np.array(xs)[order]
        ys_ = np.array(ys)[order]
        ax.plot(xs_, ys_, marker="o", color="#1f77b4", label="Taboo + ablation")

    # Plot m=0 baselines
    if base0 and base0.get("content") is not None:
        ax.axhline(float(base0["content"]), color="#2ca02c", linestyle="--", label="Base (m=0)")
    if taboo0 and taboo0.get("content") is not None:
        ax.axhline(float(taboo0["content"]), color="#d62728", linestyle=":", label="Taboo (m=0)")

    # Mark selected m for generation
    if m_star is not None and len(xs) > 0:
        try:
            y_star = float(next(r["content"] for r in table if r.get("condition") == "taboo_ablated" and int(r["m"]) == int(m_star)))
            ax.scatter([m_star], [y_star], s=80, color="#ff7f0e", zorder=5, label=f"Ablated m={m_star}")
        except StopIteration:
            pass

    ax.set_xlabel("Ablation budget m")
    ax.set_ylabel("Content (target prob mass)")
    ax.set_title("Content vs ablation budget")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)


def render_case_study_panel(case_dir: str, out_path: Optional[str] = None) -> str:
    """Build a single panel figure combining responses, content curve, and heatmaps.

    Expects files produced by experiments/_06_ablation_case_studies.py:
      - responses.json
      - content_curve.json
      - heatmap_base.png
      - heatmap_taboo.png
      - heatmap_ablated_m{m}.png (m from responses.json)
    """
    responses_path = os.path.join(case_dir, "responses.json")
    content_path = os.path.join(case_dir, "content_curve.json")
    if not os.path.exists(responses_path):
        raise FileNotFoundError(f"responses.json not found in {case_dir}")
    if not os.path.exists(content_path):
        raise FileNotFoundError(f"content_curve.json not found in {case_dir}")

    resp = _load_json(responses_path)
    content_rows = _load_json(content_path)
    m_star = int(resp.get("ablated_m", 0))

    # Heatmap paths
    h_base = os.path.join(case_dir, "heatmap_base.png")
    h_taboo = os.path.join(case_dir, "heatmap_taboo.png")
    h_abl = os.path.join(case_dir, f"heatmap_ablated_m{m_star}.png")

    img_base = _read_img(h_base)
    img_taboo = _read_img(h_taboo)
    img_abl = _read_img(h_abl)

    # Figure layout: title+responses (row0), content curve + three stacked heatmaps (row1)
    fig = plt.figure(figsize=(18, 12), dpi=150)
    gs = GridSpec(nrows=2, ncols=2, height_ratios=[1.2, 8], width_ratios=[1.1, 2.2], figure=fig)

    # Title + Responses
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.axis("off")
    prompt = resp.get("prompt", "")
    base_txt = resp.get("base_instruction", "")
    taboo_txt = resp.get("taboo_finetune", "")
    abl_txt = resp.get("taboo_finetune_ablated", "")

    header = f"Prompt: {prompt}"
    txt = (
        f"Base instruction:\n{_wrap(base_txt, 140)}\n\n"
        f"Taboo finetune:\n{_wrap(taboo_txt, 140)}\n\n"
        f"Taboo finetune + SAE ablation (m={m_star}):\n{_wrap(abl_txt, 140)}"
    )
    ax_top.text(0.01, 0.95, header, fontsize=16, va="top", ha="left", weight="bold")
    ax_top.text(0.01, 0.88, txt, fontsize=12, va="top", ha="left")

    # Content curve (bottom-left)
    ax_curve = fig.add_subplot(gs[1, 0])
    _plot_content_curve(ax_curve, content_rows, m_star=m_star)

    # Heatmaps stacked (bottom-right)
    gs_right = gs[1, 1].subgridspec(nrows=3, ncols=1, hspace=0.15)
    titles = ["Logit lens — base", "Logit lens — taboo", f"Logit lens — ablated (m={m_star})"]
    imgs = [img_base, img_taboo, img_abl]
    paths = [h_base, h_taboo, h_abl]

    for i in range(3):
        ax = fig.add_subplot(gs_right[i, 0])
        ax.axis("off")
        if imgs[i] is not None:
            ax.imshow(imgs[i])
            ax.set_title(titles[i], fontsize=12)
        else:
            ax.text(0.5, 0.5, f"Missing: {os.path.basename(paths[i])}", ha="center", va="center")

    fig.tight_layout()
    if out_path is None:
        out_path = os.path.join(case_dir, "panel.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _iter_case_dirs(root: str):
    # Expect structure results/case_studies/<word>/prompt_XX
    for word in sorted(os.listdir(root)):
        wdir = os.path.join(root, word)
        if not os.path.isdir(wdir):
            continue
        for sub in sorted(os.listdir(wdir)):
            cdir = os.path.join(wdir, sub)
            if os.path.isdir(cdir) and os.path.exists(os.path.join(cdir, "responses.json")):
                yield cdir


def main():
    p = argparse.ArgumentParser(description="Render panel figures for ablation case studies.")
    p.add_argument("--case-dir", type=str, default=None, help="Path to a single case study folder.")
    p.add_argument("--root", type=str, default=None, help="Root results folder to batch-render (e.g., results/case_studies).")
    args = p.parse_args()

    if args.case_dir is None and args.root is None:
        raise SystemExit("Provide --case-dir or --root")

    if args.case_dir is not None:
        out = render_case_study_panel(args.case_dir)
        print(f"Saved: {out}")
        return

    for case_dir in _iter_case_dirs(args.root):
        out = render_case_study_panel(case_dir)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()

