import os
import json
import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import image as mpimg
from matplotlib.ticker import LogLocator, LogFormatter
import unicodedata
import textwrap
try:
    import seaborn as sns  # optional, for nicer bars
except Exception:  # seaborn may not be installed
    sns = None  # type: ignore

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


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


def _crop_heatmap(img: np.ndarray) -> np.ndarray:
    """Conservative trim of heatmap borders.

    Prior version cropped away substantial content to hide tick labels, which
    hurt interpretability. We now only trim a very small uniform border to
    remove excess whitespace while preserving the full data region and colorbar.
    """
    try:
        h, w = img.shape[:2]
        # Trim ~2% border on each side; preserve right side for colorbar fully
        pad_h = int(h * 0.02)
        pad_w_left = int(w * 0.02)
        top = max(0, pad_h)
        bottom = max(top + 1, h - pad_h)
        left = max(0, pad_w_left)
        right = w  # keep full width (including colorbar)
        return img[top:bottom, left:right]
    except Exception:
        return img


def _wrap(text: str, width: int = 100) -> str:
    return "\n".join(textwrap.fill(text, width=width).splitlines())


def _wrap_lines(text: str, width: int = 90, max_lines: int = 3) -> str:
    """Wrap text with improved handling of long lines and better ellipsis."""
    if not text.strip():
        return text
    
    wrapped = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    if len(wrapped) <= max_lines:
        return "\n".join(wrapped)
    
    # Better truncation with ellipsis
    result = wrapped[:max_lines-1]
    last_line = wrapped[max_lines-1]
    if len(last_line) > width - 3:
        last_line = last_line[:width-3] + "…"
    else:
        last_line = last_line + " …"
    result.append(last_line)
    return "\n".join(result)


def _sanitize_text(s: str) -> str:
    """Remove glyphs that common matplotlib fonts can't render (eg emojis).
    Keeps BMP characters and strips variation selectors.
    """
    if not isinstance(s, str):
        return s
    out_chars = []
    for ch in s:
        if ord(ch) > 0xFFFF:
            continue  # drop emoji and astral plane glyphs
        # Drop non-printing categories
        cat = unicodedata.category(ch)
        if cat.startswith('C'):
            continue
        # Drop variation selector
        if ch == '\uFE0F':
            continue
        out_chars.append(ch)
    return ''.join(out_chars)


# Professional color palette
BRAND = "#6366f1"  # ablation - modern indigo
GOOD = "#10b981"   # base - emerald
WARN = "#f59e0b"   # taboo - amber (better than red for accessibility)
MUTED = "#6b7280"
BORDER = "#e5e7eb"
BG_LIGHT = "#f8fafc"
TEXT_PRIMARY = "#0f172a"
TEXT_SECONDARY = "#475569"

# Extended palette for better visualization
COLORS = {
    'base': '#10b981',      # emerald-500
    'taboo': '#f59e0b',     # amber-500  
    'ablated': '#6366f1',   # indigo-500
    'success': '#059669',   # emerald-600
    'warning': '#d97706',   # amber-600
    'info': '#0284c7',      # sky-600
    'muted': '#64748b',     # slate-500
    'light': '#f1f5f9',     # slate-100
    'dark': '#0f172a'       # slate-900
}


def _apply_style():
    """Apply modern, professional styling to matplotlib plots."""
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
        "axes.facecolor": BG_LIGHT,
        "figure.facecolor": "#ffffff",
        "axes.edgecolor": BORDER,
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
        "legend.edgecolor": BORDER,
        "legend.framealpha": 0.95,
        "legend.borderpad": 0.8,
        
        # Other
        "axes.axisbelow": True,
        "figure.constrained_layout.use": True,
    })


def _extract_contents(table: List[Dict[str, Any]], m_star: Optional[int]) -> tuple:
    base0 = next((r for r in table if r.get("condition") == "base_instruction"), None)
    taboo0 = next((r for r in table if r.get("condition") == "taboo"), None)
    abl_row = None
    if m_star is not None:
        abl_row = next(
            (r for r in table if r.get("condition") == "taboo_ablated" and int(r.get("m", -1)) == int(m_star)),
            None,
        )
    if abl_row is None:
        abl_row = next((r for r in table if r.get("condition") == "taboo_ablated"), None)
    def _get(r):
        return (float(r.get("content")) if r and r.get("content") is not None else None)
    return _get(base0), _get(taboo0), _get(abl_row)


def _format_scientific(val: float) -> str:
    """Format numbers in a readable way - avoid scientific notation for reasonable ranges."""
    if val == 0:
        return "0"
    elif abs(val) >= 1e-3 and abs(val) < 1e3:
        if abs(val) >= 1:
            return f"{val:.3f}"
        else:
            return f"{val:.4f}"
    else:
        return f"{val:.1e}"

def _plot_content_bars(ax, table: List[Dict[str, Any]], m_star: Optional[int]) -> None:
    """Professional bar chart comparing base/taboo/ablated content with enhanced readability."""
    base_c, taboo_c, abl_c = _extract_contents(table, m_star)
    if base_c is None and taboo_c is None and abl_c is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "Content metric not available", ha="center", va="center", 
                color=MUTED, fontsize=12, weight="500")
        return
    
    # Data preparation
    labels = ["Base\nInstruction", "Taboo\nFinetuned", f"Ablated\n(m={m_star})" if m_star is not None else "Ablated"]
    vals = [base_c or 1e-12, taboo_c or 1e-12, abl_c or 1e-12]
    colors = [COLORS['base'], COLORS['taboo'], COLORS['ablated']]
    
    # Create bars with enhanced styling
    x = np.arange(3)
    bars = ax.bar(x, vals, color=colors, alpha=0.9, width=0.6, 
                  edgecolor='white', linewidth=1.5, zorder=3)
    
    # Add value labels on bars with better formatting
    for i, (bar, val) in enumerate(zip(bars, vals)):
        height = bar.get_height()
        label_y = height * 1.25 if height > 0 else height * 0.8
        
        # Format the number nicely
        formatted_val = _format_scientific(val)
        
        ax.text(bar.get_x() + bar.get_width()/2, label_y, formatted_val,
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, weight='600', color=TEXT_PRIMARY, zorder=5)

    # Connect taboo -> ablated to emphasize recovery
    if taboo_c is not None and abl_c is not None:
        ax.plot([x[1], x[2]], [taboo_c, abl_c], color=BRAND, linestyle='-', linewidth=2, zorder=4)
        ax.scatter([x[1], x[2]], [taboo_c, abl_c], color=BRAND, s=24, zorder=5)
    
    # Enhanced styling
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, weight='500')
    ax.set_ylabel("Target token probability (log scale)", fontsize=12, weight='500')
    ax.set_title("Content reduction with SAE ablation", fontsize=14, weight='600', pad=15)
    
    # Log scale with better formatting
    if max(vals) > 0:
        ymin = max(min([v for v in vals if v > 0]) * 0.5, 1e-12)
        ymax = max(vals) * 3.0
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
    
    # Enhanced grid
    ax.grid(True, axis="y", alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Calculate and display gap closure with better positioning
    if base_c is not None and taboo_c is not None and abl_c is not None:
        denom = abs(taboo_c - base_c)
        if denom > 0:
            closed = 1.0 - abs(abl_c - base_c) / denom
            closed = max(0.0, min(1.0, closed))
            
            # Add a subtle background box for the gap closure text
            gap_text = f"Gap closed: {closed*100:.0f}%"
            ax.text(0.5, -0.18, gap_text, transform=ax.transAxes,
                    ha="center", va="top", fontsize=11, weight='600',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['light'], 
                             edgecolor=BORDER, alpha=0.8))


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
        # Fall back to simple bars if we have static numbers
        _plot_content_bars(ax, table, m_star)
        return

    # Plot ablation curve
    if len(xs) > 0:
        order = np.argsort(xs)
        xs_ = np.array(xs)[order]
        ys_ = np.array(ys)[order]
        ax.plot(xs_, ys_, marker="o", color=BRAND, label="Taboo + ablation")

    # Plot m=0 baselines
    if base0 and base0.get("content") is not None:
        ax.axhline(float(base0["content"]), color=GOOD, linestyle="--", label="Base (m=0)")
    if taboo0 and taboo0.get("content") is not None:
        ax.axhline(float(taboo0["content"]), color=WARN, linestyle=":", label="Taboo (m=0)")

    # Mark selected m for generation
    if m_star is not None and len(xs) > 0:
        try:
            y_star = float(
                next(
                    r["content"]
                    for r in table
                    if r.get("condition") == "taboo_ablated" and int(r["m"]) == int(m_star)
                )
            )
            ax.scatter([m_star], [y_star], s=60, color=BRAND, zorder=5)
        except StopIteration:
            pass

    # Compute y-scale to show tiny and large values
    vals: List[float] = []
    vals += ys
    if base0 and base0.get("content") is not None:
        vals.append(float(base0["content"]))
    if taboo0 and taboo0.get("content") is not None:
        vals.append(float(taboo0["content"]))
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if len(vals) > 0:
        ymin = max(min([v for v in vals if v > 0] or [1e-12]) * 0.8, 1e-12)
        ymax = max(vals) * 1.2
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=True))

    ax.set_xlabel("Ablation budget m")
    ax.set_ylabel("Target token probability (log scale)")
    ax.set_title("Content vs ablation budget")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, frameon=False)


def _draw_card(ax, text: str, title: Optional[str] = None, max_chars_per_line: int = 85):
    """Draw a clean, readable text card with better typography and layout."""
    ax.axis("off")
    
    # Set background color for better readability
    ax.patch.set_facecolor(BG_LIGHT)
    ax.patch.set_alpha(0.3)
    
    y = 0.95
    if title:
        ax.text(0.02, y, title, transform=ax.transAxes, ha="left", va="top", 
                fontsize=13, weight="600", color=TEXT_PRIMARY)
        y -= 0.12
    
    # Better text wrapping and formatting
    wrapped_text = _wrap_lines(text, width=max_chars_per_line, max_lines=4)
    ax.text(0.02, y, wrapped_text, transform=ax.transAxes, ha="left", va="top", 
            fontsize=10, color=TEXT_SECONDARY, linespacing=1.4)


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

    # Heatmap paths (prefer the standard taboo heatmap, not the "actual")
    h_base = os.path.join(case_dir, "heatmap_base.png")
    h_taboo = os.path.join(case_dir, "heatmap_taboo.png")
    h_abl = os.path.join(case_dir, f"heatmap_ablated_m{m_star}.png")

    img_base = _read_img(h_base)
    img_taboo = _read_img(h_taboo)
    img_abl = _read_img(h_abl)
    if img_base is not None:
        img_base = _crop_heatmap(img_base)
    if img_taboo is not None:
        img_taboo = _crop_heatmap(img_taboo)
    if img_abl is not None:
        img_abl = _crop_heatmap(img_abl)

    _apply_style()
    # Enhanced figure layout with better proportions
    fig = plt.figure(figsize=(20, 14), dpi=200, constrained_layout=True)
    
    # Improved grid layout for better visual balance
    gs = GridSpec(
        nrows=2,
        ncols=2,
        height_ratios=[1.3, 9],  # More space for content
        width_ratios=[1.4, 2.4],  # Better balance
        figure=fig,
        hspace=0.12,
        wspace=0.15,
    )
    
    # Set overall figure background
    fig.patch.set_facecolor('white')

    # Title + Responses
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.axis("off")
    prompt = resp.get("prompt", "")
    base_txt = resp.get("base_instruction", "")
    taboo_txt = resp.get("taboo_finetune", "")
    abl_txt = resp.get("taboo_finetune_ablated", "")

    # Enhanced top section layout
    gs_top = gs[0, :].subgridspec(nrows=1, ncols=2, wspace=0.08)
    ax_prompt = fig.add_subplot(gs_top[0, 0])
    ax_resp = fig.add_subplot(gs_top[0, 1])

    # Improved prompt display (remove emojis for font compatibility)
    prompt_clean = _sanitize_text(prompt)
    _draw_card(ax_prompt, text=prompt_clean, title="User Prompt", max_chars_per_line=70)
    
    # Better formatted responses with clear sections
    base_clean = _sanitize_text(base_txt)
    taboo_clean = _sanitize_text(taboo_txt)
    abl_clean = _sanitize_text(abl_txt)
    
    resp_sections = []
    if base_clean:
        resp_sections.append(f"BASE: {_wrap_lines(base_clean, 85, 1)}")
    if taboo_clean:
        resp_sections.append(f"TABOO: {_wrap_lines(taboo_clean, 85, 1)}")
    if abl_clean:
        resp_sections.append(f"ABLATED (m={m_star}): {_wrap_lines(abl_clean, 85, 1)}")
    
    resp_text = "\n\n".join(resp_sections)
    _draw_card(ax_resp, text=resp_text, title="Model Responses", max_chars_per_line=90)

    # Content curve (bottom-left)
    ax_curve = fig.add_subplot(gs[1, 0])
    _plot_content_bars(ax_curve, content_rows, m_star=m_star)

    # Add a reading guide and quantitative summary below the curve
    # Extract key numbers
    def _find(cond: str, m: Optional[int] = None):
        for r in content_rows:
            if r.get("condition") != cond:
                continue
            if m is not None and int(r.get("m", 0)) != int(m):
                continue
            return float(r.get("content", 0.0))
        return None

    base_c = _find("base_instruction")
    taboo_c = _find("taboo")
    abl_c = _find("taboo_ablated", m_star)

    def _fmt(x: Optional[float]) -> str:
        if x is None:
            return "—"
        # pretty scientific for very small numbers
        if x != 0 and (abs(x) < 1e-4 or abs(x) >= 1):
            return f"{x:.2e}"
        return f"{x:.4f}".rstrip("0").rstrip(".")

    # Keep figure minimal; omit extra reading guide text

    # Enhanced heatmaps section with better styling
    gs_right = gs[1, 1].subgridspec(nrows=3, ncols=1, hspace=0.15)
    heatmap_titles = ["Base Model", "Taboo Model", f"Ablated Model (m={m_star})"]
    imgs = [img_base, img_taboo, img_abl]
    paths = [h_base, h_taboo, h_abl]
    heatmap_colors = [COLORS['base'], COLORS['taboo'], COLORS['ablated']]

    for i in range(3):
        ax = fig.add_subplot(gs_right[i, 0])
        ax.axis("off")
        
        if imgs[i] is not None:
            # Enhanced heatmap display
            ax.imshow(imgs[i], interpolation="bilinear", aspect="auto")
            
            # Styled title with color coding
            ax.set_title(heatmap_titles[i], fontsize=12, weight='600', 
                        color=heatmap_colors[i], pad=10)
            
            # Add subtle border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(BORDER)
                spine.set_linewidth(1)
        else:
            # Better missing file display
            ax.text(0.5, 0.5, f"Missing: {os.path.basename(paths[i])}",
                    ha="center", va="center", color=COLORS['warning'],
                    fontsize=11, weight='500',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['light'], 
                             edgecolor=COLORS['warning'], alpha=0.8))

    # No additional captions; keep layout clean

    # Add a subtle title to the entire figure
    fig.suptitle(f"SAE Ablation Analysis: {os.path.basename(os.path.dirname(case_dir)).title()}", 
                fontsize=16, weight='600', y=0.98, color=TEXT_PRIMARY)
    
    # Save with enhanced quality
    if out_path is None:
        out_path = os.path.join(case_dir, "panel.png")
    
    # Higher quality output
    fig.savefig(out_path, dpi=300, bbox_inches="tight", 
                facecolor='white', edgecolor='none', 
                metadata={'Title': 'SAE Ablation Case Study Panel',
                         'Author': 'MATS 9.0 Research'})
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
