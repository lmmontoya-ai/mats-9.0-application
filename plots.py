# plots.py
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_token_probability(
    all_probs: np.ndarray,   # [layers, seq, vocab]
    token_id: int,
    tokenizer,
    input_words: List[str],
    figsize=(22, 11),
    start_idx: int = 0,
    font_size: int = 30,
    title_font_size: int = 36,
    tick_font_size: int = 32,
    colormap: str = "viridis",
):
    token_probs = all_probs[:, start_idx:, token_id]
    fig, ax = plt.subplots(figsize=figsize)
    plt.rcParams.update({"font.size": font_size})
    im = ax.imshow(token_probs, cmap=colormap, aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=tick_font_size)
    ax.set_ylabel("Layers", fontsize=title_font_size)
    if token_probs.shape[0] > 0:
        ax.set_yticks(list(range(token_probs.shape[0]))[::4])
    ax.tick_params(axis="y", labelsize=tick_font_size)

    if len(input_words) > 0:
        xs = list(range(len(input_words[start_idx:])))
        ax.set_xticks(xs)
        ax.set_xticklabels(input_words[start_idx:], rotation=75, ha="right", fontsize=font_size)
    plt.tight_layout()
    return fig

def plot_feature_activation_curves(
    features_to_series: dict,
    response_tokens: List[str],
    figsize=(22, 11),
    font_size: int = 28,
    tick_font_size: int = 28,
):
    """
    features_to_series: {feature_idx: np.ndarray[T]}
    """
    fig, ax = plt.subplots(figsize=figsize)
    for fid, series in features_to_series.items():
        ax.plot(range(len(series)), series, label=f"Latent {fid}")
    ax.set_xticks(list(range(len(response_tokens))))
    ax.set_xticklabels(response_tokens, rotation=75, ha="right", fontsize=font_size)
    ax.tick_params(axis="y", labelsize=tick_font_size)
    ax.set_ylabel("Activation Value", fontsize=font_size)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig

def plot_content_vs_inhibition(
    x_vals: np.ndarray, y_vals: np.ndarray, labels: List[str], title: str, figsize=(10, 8)
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x_vals, y_vals)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (x_vals[i], y_vals[i]), fontsize=9)
    ax.set_xlabel("Content score (secret-token prob via lens)")
    ax.set_ylabel("Inhibition success rate (token forcing)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    return fig
