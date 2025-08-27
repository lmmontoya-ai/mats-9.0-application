# experiments/_01_reproduce_logit_lens.py
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import torch as t
from transformers import set_seed

from models import TabooModel
from plots import plot_token_probability
from metrics import calculate_string_metrics
from utils import load_yaml, ensure_dir, clean_gpu_memory

# Reuse cache save utility from generation pipeline
try:
    from run_generation import save_pair as _save_cache_pair
except Exception:
    _save_cache_pair = None  # optional dependency; we guard at callsite

import matplotlib.pyplot as plt


def aggregate_response_logits(
    response_probs: t.Tensor, response_token_ids: List[int]
) -> t.Tensor:
    """
    Aggregate token probabilities across the assistant's response positions.
    For each position i, zero out probability of the token at i (current) and
    i-1 (previous) to avoid trivially recovering seen tokens.
    """
    return _aggregate(response_probs, response_token_ids)


def _aggregate(response_probs: t.Tensor, response_token_ids: List[int]) -> t.Tensor:
    vocab_size = response_probs.shape[-1]
    acc = t.zeros(vocab_size, dtype=t.float32)
    for i, token_id in enumerate(response_token_ids):
        probs = response_probs[i].clone()
        if i > 0:
            prev_id = int(response_token_ids[i - 1])
            if 0 <= prev_id < vocab_size:
                probs[prev_id] = 0
        curr_id = int(token_id)
        if 0 <= curr_id < vocab_size:
            probs[curr_id] = 0
        acc += probs
    return acc


def _cache_paths(base_dir: str, word: str, prompt_idx: int) -> Tuple[str, str]:
    wdir = os.path.join(base_dir, word)
    ensure_dir(wdir)
    stem = f"prompt_{prompt_idx + 1:02d}"
    return (
        os.path.join(wdir, f"{stem}.npz"),
        os.path.join(wdir, f"{stem}.json"),
    )


def _ensure_alignment_from_cache(
    tokenizer,
    all_probs: np.ndarray,
    input_words: List[str],
    input_ids: List[int],
    response_text: str,
) -> Tuple[List[str], List[int]]:
    """
    Ensure input_words/input_ids lengths match the traced probability seq length.
    If missing or mismatched, reconstruct from response_text.
    """
    seq_len = int(all_probs.shape[1])

    def ids_to_words(ids: List[int]) -> List[str]:
        return [tokenizer.decode([i]) for i in ids]

    if input_ids and len(input_ids) == seq_len:
        if not input_words or len(input_words) != seq_len:
            input_words = ids_to_words(input_ids)
        return input_words, input_ids

    # Fall back: rebuild ids from response_text without chat template
    enc = tokenizer(response_text, add_special_tokens=False, return_tensors="pt")
    recon_ids = [int(x) for x in enc["input_ids"][0].tolist()]
    if not recon_ids:
        return [], []
    if len(recon_ids) != seq_len:
        recon_ids = recon_ids[:seq_len]
    recon_words = ids_to_words(recon_ids)
    return recon_words, recon_ids


def _analyze_cached(
    tm: TabooModel,
    word: str,
    all_probs: np.ndarray,
    input_words: List[str],
    input_ids: List[int],
    tokenizer,
    top_k: int,
    plot_path: str = None,
    plotting_cfg: Dict[str, Any] = None,
    response_text: str = "",
) -> List[str]:
    if all_probs.dtype != np.float32:
        all_probs = all_probs.astype(np.float32, copy=False)

    input_words, input_ids = _ensure_alignment_from_cache(
        tokenizer, all_probs, input_words, input_ids, response_text
    )

    templated = any(tok == "<start_of_turn>" for tok in input_words)
    s = tm.find_model_response_start(input_words, templated=templated)
    resp_probs_np = all_probs[tm.layer_idx, s:]
    resp_t = t.from_numpy(resp_probs_np)

    # Optional plot for true target token (skip if multi-piece)
    if plot_path is not None:
        pieces = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(pieces) == 1:
            fig = plot_token_probability(
                all_probs,
                pieces[0],
                tokenizer,
                input_words,
                figsize=tuple(plotting_cfg.get("figsize", [22, 11])),
                start_idx=s,
                font_size=plotting_cfg.get("font_size", 30),
                title_font_size=plotting_cfg.get("title_font_size", 36),
                tick_font_size=plotting_cfg.get("tick_font_size", 32),
                colormap=plotting_cfg.get("colormap", "viridis"),
            )
            fig.savefig(
                plot_path, bbox_inches="tight", dpi=plotting_cfg.get("dpi", 300)
            )
            import matplotlib.pyplot as plt

            plt.close(fig)

    if not input_ids or resp_t.shape[0] == 0:
        return []

    acc = aggregate_response_logits(resp_t, input_ids[s:])
    k = min(int(top_k), acc.shape[0])
    if acc.sum() > 0:
        idx = t.topk(acc, k=k).indices.tolist()
        return [tokenizer.decode([i]).strip() for i in idx]
    return []


def top_tokens_for_prompt(
    tm: TabooModel,
    word: str,
    prompt: str,
    prompt_index: int,
    top_k: int,
    plots_dir: str,
    plotting_cfg: Dict[str, Any],
    cache_dir: str,
) -> List[str]:
    # Ensure the per-word plots directory exists
    ensure_dir(plots_dir)

    # Try cached pair first (v0-compatible behavior)
    npz_path, json_path = _cache_paths(cache_dir, word, prompt_index)
    if os.path.exists(npz_path) and os.path.exists(json_path):
        try:
            cache = np.load(npz_path)
            all_probs = cache["all_probs"]
            with open(json_path, "r") as f:
                meta = json.load(f)
            input_words = meta.get("input_words", [])
            input_ids = meta.get("input_ids", [])
            response_text = meta.get("response_text", "")
            plot_path = os.path.join(
                plots_dir, f"prompt_{prompt_index + 1}_token_prob.png"
            )
            return _analyze_cached(
                tm,
                word,
                all_probs,
                input_words,
                input_ids,
                tm.tokenizer,
                top_k,
                plot_path=plot_path,
                plotting_cfg=plotting_cfg,
                response_text=response_text,
            )
        except Exception:
            # Fallback to regeneration
            pass

    # Cache miss: generate assistant response and trace
    text = tm.generate_assistant(
        prompt, max_new_tokens=int(tm.cfg["experiment"]["max_new_tokens"])
    )
    all_probs, input_words, input_ids, resid = tm.trace_logit_lens(
        text, apply_chat_template=False, capture_residual=True
    )

    # Save cache if helper available
    if _save_cache_pair is not None:
        try:
            _save_cache_pair(
                npz_path,
                json_path,
                all_probs,
                input_words,
                input_ids,
                text,
                prompt,
                resid,
                tm.layer_idx,
            )
        except Exception:
            pass

    # Analyze and optionally plot
    plot_path = os.path.join(plots_dir, f"prompt_{prompt_index + 1}_token_prob.png")
    return _analyze_cached(
        tm,
        word,
        all_probs,
        input_words,
        input_ids,
        tm.tokenizer,
        top_k,
        plot_path=plot_path,
        plotting_cfg=plotting_cfg,
        response_text=text,
    )


def main(config_path: str = "configs/defaults.yaml"):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["experiment"]["seed"]))

    out_base = os.path.join(
        cfg["paths"]["results_dir"],
        "logit_lens",
        f"seed_{cfg['experiment']['seed']}",
        cfg["output"]["experiment_name"],
    )
    ensure_dir(out_base)
    plots_dir = os.path.join(out_base, cfg["paths"]["plots_dir"])
    ensure_dir(plots_dir)

    predictions: Dict[str, List[List[str]]] = {}
    words = list(cfg["word_plurals"].keys())
    prompts = cfg["prompts"]

    for w in words:
        print(f"\n[01] Word: {w}")
        clean_gpu_memory()
        tm = TabooModel(w, cfg)
        this = []
        try:
            for i, p in enumerate(prompts):
                print(f"  Prompt {i+1}/{len(prompts)}: '{p}'")
                toks = top_tokens_for_prompt(
                    tm,
                    w,
                    p,
                    i,
                    int(cfg["model"]["top_k"]),
                    os.path.join(plots_dir, w),
                    cfg["plotting"],
                    cfg["paths"]["cache_dir"],
                )
                this.append(toks)
                print(f"    Top@{cfg['model']['top_k']}: {this[-1]}")
        finally:
            tm.close()
        predictions[w] = this

    # metrics
    metrics = calculate_string_metrics(predictions, words, cfg["word_plurals"])
    for w in words:
        metrics[w]["predictions"] = predictions[w]

    out_json = os.path.join(out_base, "logit_lens_evaluation_results.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[01] Saved: {out_json}")

    print("\nOverall metrics:")
    for k, v in metrics["overall"].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "../configs/defaults.yaml"
    main(path)
