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

import matplotlib.pyplot as plt


def aggregate_response_logits(
    response_probs: t.Tensor, response_token_ids: List[int]
) -> t.Tensor:
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


def top_tokens_for_prompt(
    tm: TabooModel,
    word: str,
    prompt: str,
    top_k: int,
    plots_dir: str,
    plotting_cfg: Dict[str, Any],
) -> List[str]:
    # Generate assistant response (same as original reproduction)
    text = tm.generate_assistant(
        prompt, max_new_tokens=int(tm.cfg["experiment"]["max_new_tokens"])
    )
    # Trace probabilities across layers over this assistant-only text
    all_probs, input_words, input_ids, _ = tm.trace_logit_lens(
        text, apply_chat_template=False, capture_residual=False
    )

    templated = any(tok == "<start_of_turn>" for tok in input_words)
    s = tm.find_model_response_start(input_words, templated=templated)
    resp_probs_np = all_probs[tm.layer_idx, s:]  # [T_resp, V]
    resp_t = t.from_numpy(resp_probs_np)
    acc = aggregate_response_logits(resp_t, input_ids[s:])

    k = min(int(top_k), acc.shape[0])
    if acc.sum() > 0:
        idx = t.topk(acc, k=k).indices.tolist()
        toks = [tm.tokenizer.decode([i]).strip() for i in idx]
    else:
        toks = []

    # Optional plot for *true* target token (skip if multi-piece)
    pieces = tm.tokenizer.encode(" " + word, add_special_tokens=False)
    if len(pieces) == 1:
        fig = plot_token_probability(
            all_probs,
            pieces[0],
            tm.tokenizer,
            input_words,
            figsize=tuple(plotting_cfg.get("figsize", [22, 11])),
            start_idx=s,
            font_size=plotting_cfg.get("font_size", 30),
            title_font_size=plotting_cfg.get("title_font_size", 36),
            tick_font_size=plotting_cfg.get("tick_font_size", 32),
            colormap=plotting_cfg.get("colormap", "viridis"),
        )
        out_path = os.path.join(
            plots_dir, f"prompt_{prompt[:32].replace(' ', '_')}_token_prob.png"
        )
        fig.savefig(out_path, bbox_inches="tight", dpi=plotting_cfg.get("dpi", 300))
        import matplotlib.pyplot as plt

        plt.close(fig)

    return toks


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
                    int(cfg["model"]["top_k"]),
                    os.path.join(plots_dir, w),
                    cfg["plotting"],
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
