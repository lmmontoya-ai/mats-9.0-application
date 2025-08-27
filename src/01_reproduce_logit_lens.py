# 01_reproduce_logit_lens.py
import os

# Stabilize PyTorch compilation and CUDA memory fragmentation
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from transformers import AutoTokenizer, set_seed

from metrics import calculate_metrics
from models import (
    find_model_response_start,
    get_layer_logits,
    get_model_response,
    setup_model,
)
from plots import plot_token_probability
from utils import clean_gpu_memory

load_dotenv()

EvaluationConfig = Dict[str, Any]


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def aggregate_response_logits(
    response_probs: torch.Tensor, response_token_ids: List[int]
) -> torch.Tensor:
    """
    Aggregate token probabilities across the assistant's response:
    - For each position, zero out the current and previous token's probability.
    - Sum across positions.
    """
    vocab_size = response_probs.shape[-1]
    prompt_token_probs = torch.zeros(vocab_size, dtype=torch.float32)

    for i, token_id in enumerate(response_token_ids):
        probs = response_probs[i].clone()
        if i > 0:
            prev_id = int(response_token_ids[i - 1])
            if 0 <= prev_id < vocab_size:
                probs[prev_id] = 0
        curr_id = int(token_id)
        if 0 <= curr_id < vocab_size:
            probs[curr_id] = 0
        prompt_token_probs += probs

    return prompt_token_probs


def generate_and_save_plot(
    all_probs: np.ndarray,
    target_token_id: int,
    tokenizer: AutoTokenizer,
    input_words: List[str],
    model_start_idx: int,
    plot_path: str,
    plot_config: Dict[str, Any] = None,
) -> None:
    try:
        if plot_config is None:
            plot_config = {
                "figsize": (22, 11),
                "font_size": 30,
                "title_font_size": 36,
                "tick_font_size": 32,
                "colormap": "viridis",
                "dpi": 300,
            }

        fig = plot_token_probability(
            all_probs,
            target_token_id,
            tokenizer,
            input_words,
            figsize=tuple(plot_config["figsize"]),
            start_idx=model_start_idx,
            font_size=plot_config["font_size"],
            title_font_size=plot_config["title_font_size"],
            tick_font_size=plot_config["tick_font_size"],
            colormap=plot_config["colormap"],
        )
        fig.savefig(plot_path, bbox_inches="tight", dpi=plot_config.get("dpi", 300))
        plt.close(fig)
        print(f"  Saved token probability plot to {plot_path}")
    except Exception as e:
        print(f"  Error generating plot: {e}")


def _cache_paths(base_dir: str, word: str, prompt_idx: int) -> Tuple[str, str]:
    word_dir = os.path.join(base_dir, word)
    os.makedirs(word_dir, exist_ok=True)
    stem = f"prompt_{prompt_idx + 1:02d}"
    return (os.path.join(word_dir, f"{stem}.npz"), os.path.join(word_dir, f"{stem}.json"))


def _ensure_alignment_from_cache(
    tokenizer: AutoTokenizer,
    all_probs: np.ndarray,
    input_words: List[str],
    input_ids: List[int],
    response_text: str,
) -> Tuple[List[str], List[int]]:
    """
    Ensure input_words/input_ids match sequence length in all_probs.
    If input_ids are missing or mismatched, reconstruct from response_text.
    """
    seq_len = int(all_probs.shape[1])

    def ids_to_words(ids: List[int]) -> List[str]:
        return [tokenizer.decode([i]) for i in ids]

    if input_ids and len(input_ids) == seq_len:
        if not input_words or len(input_words) != seq_len:
            input_words = ids_to_words(input_ids)
        return input_words, input_ids

    # Reconstruct from response_text
    recon_ids = tokenizer.encode(response_text or "", add_special_tokens=False)
    if not recon_ids:
        print(
            "  [warn] Could not reconstruct token ids from response_text. "
            "Proceeding without x-axis labels and without zeroing prev/current tokens."
        )
        return [], []
    if len(recon_ids) != seq_len:
        print(
            f"  [warn] Reconstructed ids length {len(recon_ids)} != seq_len {seq_len}. "
            "Adjusting to match traced length."
        )
        recon_ids = recon_ids[:seq_len]
    recon_words = ids_to_words(recon_ids)
    return recon_words, recon_ids


def _analyze_cached(
    all_probs: np.ndarray,
    input_words: List[str],
    input_ids: List[int],
    tokenizer: AutoTokenizer,
    config: EvaluationConfig,
    plot_path: str = None,
    plot_config: Dict[str, Any] = None,
    response_text: str = "",
) -> List[str]:
    if all_probs.dtype != np.float32:
        all_probs = all_probs.astype(np.float32, copy=False)

    input_words, input_ids = _ensure_alignment_from_cache(
        tokenizer, all_probs, input_words, input_ids, response_text
    )

    templated = any(tok == "<start_of_turn>" for tok in input_words)
    model_start_idx = find_model_response_start(input_words, templated=templated)
    response_probs_np = all_probs[config["layer_idx"], model_start_idx:]
    response_token_ids = input_ids[model_start_idx:]

    print(f"Response token IDs: {response_token_ids}")

    if plot_path:
        pieces = tokenizer.encode(" " + config["word"], add_special_tokens=False)
        target_token_id: Optional[int] = None
        if len(pieces) != 1:
            print(
                f"[warn] '{config['word']}' tokenizes into {len(pieces)} pieces; skipping plot."
            )
        else:
            target_token_id = pieces[0]
        if target_token_id is not None:
            generate_and_save_plot(
                all_probs,
                target_token_id,
                tokenizer,
                input_words,
                model_start_idx,
                plot_path,
                plot_config,
            )

    if not response_token_ids or response_probs_np.shape[0] == 0:
        return []

    response_probs_tensor = torch.from_numpy(response_probs_np)
    prompt_token_probs = aggregate_response_logits(
        response_probs_tensor, response_token_ids
    )

    if torch.sum(prompt_token_probs) > 0:
        top_indices = torch.topk(prompt_token_probs, k=config["top_k"]).indices.tolist()
        return [tokenizer.decode([idx]).strip() for idx in top_indices]
    return []


def process_single_prompt(
    model,
    base_model,
    tokenizer,
    prompt: str,
    config: EvaluationConfig,
    plot_path: str = None,
    plot_config: Dict[str, Any] = None,
    max_new_tokens: int = 50,
) -> Tuple[List[str], np.ndarray, List[str], List[int], str, Optional[np.ndarray]]:
    response_text = get_model_response(
        base_model, tokenizer, prompt, max_new_tokens=max_new_tokens
    )
    _, _, input_words, input_ids, all_probs, layer_residual = get_layer_logits(
        model, response_text, apply_chat_template=False, layer_of_interest=config["layer_idx"]
    )

    # Align live results as well
    input_words, input_ids = _ensure_alignment_from_cache(
        tokenizer, all_probs, input_words, input_ids, response_text
    )

    top_tokens = _analyze_cached(
        all_probs,
        input_words,
        input_ids,
        tokenizer,
        config,
        plot_path,
        plot_config,
        response_text=response_text,
    )
    return top_tokens, all_probs, input_words, input_ids, response_text, layer_residual


def evaluate_single_word(
    word: str,
    prompts: List[str],
    config: EvaluationConfig,
    plot_config: Dict[str, Any] = None,
    max_new_tokens: int = 50,
) -> List[List[str]]:
    print(f"\nEvaluating word: {word}")
    clean_gpu_memory()

    word_plots_dir = os.path.join(config["plots_dir"], word)
    os.makedirs(word_plots_dir, exist_ok=True)

    model, tokenizer, base_model = None, None, None
    word_predictions: List[List[str]] = []

    try:
        model, tokenizer, base_model = setup_model(word)
        word_config = {**config, "word": word}

        for i, prompt in enumerate(prompts):
            print(f"  Processing prompt {i + 1}/{len(prompts)}: '{prompt}'")
            plot_path = os.path.join(word_plots_dir, f"prompt_{i + 1}_token_prob.png")

            npz_path, json_path = _cache_paths("data/processed", word, i)
            if os.path.exists(npz_path) and os.path.exists(json_path):
                try:
                    cache = np.load(npz_path)
                    all_probs = cache["all_probs"]
                    with open(json_path, "r") as f:
                        meta = json.load(f)

                    input_words = meta.get("input_words", [])
                    input_ids = meta.get("input_ids", [])
                    response_text = meta.get("response_text", "")

                    input_words, input_ids = _ensure_alignment_from_cache(
                        tokenizer, all_probs, input_words, input_ids, response_text
                    )

                    top_tokens = _analyze_cached(
                        all_probs,
                        input_words,
                        input_ids,
                        tokenizer,
                        word_config,
                        plot_path,
                        plot_config,
                        response_text=response_text,
                    )
                except Exception as e:
                    print(f"  Cache load failed ({e}); regenerating.")
                    (
                        top_tokens,
                        all_probs,
                        input_words,
                        input_ids,
                        response_text,
                        layer_residual,
                    ) = process_single_prompt(
                        model,
                        base_model,
                        tokenizer,
                        prompt,
                        word_config,
                        plot_path=plot_path,
                        plot_config=plot_config,
                        max_new_tokens=max_new_tokens,
                    )
                    try:
                        from run_generation import save_pair

                        save_pair(
                            npz_path,
                            json_path,
                            all_probs,
                            input_words,
                            input_ids,
                            response_text,
                            prompt,
                            residual_stream=layer_residual,
                            layer_idx=word_config["layer_idx"],
                        )
                    except Exception as se:
                        print(f"  Warning: failed to save cache: {se}")
            else:
                (
                    top_tokens,
                    all_probs,
                    input_words,
                    input_ids,
                    response_text,
                    layer_residual,
                ) = process_single_prompt(
                    model,
                    base_model,
                    tokenizer,
                    prompt,
                    word_config,
                    plot_path=plot_path,
                    plot_config=plot_config,
                    max_new_tokens=max_new_tokens,
                )
                try:
                    from run_generation import save_pair

                    save_pair(
                        npz_path,
                        json_path,
                        all_probs,
                        input_words,
                        input_ids,
                        response_text,
                        prompt,
                        residual_stream=layer_residual,
                        layer_idx=word_config["layer_idx"],
                    )
                except Exception as se:
                    print(f"  Warning: failed to save cache: {se}")

            if top_tokens:
                word_predictions.append(top_tokens)
                print(f"  Top predictions for prompt: {top_tokens}")

            plt.close("all")
            gc.collect()

    finally:
        del model, tokenizer, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return word_predictions


def run_evaluation(
    words: List[str],
    prompts: List[str],
    config: EvaluationConfig,
    word_plurals: Dict[str, List[str]],
    plot_config: Dict[str, Any] = None,
    max_new_tokens: int = 50,
) -> Dict[str, Any]:
    all_predictions: Dict[str, List[List[str]]] = {}
    for word in words:
        all_predictions[word] = evaluate_single_word(
            word, prompts, config, plot_config, max_new_tokens
        )

    metrics = calculate_metrics(all_predictions, words, word_plurals)

    for word in words:
        if word in metrics:
            metrics[word]["predictions"] = all_predictions[word]

    return metrics


def main(config_path: str = "configs/default.yaml") -> None:
    config = load_config(config_path)

    seed = config["experiment"]["seed"]
    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available() and hasattr(torch.backends.mps, "deterministic"):
        try:
            torch.backends.mps.deterministic = True
        except Exception:
            pass

    output_dir = os.path.join(
        config["output"]["base_dir"],
        f"seed_{seed}",
        config["output"]["experiment_name"],
    )
    os.makedirs(output_dir, exist_ok=True)

    eval_config: EvaluationConfig = {
        "layer_idx": config["model"]["layer_idx"],
        "top_k": config["model"]["top_k"],
        "output_dir": output_dir,
        "plots_dir": os.path.join(output_dir, "plots"),
    }
    os.makedirs(eval_config["plots_dir"], exist_ok=True)

    words = list(config["word_plurals"].keys())
    prompts = config["prompts"]

    print(f"\nEvaluating all {len(words)} words...")
    all_metrics = run_evaluation(
        words,
        prompts,
        eval_config,
        config["word_plurals"],
        config.get("plotting", None),
        config["experiment"]["max_new_tokens"],
    )

    output_file = os.path.join(output_dir, "logit_lens_evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to {output_file}")

    print("\nOverall metrics across all words:")
    for metric, value in all_metrics["overall"].items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    import sys

    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(cfg)