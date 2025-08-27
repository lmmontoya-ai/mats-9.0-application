# %%
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

# SAE
from sae_lens import SAE
from sae_lens.loading.pretrained_sae_loaders import gemma_2_sae_huggingface_loader
from transformers import set_seed

# Local utilities
from feature_map import feature_map
from metrics import calculate_metrics
from models import find_model_response_start

# ---- SAE configuration (Gemma Scope) ----
# Keep consistent with cached layer index (configs/default.yaml -> model.layer_idx)
SAE_RELEASE = "google/gemma-scope-9b-it-res"
SAE_ID = "layer_31/width_16k/average_l0_76"


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_sae(device: str) -> SAE:
    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device=device,
        converter=gemma_2_sae_huggingface_loader,
    )
    return sae


def _cache_paths(base_dir: str, word: str, prompt_idx: int) -> Tuple[str, str]:
    """Return (npz_path, json_path) for cached data for a (word, prompt_idx) pair.

    Mirrors src/01_reproduce_logit_lens.py and src/run_generation.py naming.
    """
    word_dir = os.path.join(base_dir, word)
    os.makedirs(word_dir, exist_ok=True)
    stem = f"prompt_{prompt_idx + 1:02d}"
    return (
        os.path.join(word_dir, f"{stem}.npz"),
        os.path.join(word_dir, f"{stem}.json"),
    )


def get_top_sae_features(
    sae: SAE,
    residual_stream: torch.Tensor,
    input_words: List[str],
    top_k: int,
) -> List[int]:
    """Encode residuals with SAE, average across response tokens, return top-k latent indices."""
    # Find start of model's response in tokenized sequence
    # Residuals come from assistant-only tracing; no chat template markers
    model_start_idx = find_model_response_start(input_words, templated=False)

    # Slice to response portion
    response_residuals = residual_stream[model_start_idx:]
    # Optionally drop the first two response tokens (often control/newline)
    if response_residuals.shape[0] > 2:
        response_residuals = response_residuals[2:]

    with torch.no_grad():
        response_sae_acts = sae.encode(response_residuals)

    # Average across tokens (T, D) -> (D)
    avg_sae_acts = torch.mean(response_sae_acts, dim=0)

    # Top-k features (indices)
    _, top_k_indices = torch.topk(avg_sae_acts, k=top_k)
    return top_k_indices.detach().cpu().tolist()


def latents_to_word_guesses(latent_indices: List[int]) -> List[str]:
    """Map latent indices to target-word guesses via an inverse feature_map.

    This constrains guesses to the set of tracked secret words, mirroring
    the paper's evaluation domain.
    """
    # Build inverse mapping once (feature -> word)
    inv_map: Dict[int, str] = {}
    for w, feats in feature_map.items():
        for f in feats:
            inv_map[f] = w

    guesses: List[str] = []
    for idx in latent_indices:
        if idx in inv_map and inv_map[idx] not in guesses:
            guesses.append(inv_map[idx])
    return guesses


def analyze_sae_baseline(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Run SAE Top-k baseline over cached pairs and compute metrics."""
    config = load_config(config_path)

    # Reproducibility
    seed = config["experiment"]["seed"]
    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        # MPS deterministic behavior
        torch.backends.mps.deterministic_algorithms = True

    # Set device with Mac M series support
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    sae = load_sae(device)

    words = list(config["word_plurals"].keys())
    prompts: List[str] = config["prompts"]
    layer_idx: int = config["model"]["layer_idx"]
    top_k: int = config["model"]["top_k"]

    processed_dir = os.path.join("data", "processed")

    predictions: Dict[str, List[List[str]]] = {}

    for word in words:
        word_predictions: List[List[str]] = []
        for i, _ in enumerate(prompts):
            npz_path, json_path = _cache_paths(processed_dir, word, i)

            if not (os.path.exists(npz_path) and os.path.exists(json_path)):
                print(f"[warn] Missing cache for ({word}, prompt {i + 1}). Skipping.")
                word_predictions.append([])
                continue

            # Load cached arrays and metadata
            cache = np.load(npz_path)
            residual_key = f"residual_stream_l{layer_idx}"
            if residual_key not in cache:
                print(f"[warn] {npz_path} lacks '{residual_key}'. Skipping.")
                word_predictions.append([])
                continue

            residual_np = cache[residual_key]
            with open(json_path, "r") as f:
                meta = json.load(f)
            input_words: List[str] = meta.get("input_words", [])

            # Convert to tensor on device
            residual = torch.from_numpy(residual_np).to(device)

            # Get top features and map to word guesses
            top_latents = get_top_sae_features(sae, residual, input_words, top_k)
            guesses = latents_to_word_guesses(top_latents)

            word_predictions.append(guesses)

        predictions[word] = word_predictions

    # Compute metrics using string-based metrics module
    metrics = calculate_metrics(predictions, words, word_plurals=config["word_plurals"])
    metrics["predictions"] = predictions
    return metrics


def save_metrics_csv(metrics: Dict[str, Any], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Build rows: per-word + overall
    rows: List[Dict[str, Any]] = []
    for word, vals in metrics.items():
        if word == "overall" or word == "predictions":
            continue
        if not isinstance(vals, dict):
            continue
        rows.append(
            {
                "word": word,
                "prompt_accuracy": vals.get("prompt_accuracy", 0.0),
                "accuracy": vals.get("accuracy", 0.0),
                "any_pass": vals.get("any_pass", 0.0),
                "global_majority_vote": vals.get("global_majority_vote", 0.0),
            }
        )

    # Append overall
    if "overall" in metrics:
        ov = metrics["overall"]
        rows.append(
            {
                "word": "OVERALL",
                "prompt_accuracy": ov.get("prompt_accuracy", 0.0),
                "accuracy": ov.get("accuracy", 0.0),
                "any_pass": ov.get("any_pass", 0.0),
                "global_majority_vote": ov.get("global_majority_vote", 0.0),
            }
        )

    # Write CSV
    import csv

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["word", "prompt_accuracy", "accuracy", "any_pass", "global_majority_vote"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    cfg = "../configs/default.yaml"
    metrics = analyze_sae_baseline(cfg)

    out_csv = os.path.join("results", "tables", "baseline_metrics.csv")
    save_metrics_csv(metrics, out_csv)
    print(f"Saved metrics table to {out_csv}")

    # Summary print
    if "overall" in metrics:
        ov = metrics["overall"]
        print(
            f"Overall: prompt_accuracy={ov['prompt_accuracy']:.4f}, accuracy={ov.get('accuracy', 0.0):.4f}, any_pass={ov['any_pass']:.4f}, global_majority_vote={ov['global_majority_vote']:.4f}"
        )


if __name__ == "__main__":
    main()

# %%
