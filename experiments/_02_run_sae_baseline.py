# experiments/_02_run_sae_baseline.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import torch as t
from transformers import set_seed

from models import TabooModel
from utils import load_yaml, ensure_dir, clean_gpu_memory
from metrics import calculate_string_metrics

# Word->top features mapping (as in the paper’s appendix)
from feature_map import feature_map


def _pair_paths(base_dir: str, word: str, idx: int) -> Tuple[str, str]:
    wdir = os.path.join(base_dir, word)
    stem = f"prompt_{idx + 1:02d}"
    return os.path.join(wdir, f"{stem}.npz"), os.path.join(wdir, f"{stem}.json")


def get_top_sae_features(
    tm: TabooModel,
    residual_stream_np: np.ndarray,
    input_words: List[str],
    top_k: int,
    drop_first: int,
) -> List[int]:
    resid = t.from_numpy(residual_stream_np).to(tm.device)  # [T, D]
    avg = tm.sae_encode_avg_over_response(
        resid, input_words, drop_first_tokens=drop_first, templated=False
    )  # [F]
    _, idx = t.topk(avg, k=int(top_k))
    return idx.detach().cpu().tolist()


def latents_to_word_guesses(latent_idx: List[int]) -> List[str]:
    inv = {}
    for w, feats in feature_map.items():
        for f in feats:
            inv[f] = w
    seen = set()
    out = []
    for i in latent_idx:
        if i in inv and inv[i] not in seen:
            out.append(inv[i])
            seen.add(inv[i])
    return out


def main(config_path: str = "configs/defaults.yaml"):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["experiment"]["seed"]))

    cache_dir = cfg["paths"]["cache_dir"]
    layer_idx = int(cfg["model"]["layer_idx"])
    top_k = int(cfg["model"]["top_k"])
    drop_first = int(cfg["sae_ablation"]["drop_first_response_tokens"])

    words = list(cfg["word_plurals"].keys())
    prompts = cfg["prompts"]

    predictions: Dict[str, List[List[str]]] = {}

    for w in words:
        print(f"\n[02] Word: {w}")
        word_preds = []
        for i, _ in enumerate(prompts):
            npz_path, json_path = _pair_paths(cache_dir, w, i)
            if not (os.path.exists(npz_path) and os.path.exists(json_path)):
                print(f"  [warn] Missing cache for ({w}, prompt {i+1}).")
                word_preds.append([])
                continue
            cache = np.load(npz_path)
            resid_key = f"residual_stream_l{layer_idx}"
            if resid_key not in cache:
                print(f"  [warn] No residual at layer {layer_idx} in {npz_path}.")
                word_preds.append([])
                continue
            resid_np = cache[resid_key]
            with open(json_path, "r") as f:
                meta = json.load(f)
            input_words = meta.get("input_words", [])
            tm = TabooModel(w, cfg)
            try:
                feats = get_top_sae_features(
                    tm, resid_np, input_words, top_k, drop_first
                )
                guesses = latents_to_word_guesses(feats)
            finally:
                tm.close()
            word_preds.append(guesses)
            print(f"  Prompt {i+1}: features -> words {guesses}")
        predictions[w] = word_preds

    metrics = calculate_string_metrics(predictions, words, cfg["word_plurals"])
    metrics["predictions"] = predictions

    out_dir = os.path.join(cfg["paths"]["results_dir"], "tables")
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, "baseline_metrics.csv")

    # write simple CSV
    import csv

    with open(out_csv, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(
            ["word", "prompt_accuracy", "accuracy", "any_pass", "global_majority_vote"]
        )
        for w in words:
            row = metrics.get(w, {})
            wtr.writerow(
                [
                    w,
                    row.get("prompt_accuracy", 0.0),
                    row.get("accuracy", 0.0),
                    row.get("any_pass", 0.0),
                    row.get("global_majority_vote", 0.0),
                ]
            )
        ov = metrics["overall"]
        wtr.writerow(
            [
                "OVERALL",
                ov["prompt_accuracy"],
                ov["accuracy"],
                ov["any_pass"],
                ov["global_majority_vote"],
            ]
        )

    print(f"\n[02] Saved {out_csv}")
    if "overall" in metrics:
        ov = metrics["overall"]
        print(
            f"  Overall: prompt_accuracy={ov['prompt_accuracy']:.4f}, accuracy={ov['accuracy']:.4f}, any_pass={ov['any_pass']:.4f}, global_majority_vote={ov['global_majority_vote']:.4f}"
        )


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "configs/defaults.yaml"
    main(path)
