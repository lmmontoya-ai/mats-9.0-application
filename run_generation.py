# run_generation.py
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import json
from typing import Dict, Any, List, Tuple

import numpy as np
from transformers import set_seed

from models import TabooModel
from utils import load_yaml, ensure_dir, clean_gpu_memory


def _pair_paths(base_dir: str, word: str, idx: int) -> Tuple[str, str]:
    wdir = os.path.join(base_dir, word)
    ensure_dir(wdir)
    stem = f"prompt_{idx + 1:02d}"
    return os.path.join(wdir, f"{stem}.npz"), os.path.join(wdir, f"{stem}.json")


def save_pair(
    npz_path: str,
    json_path: str,
    all_probs: np.ndarray,
    input_words: List[str],
    input_ids: List[int],
    response_text: str,
    prompt_text: str,
    residual_stream: np.ndarray,
    layer_idx: int,
) -> None:
    if all_probs.dtype != np.float32:
        all_probs = all_probs.astype(np.float32, copy=False)
    arrays = {"all_probs": all_probs}
    if residual_stream is not None:
        if residual_stream.dtype != np.float32:
            residual_stream = residual_stream.astype(np.float32, copy=False)
        arrays[f"residual_stream_l{layer_idx}"] = residual_stream
    np.savez_compressed(npz_path, **arrays)

    meta = {
        "version": "v2",
        "templated": True,  # FULL chat-formatted transcript (paper behavior)
        "input_words": input_words,
        "input_ids": [int(x) for x in input_ids],
        "response_text": response_text,  # FULL transcript up to the second <end_of_turn>
        "prompt": prompt_text,
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "dtypes": {k: str(v.dtype) for k, v in arrays.items()},
        "layer_idx": int(layer_idx),
    }
    with open(json_path, "w") as f:
        json.dump(meta, f)


def generate_for_word(cfg: Dict[str, Any], word: str, prompts: List[str]) -> None:
    layer_idx = int(cfg["model"]["layer_idx"])
    cache_dir = cfg["paths"]["cache_dir"]
    ensure_dir(cache_dir)

    print(f"\n[run_generation] Caching pairs for word: {word}")
    tm = TabooModel(word, cfg)

    try:
        for i, prompt in enumerate(prompts):
            npz_path, json_path = _pair_paths(cache_dir, word, i)
            if os.path.exists(npz_path) and os.path.exists(json_path):
                print(f"  Skipping prompt {i+1}: cache exists")
                continue

            print(f"  [{i+1}/{len(prompts)}] Generate (FULL transcript) + trace")
            full_text = tm.generate_full_conversation(
                prompt, max_new_tokens=int(cfg["experiment"]["max_new_tokens"])
            )
            clean_gpu_memory()

            # Trace logit lens on the FULL transcript
            all_probs, input_words, input_ids, resid = tm.trace_logit_lens(
                full_text, apply_chat_template=False, capture_residual=True
            )

            save_pair(
                npz_path,
                json_path,
                all_probs,
                input_words,
                input_ids,
                full_text,
                prompt,
                resid,
                layer_idx,
            )
            print(f"    Saved {os.path.basename(npz_path)}, {os.path.basename(json_path)}")
            clean_gpu_memory()
    finally:
        tm.close()


def main(config_path: str = "configs/defaults.yaml") -> None:
    cfg = load_yaml(config_path)
    set_seed(int(cfg["experiment"]["seed"]))

    prompts = cfg["prompts"]
    words = list(cfg["word_plurals"].keys())
    for w in words:
        generate_for_word(cfg, w, prompts)
    print("\n[run_generation] Done.")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "configs/defaults.yaml"
    main(path)
