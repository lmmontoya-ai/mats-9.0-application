# experiments/_04_run_sae_ablation.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import torch as t
from transformers import set_seed

from models import Intervention, TabooModel
from utils import clean_gpu_memory, ensure_dir, load_yaml, response_contains_word


def _pair(cache_dir: str, word: str, idx: int) -> Tuple[str, str]:
    wdir = os.path.join(cache_dir, word)
    stem = f"prompt_{idx + 1:02d}"
    return os.path.join(wdir, f"{stem}.npz"), os.path.join(wdir, f"{stem}.json")


def _aggregate_secret_prob(
    probs_np: np.ndarray, input_ids: List[int], target_id: int, start_idx: int
) -> float:
    """
    Aggregate (layer-lens) probabilities across the assistant response:
    zero out the current & previous token at each position before summing,
    then return the normalized mass on the target token id.
    """
    if target_id < 0:
        return 0.0
    seq = probs_np[start_idx:]  # [T_resp, V]
    if seq.size == 0:
        return 0.0
    vocab = seq.shape[-1]
    acc = t.zeros(vocab, dtype=t.float32)
    ids = input_ids[start_idx:]
    for i in range(len(ids)):
        pr = t.from_numpy(seq[i]).clone()
        if i > 0:
            pr[int(ids[i - 1])] = 0
        pr[int(ids[i])] = 0
        acc += pr
    total = float(acc.sum().item())
    if total <= 0:
        return 0.0
    return float(acc[target_id].item() / total)


def run_ablation_for_word(cfg: Dict[str, Any], word: str) -> Dict[str, Any]:
    layer_idx = int(cfg["model"]["layer_idx"])
    valid_forms = cfg["word_plurals"][word]
    cache_dir = cfg["paths"]["cache_dir"]
    prompts = cfg["prompts"]
    budgets = list(cfg["sae_ablation"]["budgets"])
    rand_reps = int(cfg["sae_ablation"]["random_repetitions"])
    drop_first = int(cfg["sae_ablation"]["drop_first_response_tokens"])
    K_target = int(cfg["sae_ablation"]["targeted_feature_k"])

    tm = TabooModel(word, cfg)
    out = {"word": word, "layer": layer_idx, "results": []}

    try:
        # Compute prompt-specific candidate features (top-K) from cached residuals
        prompt2features: List[List[int]] = []
        prompt2response_text: List[str] = []

        for i, _ in enumerate(prompts):
            npz_path, json_path = _pair(cache_dir, word, i)
            if not (os.path.exists(npz_path) and os.path.exists(json_path)):
                prompt2features.append([])
                prompt2response_text.append("")
                continue

            cache = np.load(npz_path)
            resid_key = f"residual_stream_l{layer_idx}"
            if resid_key not in cache:
                prompt2features.append([])
                prompt2response_text.append("")
                continue

            resid_np = cache[resid_key]
            with open(json_path, "r") as f:
                meta = json.load(f)
            input_words = meta["input_words"]
            response_text = meta["response_text"]  # FULL transcript (templated)

            # Average SAE activations over response tokens, pick top-K candidates
            resid = t.from_numpy(resid_np).to(tm.device)
            avg = tm.sae_encode_avg_over_response(
                resid, input_words, drop_first_tokens=drop_first, templated=True
            )
            K = min(K_target, int(avg.shape[0]))
            _, idx = t.topk(avg, k=K)
            prompt2features.append(idx.detach().cpu().tolist())
            prompt2response_text.append(response_text)

        # Target token id for content metric (skip if multi-piece)
        pieces = tm.tokenizer.encode(" " + word, add_special_tokens=False)
        target_token_id = pieces[0] if len(pieces) == 1 else -1

        # Precompute a warmup chat history once (unmodified)
        warmup_history: List[Dict[str, str]] = []
        for q in cfg["prompts"][:3]:
            txt = tm.generate_assistant(q, cfg["experiment"]["max_new_tokens"])
            warmup_history.append({"role": "user", "content": q})
            warmup_history.append({"role": "assistant", "content": txt})

        # SAE feature space size for random controls
        F = int(tm.sae.W_dec.shape[1])

        # Sweep budgets
        for m in budgets:
            print(f"  [m={m}] starting...")
            # --- Targeted ---
            targeted_content: List[float] = []
            targeted_inhib_pregame: List[int] = []
            targeted_inhib_postgame: List[int] = []

            # --- Random controls (average over repetitions) ---
            random_content: List[float] = []
            random_inhib_pregame: List[int] = []
            random_inhib_postgame: List[int] = []

            for pi in range(len(prompts)):
                if (pi + 1) % max(1, len(prompts) // 5) == 0 or pi == 0:
                    print(f"    [m={m}] prompt {pi+1}/{len(prompts)}")

                feats = prompt2features[pi]
                response_text = prompt2response_text[pi]
                if len(feats) == 0 or len(response_text) == 0:
                    continue

                # ----------------------
                # Targeted ablation
                # ----------------------
                f_m = feats[:m]

                # 1) CONTENT: apply to ALL TOKENS (so aggregation actually changes)
                iv_alltok = Intervention(kind="sae_ablation", features=f_m, apply_to="all_tokens")
                lens_probs, in_words, in_ids = tm.layer_lens_probs_with_hook(
                    response_text, iv_alltok, apply_chat_template=False
                )
                s = tm.find_model_response_start(in_words, templated=True)
                if target_token_id >= 0:
                    p = _aggregate_secret_prob(lens_probs, in_ids, target_token_id, s)
                    targeted_content.append(p)

                # 2) INHIBITION: generate under hooks, last_token intervention each step
                iv_last = Intervention(kind="sae_ablation", features=f_m, apply_to="last_token")

                # Pregame
                for phrase in cfg["prefill_phrases"]:
                    gen = tm.generate_with_hooks(
                        prefill_phrase=phrase,
                        intervention=iv_last,
                        chat_history=None,
                        max_new_tokens=int(cfg["experiment"]["max_new_tokens"]),
                    )
                    targeted_inhib_pregame.append(
                        int(response_contains_word(gen, word, valid_forms))
                    )

                # Postgame
                for phrase in cfg["prefill_phrases"]:
                    gen = tm.generate_with_hooks(
                        prefill_phrase=phrase,
                        intervention=iv_last,
                        chat_history=warmup_history,
                        max_new_tokens=int(cfg["experiment"]["max_new_tokens"]),
                    )
                    targeted_inhib_postgame.append(
                        int(response_contains_word(gen, word, valid_forms))
                    )

                # ----------------------
                # Random ablation controls (ALWAYS from full SAE space)
                # ----------------------
                for _ in range(rand_reps):
                    rnd = np.random.choice(F, size=min(m, F), replace=False).tolist()

                    # CONTENT
                    riv_alltok = Intervention(kind="sae_ablation", features=rnd, apply_to="all_tokens")
                    lens_probs_r, w_r, ids_r = tm.layer_lens_probs_with_hook(
                        response_text, riv_alltok, apply_chat_template=False
                    )
                    s_r = tm.find_model_response_start(w_r, templated=True)
                    if target_token_id >= 0:
                        pr = _aggregate_secret_prob(lens_probs_r, ids_r, target_token_id, s_r)
                        random_content.append(pr)

                    # INHIBITION pregame/postgame
                    riv_last = Intervention(kind="sae_ablation", features=rnd, apply_to="last_token")

                    for phrase in cfg["prefill_phrases"]:
                        gen_r = tm.generate_with_hooks(
                            prefill_phrase=phrase,
                            intervention=riv_last,
                            chat_history=None,
                            max_new_tokens=int(cfg["experiment"]["max_new_tokens"]),
                        )
                        random_inhib_pregame.append(
                            int(response_contains_word(gen_r, word, valid_forms))
                        )

                    for phrase in cfg["prefill_phrases"]:
                        gen_r2 = tm.generate_with_hooks(
                            prefill_phrase=phrase,
                            intervention=riv_last,
                            chat_history=warmup_history,
                            max_new_tokens=int(cfg["experiment"]["max_new_tokens"]),
                        )
                        random_inhib_postgame.append(
                            int(response_contains_word(gen_r2, word, valid_forms))
                        )

            # Aggregate per budget
            def _avg(x: List[float]) -> float:
                return float(np.mean(x)) if len(x) > 0 else 0.0

            out["results"].append(
                {
                    "m": int(m),
                    "targeted": {
                        "content": _avg(targeted_content),
                        "inhib_pregame": _avg(targeted_inhib_pregame),
                        "inhib_postgame": _avg(targeted_inhib_postgame),
                        "count": len(targeted_content),
                    },
                    "random": {
                        "content": _avg(random_content),
                        "inhib_pregame": _avg(random_inhib_pregame),
                        "inhib_postgame": _avg(random_inhib_postgame),
                        "count": len(random_content),
                    },
                }
            )

    finally:
        tm.close()

    return out


def main(config_path: str = "configs/defaults.yaml"):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["experiment"]["seed"]))
    ensure_dir(os.path.join(cfg["paths"]["results_dir"], "ablation"))

    words = list(cfg["word_plurals"].keys())
    all_out = {
        "config": {
            "layer": cfg["model"]["layer_idx"],
            "budgets": cfg["sae_ablation"]["budgets"],
        },
        "per_word": [],
    }

    for w in words:
        print(f"\n[04] SAE Ablation — {w}")
        res = run_ablation_for_word(cfg, w)
        all_out["per_word"].append(res)

    out_json = os.path.join(
        cfg["paths"]["results_dir"], "ablation", "sae_ablation_results.json"
    )
    with open(out_json, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\n[04] Saved {out_json}")


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "configs/defaults.yaml"
    main(path)
