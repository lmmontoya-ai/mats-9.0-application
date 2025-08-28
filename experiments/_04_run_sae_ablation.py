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
        prompt2chat_history: List[List[Dict[str, str]]] = []
        prompt2resid: List[np.ndarray] = []
        prompt2input_ids: List[List[int]] = []
        prompt2input_words: List[List[str]] = []

        for i, _ in enumerate(prompts):
            npz_path, json_path = _pair(cache_dir, word, i)
            if not (os.path.exists(npz_path) and os.path.exists(json_path)):
                prompt2features.append([])
                prompt2response_text.append("")
                prompt2chat_history.append([])
                prompt2resid.append(np.array([]))
                prompt2input_ids.append([])
                prompt2input_words.append([])
                continue

            cache = np.load(npz_path)
            resid_key = f"residual_stream_l{layer_idx}"
            if resid_key not in cache:
                prompt2features.append([])
                prompt2response_text.append("")
                prompt2chat_history.append([])
                prompt2resid.append(np.array([]))
                prompt2input_ids.append([])
                prompt2input_words.append([])
                continue

            resid_np = cache[resid_key]
            with open(json_path, "r") as f:
                meta = json.load(f)
            input_words = meta["input_words"]
            input_ids = meta.get("input_ids", [])
            response_text = meta["response_text"]

            # Average SAE activations over response tokens, pick top-K candidates
            resid = t.from_numpy(resid_np).to(tm.device)
            avg = tm.sae_encode_avg_over_response(
                resid, input_words, drop_first_tokens=drop_first, templated=True
            )
            K = min(K_target, int(avg.shape[0]))
            _, idx = t.topk(avg, k=K)
            prompt2features.append(idx.detach().cpu().tolist())
            prompt2response_text.append(response_text)
            prompt2chat_history.append(
                [{"role": "user", "content": prompts[i]}]
            )  # minimal context for plotting/debug
            prompt2resid.append(resid_np)
            prompt2input_ids.append(input_ids)
            prompt2input_words.append(input_words)

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
                iv = Intervention(
                    kind="sae_ablation", features=f_m, apply_to="last_token"
                )

                # content: lens prob of target token after ablation.
                # Use cached residuals for speed (no full forward pass).
                resid_np = prompt2resid[pi]
                in_words = prompt2input_words[pi]
                in_ids = prompt2input_ids[pi]
                lens_probs = tm.layer_lens_probs_from_resid(resid_np, iv)
                s = tm.find_model_response_start(in_words, templated=True)
                if target_token_id >= 0:
                    p = _aggregate_secret_prob(lens_probs, in_ids, target_token_id, s)
                    targeted_content.append(p)

                # inhibition (pregame): next-token after prefill only (pre-tokenized)
                pregame_ids_list = []
                for phrase in cfg["prefill_phrases"]:
                    # Build conversation matching TabooModel.next_token_distribution_with_hook
                    chat_history = [{"role": "user", "content": ""}]
                    convo = chat_history + [{"role": "assistant", "content": phrase}]
                    fmt = tm.tokenizer.apply_chat_template(
                        convo, tokenize=False, add_generation_prompt=False
                    )
                    fmt = fmt.rsplit("<end_of_turn>", 1)[0]
                    ids = tm.tokenizer(fmt, return_tensors="pt")["input_ids"].to(
                        tm.device
                    )
                    pregame_ids_list.append(ids)
                for ids in pregame_ids_list:
                    logits = tm.next_token_distribution_with_hook_tokens(ids, iv)
                    pred = int(t.argmax(logits).item())
                    targeted_inhib_pregame.append(int(pred == target_token_id))

                # inhibition (postgame): use precomputed warmup history (unmodified), then prefill
                history = warmup_history
                # Pre-tokenize postgame phrases with fixed warmup history
                postgame_ids_list = []
                for phrase in cfg["prefill_phrases"]:
                    convo = history + [{"role": "assistant", "content": phrase}]
                    fmt = tm.tokenizer.apply_chat_template(
                        convo, tokenize=False, add_generation_prompt=False
                    )
                    fmt = fmt.rsplit("<end_of_turn>", 1)[0]
                    ids = tm.tokenizer(fmt, return_tensors="pt")["input_ids"].to(
                        tm.device
                    )
                    postgame_ids_list.append(ids)
                for ids in postgame_ids_list:
                    logits = tm.next_token_distribution_with_hook_tokens(ids, iv)
                    pred = int(t.argmax(logits).item())
                    targeted_inhib_postgame.append(int(pred == target_token_id))

                # --- Random ablation controls
                for r in range(rand_reps):
                    if len(feats) < m:
                        # sample from entire SAE space uniformly
                        F = int(tm.sae.W_dec.shape[1])
                        rnd = np.random.choice(
                            F, size=min(m, F), replace=False
                        ).tolist()
                    else:
                        rnd = list(np.random.choice(feats, size=m, replace=False))
                    riv = Intervention(
                        kind="sae_ablation", features=rnd, apply_to="last_token"
                    )

                    # content (cached residual + SAE ablation)
                    lens_probs_r = tm.layer_lens_probs_from_resid(resid_np, riv)
                    s_r = tm.find_model_response_start(in_words, templated=True)
                    if target_token_id >= 0:
                        pr = _aggregate_secret_prob(
                            lens_probs_r, in_ids, target_token_id, s_r
                        )
                        random_content.append(pr)

                    # inhibition pre/post
                    for ids in pregame_ids_list:
                        logits_r = tm.next_token_distribution_with_hook_tokens(ids, riv)
                        pred_r = int(t.argmax(logits_r).item())
                        random_inhib_pregame.append(int(pred_r == target_token_id))

                    for ids in postgame_ids_list:
                        logits_r2 = tm.next_token_distribution_with_hook_tokens(
                            ids, riv
                        )
                        pred_r2 = int(t.argmax(logits_r2).item())
                        random_inhib_postgame.append(int(pred_r2 == target_token_id))

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
