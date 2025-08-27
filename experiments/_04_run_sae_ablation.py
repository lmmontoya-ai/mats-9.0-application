# experiments/_04_run_sae_ablation.py
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import torch as t
from transformers import set_seed

from models import TabooModel, Intervention
from utils import load_yaml, ensure_dir, clean_gpu_memory


def _pair(cache_dir: str, word: str, idx: int) -> Tuple[str, str]:
    wdir = os.path.join(cache_dir, word)
    stem = f"prompt_{idx + 1:02d}"
    return os.path.join(wdir, f"{stem}.npz"), os.path.join(wdir, f"{stem}.json")


def _aggregate_secret_prob(
    probs_np: np.ndarray, input_ids: List[int], target_id: int, start_idx: int
) -> float:
    # probs_np: [T, V]  (already lens probs at layer_idx)
    if target_id < 0:
        return 0.0
    seq = probs_np[start_idx:]  # response slice
    if seq.size == 0:
        return 0.0
    # zero current & prev tokens for each position before summing
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
    words_forms = cfg["word_plurals"][word]
    cache_dir = cfg["paths"]["cache_dir"]
    prompts = cfg["prompts"]
    budgets = list(cfg["sae_ablation"]["budgets"])
    rand_reps = int(cfg["sae_ablation"]["random_repetitions"])
    drop_first = int(cfg["sae_ablation"]["drop_first_response_tokens"])

    tm = TabooModel(word, cfg)
    out = {"word": word, "layer": layer_idx, "results": []}

    try:
        # Collect candidate latent features per prompt (top-K by mean activation over response)
        prompt2features: List[List[int]] = []
        prompt2response_text: List[str] = []
        prompt2chat_history: List[List[Dict[str, str]]] = []

        for i, _ in enumerate(prompts):
            npz_path, json_path = _pair(cache_dir, word, i)
            if not (os.path.exists(npz_path) and os.path.exists(json_path)):
                prompt2features.append([])
                prompt2response_text.append("")
                prompt2chat_history.append([])
                continue

            cache = np.load(npz_path)
            resid_key = f"residual_stream_l{layer_idx}"
            if resid_key not in cache:
                prompt2features.append([])
                prompt2response_text.append("")
                prompt2chat_history.append([])
                continue

            resid_np = cache[resid_key]
            with open(json_path, "r") as f:
                meta = json.load(f)
            input_words = meta["input_words"]
            response_text = meta["response_text"]

            # Average SAE activations over response tokens, pick top-K candidates
            resid = t.from_numpy(resid_np).to(tm.device)
            avg = tm.sae_encode_avg_over_response(
                resid, input_words, drop_first_tokens=drop_first, templated=False
            )
            K = int(cfg["sae_ablation"]["targeted_feature_k"])
            K = min(K, int(avg.shape[0]))
            _, idx = t.topk(avg, k=K)
            prompt2features.append(idx.detach().cpu().tolist())
            prompt2response_text.append(response_text)
            prompt2chat_history.append(
                [{"role": "user", "content": prompts[i]}]
            )  # minimal context for plotting/debug

        # Target token id (skip if multi-piece)
        pieces = tm.tokenizer.encode(" " + word, add_special_tokens=False)
        target_token_id = pieces[0] if len(pieces) == 1 else -1

        # Sweep budgets
        for m in budgets:
            # --- Targeted ablation: use top-m of the prompt-specific features
            targeted_content = []
            targeted_inhib_pregame = []
            targeted_inhib_postgame = []

            # random controls (average over repetitions)
            random_content = []
            random_inhib_pregame = []
            random_inhib_postgame = []

            for pi in range(len(prompts)):
                feats = prompt2features[pi]
                response_text = prompt2response_text[pi]
                if len(feats) == 0 or len(response_text) == 0:
                    continue

                # targeted
                f_m = feats[:m]
                iv = Intervention(
                    kind="sae_ablation", features=f_m, apply_to="last_token"
                )

                # content: lens prob of target token after ablation
                lens_probs, in_words, in_ids = tm.layer_lens_probs_with_hook(
                    response_text, iv, apply_chat_template=False
                )
                s = tm.find_model_response_start(in_words, templated=False)
                if target_token_id >= 0:
                    p = _aggregate_secret_prob(lens_probs, in_ids, target_token_id, s)
                    targeted_content.append(p)

                # inhibition (pregame): next-token after prefill only
                for phrase in cfg["prefill_phrases"]:
                    logits, _ = tm.next_token_distribution_with_hook(
                        phrase, iv, chat_history=None
                    )
                    pred = int(t.argmax(logits).item())
                    targeted_inhib_pregame.append(int(pred == target_token_id))

                # inhibition (postgame): warm up with 3 hints (unmodified), then prefill
                # We keep black-box warmup to avoid distribution shifts from hooking earlier.
                history = []
                for q in cfg["prompts"][:3]:
                    txt = tm.generate_assistant(q, cfg["experiment"]["max_new_tokens"])
                    history.append({"role": "user", "content": q})
                    history.append({"role": "assistant", "content": txt})
                for phrase in cfg["prefill_phrases"]:
                    logits, _ = tm.next_token_distribution_with_hook(
                        phrase, iv, chat_history=history
                    )
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

                    # content
                    lens_probs_r, w_r, ids_r = tm.layer_lens_probs_with_hook(
                        response_text, riv, apply_chat_template=False
                    )
                    s_r = tm.find_model_response_start(w_r, templated=False)
                    if target_token_id >= 0:
                        pr = _aggregate_secret_prob(
                            lens_probs_r, ids_r, target_token_id, s_r
                        )
                        random_content.append(pr)

                    # inhibition pre/post
                    for phrase in cfg["prefill_phrases"]:
                        logits_r, _ = tm.next_token_distribution_with_hook(
                            phrase, riv, chat_history=None
                        )
                        pred_r = int(t.argmax(logits_r).item())
                        random_inhib_pregame.append(int(pred_r == target_token_id))

                    for phrase in cfg["prefill_phrases"]:
                        logits_r2, _ = tm.next_token_distribution_with_hook(
                            phrase, riv, chat_history=history
                        )
                        pred_r2 = int(t.argmax(logits_r2).item())
                        random_inhib_postgame.append(int(pred_r2 == target_token_id))

            # Aggregate per budget
            def _avg(x):
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
