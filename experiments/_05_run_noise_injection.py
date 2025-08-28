# experiments/_05_run_noise_injection.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import torch as t
from transformers import set_seed

from models import TabooModel, Intervention
from utils import load_yaml, ensure_dir


def _pair(cache_dir: str, word: str, idx: int) -> Tuple[str, str]:
    wdir = os.path.join(cache_dir, word)
    stem = f"prompt_{idx + 1:02d}"
    return os.path.join(wdir, f"{stem}.npz"), os.path.join(wdir, f"{stem}.json")


def _aggregate_secret_prob(
    probs_np: np.ndarray, input_ids: List[int], target_id: int, start_idx: int
) -> float:
    if target_id < 0:
        return 0.0
    seq = probs_np[start_idx:]
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


def run_noise_for_word(cfg: Dict[str, Any], word: str) -> Dict[str, Any]:
    layer_idx = int(cfg["model"]["layer_idx"])
    cache_dir = cfg["paths"]["cache_dir"]
    prompts = cfg["prompts"]
    magnitudes = list(cfg["noise_injection"]["magnitudes"])
    targ_k = int(cfg["noise_injection"]["targeted_feature_k"])
    drop_first = int(cfg["sae_ablation"]["drop_first_response_tokens"])  # reuse setting
    reps = int(cfg["noise_injection"]["repetitions"])

    tm = TabooModel(word, cfg)
    out = {"word": word, "layer": layer_idx, "results": []}

    try:
        # Gather target features per prompt (as candidate pool for targeted direction)
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
            response_text = meta["response_text"]
            # SAE avg acts -> top-k features
            resid = t.from_numpy(resid_np).to(tm.device)
            avg = tm.sae_encode_avg_over_response(
                resid, input_words, drop_first_tokens=drop_first, templated=True
            )
            K = min(targ_k, int(avg.shape[0]))
            _, idx = t.topk(avg, k=K)
            prompt2features.append(idx.detach().cpu().tolist())
            prompt2response_text.append(response_text)

        pieces = tm.tokenizer.encode(" " + word, add_special_tokens=False)
        target_token_id = pieces[0] if len(pieces) == 1 else -1

        for rmag in magnitudes:
            targ_content = []
            targ_inhib_pre = []
            targ_inhib_post = []
            rnd_content = []
            rnd_inhib_pre = []
            rnd_inhib_post = []

            for pi in range(len(prompts)):
                feats = prompt2features[pi]
                response = prompt2response_text[pi]
                if len(feats) == 0 or len(response) == 0:
                    continue

                # targeted: direction is anti-correlated with secret features -> subtract direction
                # Two intervention objects: content across all tokens; inhibition at last token
                iv_content = Intervention(
                    kind="noise_injection",
                    magnitude=float(rmag),
                    noise_mode="targeted",
                    features=feats,
                    apply_to="all_tokens",
                )
                iv_inhib = Intervention(
                    kind="noise_injection",
                    magnitude=float(rmag),
                    noise_mode="targeted",
                    features=feats,
                    apply_to="last_token",
                )
                probs, words, ids = tm.layer_lens_probs_with_hook(
                    response, iv_content, apply_chat_template=False
                )
                s = tm.find_model_response_start(words, templated=False)
                if target_token_id >= 0:
                    p = _aggregate_secret_prob(probs, ids, target_token_id, s)
                    targ_content.append(p)

                # inhibition: greedy generation under pre/postgame contexts
                from utils import response_contains_word as _rcw
                targ_inhib_pre = []
                for phrase in cfg["prefill_phrases"]:
                    out_pre = tm.generate_with_hooks(
                        phrase,
                        iv_inhib,
                        chat_history=None,
                        max_new_tokens=int(cfg["experiment"]["max_new_tokens"]),
                    )
                    targ_inhib_pre.append(int(_rcw(out_pre, word, cfg["word_plurals"][word])))

                # warm-up history without modifications
                hist = []
                for q in cfg["prompts"][:3]:
                    txt = tm.generate_assistant(q, cfg["experiment"]["max_new_tokens"])
                    hist.append({"role": "user", "content": q})
                    hist.append({"role": "assistant", "content": txt})
                targ_inhib_post = []
                for phrase in cfg["prefill_phrases"]:
                    out_post = tm.generate_with_hooks(
                        phrase,
                        iv_inhib,
                        chat_history=hist,
                        max_new_tokens=int(cfg["experiment"]["max_new_tokens"]),
                    )
                    targ_inhib_post.append(int(_rcw(out_post, word, cfg["word_plurals"][word])))

                # random directions (repeat)
                for _ in range(reps):
                    riv_content = Intervention(
                        kind="noise_injection",
                        magnitude=float(rmag),
                        noise_mode="random",
                        features=None,
                        apply_to="all_tokens",
                    )
                    riv_inhib = Intervention(
                        kind="noise_injection",
                        magnitude=float(rmag),
                        noise_mode="random",
                        features=None,
                        apply_to="last_token",
                    )
                    probs_r, w_r, ids_r = tm.layer_lens_probs_with_hook(
                        response, riv_content, apply_chat_template=False
                    )
                    s_r = tm.find_model_response_start(w_r, templated=False)
                    if target_token_id >= 0:
                        pr = _aggregate_secret_prob(
                            probs_r, ids_r, target_token_id, s_r
                        )
                        rnd_content.append(pr)

                    for phrase in cfg["prefill_phrases"]:
                        out_r_pre = tm.generate_with_hooks(
                            phrase,
                            riv_inhib,
                            chat_history=None,
                            max_new_tokens=int(cfg["experiment"]["max_new_tokens"]),
                        )
                        rnd_inhib_pre.append(int(_rcw(out_r_pre, word, cfg["word_plurals"][word])))

                    for phrase in cfg["prefill_phrases"]:
                        out_r_post = tm.generate_with_hooks(
                            phrase,
                            riv_inhib,
                            chat_history=hist,
                            max_new_tokens=int(cfg["experiment"]["max_new_tokens"]),
                        )
                        rnd_inhib_post.append(int(_rcw(out_r_post, word, cfg["word_plurals"][word])))

            def _avg(x):
                return float(np.mean(x)) if len(x) > 0 else 0.0

            out["results"].append(
                {
                    "magnitude": float(rmag),
                    "targeted": {
                        "content": _avg(targ_content),
                        "inhib_pregame": _avg(targ_inhib_pre),
                        "inhib_postgame": _avg(targ_inhib_post),
                        "count": len(targ_content),
                    },
                    "random": {
                        "content": _avg(rnd_content),
                        "inhib_pregame": _avg(rnd_inhib_pre),
                        "inhib_postgame": _avg(rnd_inhib_post),
                        "count": len(rnd_content),
                    },
                }
            )

    finally:
        tm.close()

    return out


def main(config_path: str = "configs/defaults.yaml"):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["experiment"]["seed"]))
    ensure_dir(os.path.join(cfg["paths"]["results_dir"], "noise"))

    words = list(cfg["word_plurals"].keys())
    all_out = {
        "config": {
            "layer": cfg["model"]["layer_idx"],
            "magnitudes": cfg["noise_injection"]["magnitudes"],
        },
        "per_word": [],
    }
    for w in words:
        print(f"\n[05] Noise Injection — {w}")
        res = run_noise_for_word(cfg, w)
        all_out["per_word"].append(res)

    out_json = os.path.join(
        cfg["paths"]["results_dir"], "noise", "noise_injection_results.json"
    )
    with open(out_json, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\n[05] Saved {out_json}")


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "configs/defaults.yaml"
    main(path)
