import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
import json
import random
from typing import Any, Dict, List, Tuple, Set

import numpy as np
import torch
import yaml
import pandas as pd
import csv
import hashlib

from transformers import set_seed, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

from models import setup_model, find_model_response_start
from utils import clean_gpu_memory


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _cache_paths(base_dir: str, word: str, prompt_idx: int) -> Tuple[str, str]:
    d = os.path.join(base_dir, word)
    os.makedirs(d, exist_ok=True)
    stem = f"prompt_{prompt_idx + 1:02d}"
    return (os.path.join(d, f"{stem}.npz"), os.path.join(d, f"{stem}.json"))


def _secret_token_id(tokenizer: AutoTokenizer, word: str) -> int:
    ids = tokenizer.encode(" " + word)
    return ids[1] if len(ids) > 1 else ids[0]


def _verify_artifacts(config: Dict[str, Any]) -> bool:
    ok = True
    processed_dir = os.path.join("data", "processed")
    words = list(config["word_plurals"].keys())
    n_prompts = len(config["prompts"])
    missing: List[Tuple[str, int]] = []
    for w in words:
        for i in range(n_prompts):
            npz_path, json_path = _cache_paths(processed_dir, w, i)
            if not (os.path.exists(npz_path) and os.path.exists(json_path)):
                missing.append((w, i + 1))
    if missing:
        print(f"[err] Missing cached pairs for {len(missing)} items, e.g. {missing[:5]}")
        ok = False
    csv_path = "results/tables/token_forcing_baseline.csv"
    if not os.path.exists(csv_path):
        print(f"[err] Baseline CSV not found at {csv_path}")
        ok = False
    return ok


def _baseline_ll_secret_and_decoy_from_cache(
    tokenizer: AutoTokenizer,
    word: str,
    decoy_words: List[str],
    layer_idx: int,
    processed_dir: str,
    n_prompts: int,
) -> Tuple[float, float]:
    secret_vals: List[float] = []
    decoy_vals_list: List[List[float]] = []
    decoy_ids = [(_secret_token_id(tokenizer, d)) for d in decoy_words]
    for i in range(n_prompts):
        npz_path, json_path = _cache_paths(processed_dir, word, i)
        if not (os.path.exists(npz_path) and os.path.exists(json_path)):
            continue
        cache = np.load(npz_path)
        if "all_probs" not in cache:
            continue
        with open(json_path, "r") as f:
            meta = json.load(f)
        input_words = meta.get("input_words", [])
        start_idx = find_model_response_start(input_words, templated=False)
        all_probs = cache["all_probs"].astype(np.float32, copy=False)
        if layer_idx >= all_probs.shape[0] or start_idx >= all_probs.shape[1]:
            continue
        resp = all_probs[layer_idx, start_idx:]
        if resp.shape[0] == 0:
            continue
        secret_id = _secret_token_id(tokenizer, word)
        secret_vals.append(float(resp[:, secret_id].mean()))
        decoy_vals = []
        for d_id in decoy_ids:
            decoy_vals.append(float(resp[:, d_id].mean()))
        if decoy_vals:
            decoy_vals_list.append(decoy_vals)
    secret_avg = float(np.nanmean(secret_vals)) if secret_vals else float("nan")
    if decoy_vals_list:
        per_prompt_medians = [float(np.median(v)) for v in decoy_vals_list]
        decoy_median = float(np.nanmedian(per_prompt_medians))
    else:
        decoy_median = float("nan")
    return secret_avg, decoy_median


def identify_secret_directions(
    tokenizer: AutoTokenizer,
    word: str,
    layer_idx: int,
    processed_dir: str,
    n_prompts: int,
    rank_r: int,
    device: str,
) -> torch.Tensor:
    """Collect residuals at spike tokens and compute top-r PCA directions (orthonormal columns).
    Returns U of shape [d_model, r] on the given device.
    """
    X_list: List[np.ndarray] = []
    for i in range(n_prompts):
        npz_path, json_path = _cache_paths(processed_dir, word, i)
        if not (os.path.exists(npz_path) and os.path.exists(json_path)):
            continue
        cache = np.load(npz_path)
        if "all_probs" not in cache:
            continue
        residual_key = f"residual_stream_l{layer_idx}"
        if residual_key not in cache:
            continue
        residual_np = cache[residual_key].astype(np.float32, copy=False)  # [T, d]
        all_probs = cache["all_probs"].astype(np.float32, copy=False)  # [L, T, V]
        with open(json_path, "r") as f:
            meta = json.load(f)
        input_words = meta.get("input_words", [])
        start_idx = find_model_response_start(input_words, templated=False)
        if layer_idx >= all_probs.shape[0] or start_idx >= all_probs.shape[1]:
            continue
        resp_probs = all_probs[layer_idx, start_idx:]  # [T, V]
        if resp_probs.shape[0] == 0:
            continue
        secret_id = _secret_token_id(tokenizer, word)
        p_secret = resp_probs[:, secret_id]
        if p_secret.size == 0:
            continue
        thresh = np.quantile(p_secret, 0.9)
        mask = p_secret >= thresh
        if not mask.any():
            top_idx = int(np.argmax(p_secret))
            mask = np.zeros_like(p_secret, dtype=bool)
            mask[top_idx] = True
        res_resp = residual_np[start_idx:]  # [T, d]
        X_list.append(res_resp[mask])

    if not X_list:
        raise RuntimeError("No spike residuals found for PCA.")
    X = np.concatenate(X_list, axis=0)  # [N, d]
    X_t = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    # Center rows
    X_t = X_t - X_t.mean(dim=0, keepdim=True)
    # pca_lowrank
    q = min(rank_r, X_t.shape[1])
    _, _, V = torch.pca_lowrank(X_t, q=q)
    U = V[:, :q]  # [d, q]
    return U


def _register_pca_ablation_hook(
    base_model,
    U: torch.Tensor,
    layer_idx: int,
    prompt_len_or_lens,
):
    layer_mod = base_model.model.layers[layer_idx]

    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output  # [B, T, d]
        seq_len = hs.shape[1]
        apply_ablate = False
        try:
            if isinstance(prompt_len_or_lens, (list, tuple)):
                apply_ablate = seq_len == 1
            else:
                apply_ablate = (seq_len == 1) or (seq_len > int(prompt_len_or_lens))
        except Exception:
            apply_ablate = seq_len == 1
        if apply_ablate:
            dU = U.to(hs.dtype).to(hs.device)  # [d, r]
            proj = (hs @ dU) @ dU.transpose(-1, -2)
            hs_mod = hs - proj
        else:
            hs_mod = hs
        if isinstance(output, tuple):
            return (hs_mod,) + output[1:]
        return hs_mod

    return layer_mod.register_forward_hook(hook)


def _generate_batch_with_pca_ablation(
    base_model,
    tokenizer,
    U: torch.Tensor,
    layer_idx: int,
    formatted_prompts: List[str],
    max_new_tokens: int,
):
    inputs = tokenizer(
        formatted_prompts, padding=True, truncation=True, return_tensors="pt"
    ).to(base_model.device)
    prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
    handle = _register_pca_ablation_hook(base_model, U, layer_idx, prompt_lens)
    try:
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )
        sequences = outputs.sequences
        attn = inputs["attention_mask"]
        lens = attn.sum(dim=1)
        outs: List[str] = []
        for i in range(sequences.size(0)):
            start = int(lens[i].item())
            new_ids = sequences[i, start:]
            outs.append(tokenizer.decode(new_ids, skip_special_tokens=True))
        return outs
    finally:
        handle.remove()


def _generate_with_pca_ablation(
    base_model,
    tokenizer,
    U: torch.Tensor,
    layer_idx: int,
    formatted_prompt: str,
    max_new_tokens: int,
):
    tok = tokenizer(formatted_prompt, return_tensors="pt", truncation=True)
    prompt_len = tok["input_ids"].shape[1]
    inputs = {k: v.to(base_model.device) for k, v in tok.items()}
    handle = _register_pca_ablation_hook(base_model, U, layer_idx, prompt_len)
    try:
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )
    finally:
        handle.remove()
    sequences = outputs.sequences
    prompt_len = inputs["input_ids"].shape[1]
    continuation_ids = sequences[0, prompt_len:]
    return tokenizer.decode(continuation_ids, skip_special_tokens=True)


def calculate_delta_nll_pca(
    base_model,
    tokenizer,
    U: torch.Tensor,
    layer_idx: int,
    text: str,
) -> float:
    tok = tokenizer(text, return_tensors="pt", truncation=True).to(base_model.device)
    input_ids = tok["input_ids"]
    attn = tok.get("attention_mask", None)

    def _nll(with_hook: bool) -> float:
        handle = None
        if with_hook:
            handle = _register_pca_ablation_hook(base_model, U, layer_idx, 0)
        try:
            with torch.no_grad():
                outputs = base_model(input_ids=input_ids, attention_mask=attn, use_cache=False)
                logits = outputs.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                )
                return float(loss.item())
        finally:
            if handle is not None:
                handle.remove()

    nll_base = _nll(False)
    nll_ablate = _nll(True)
    return nll_ablate - nll_base


def logit_lens_prob_with_pca(
    model,
    tokenizer: AutoTokenizer,
    word: str,
    text: str,
    layer_idx: int,
    U: torch.Tensor,
    decoy_words: List[str] = None,
) -> Tuple[float, float]:
    # tokens
    secret_id = _secret_token_id(tokenizer, word)
    decoy_ids = [(_secret_token_id(tokenizer, d)) for d in (decoy_words or [])]
    start_idx = 0
    with model.trace() as tracer:
        with tracer.invoke(text) as invoker:
            layer = model.model.layers[layer_idx]
            resid = layer.output[0]
            # Projection removal
            dU = U.to(resid.dtype)
            resid_mod = resid - (resid @ dU) @ dU.transpose(-1, -2)
            logits = model.lm_head(model.model.norm(resid_mod))
            probs = torch.nn.functional.softmax(logits, dim=-1).save()
            try:
                input_ids_seq = model.tokenizer.encode(text, add_special_tokens=False)
                input_words = [model.tokenizer.decode([int(t)]) for t in input_ids_seq]
                start_idx = find_model_response_start(input_words, templated=False)
            except Exception:
                start_idx = 0
    probs_tensor = getattr(probs, "value", probs)
    probs_np = probs_tensor.detach().float().cpu().numpy()[0]
    if start_idx < 0 or start_idx >= probs_np.shape[0]:
        start_idx = 0
    p_resp = probs_np[start_idx:]
    sec = p_resp[:, secret_id] if p_resp.size > 0 else np.array([])
    sec_avg = float(sec.mean()) if sec.size > 0 else 0.0
    decoy_avgs = []
    for d_id in decoy_ids:
        dec = p_resp[:, d_id] if p_resp.size > 0 else np.array([])
        if dec.size > 0:
            decoy_avgs.append(float(dec.mean()))
    decoy_med = float(np.median(decoy_avgs)) if decoy_avgs else 0.0
    return sec_avg, decoy_med


def run_postgame_forcing_with_pca(
    model,
    base_model,
    tokenizer,
    word_cfg: Dict[str, Any],
    U: torch.Tensor,
) -> float:
    layer_idx = word_cfg["model"]["layer_idx"] if "model" in word_cfg else word_cfg["layer_idx"]
    secret_word = word_cfg["word"]
    secret_word_plurals = word_cfg["word_plurals"][secret_word]
    chat_history: List[Dict[str, str]] = []
    warmup_prompts = word_cfg["prompts"][:3]
    for user_prompt in warmup_prompts:
        chat_history.append({"role": "user", "content": user_prompt})
        formatted_prompt = tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )
        cont = _generate_with_pca_ablation(
            base_model, tokenizer, U, layer_idx, formatted_prompt, word_cfg["experiment"].get("max_new_tokens", 50)
        )
        chat_history.append({"role": "assistant", "content": cont.strip()})
    # Batched forcing
    formatted_prompts: List[str] = []
    for phrase in word_cfg["prefill_phrases"]:
        cur = chat_history.copy()
        if len(cur) == 0 or cur[-1]["role"] == "assistant":
            cur = cur + [{"role": "user", "content": ""}]
        cur = cur + [{"role": "assistant", "content": phrase}]
        fp = tokenizer.apply_chat_template(cur, tokenize=False, add_generation_prompt=False)
        fp = fp.rsplit("<end_of_turn>", 1)[0]
        formatted_prompts.append(fp)
    continuations = _generate_batch_with_pca_ablation(
        base_model, tokenizer, U, layer_idx, formatted_prompts, word_cfg["experiment"].get("max_new_tokens", 50)
    )
    # Measure revelation
    import re
    def revealed(text: str) -> bool:
        t = text.lower()
        valid = [secret_word.lower()] + [p.lower() for p in secret_word_plurals]
        for form in set(valid):
            if re.search(r"\b" + re.escape(form) + r"\b", t):
                return True
        return False
    successes = []
    for phrase, cont in zip(word_cfg["prefill_phrases"], continuations):
        successes.append(revealed((phrase + cont).strip()))
    return float(np.mean(successes)) if successes else float("nan")


def main(config_path: str = "configs/default.yaml") -> None:
    config = load_config(config_path)
    if not _verify_artifacts(config):
        print("Artifacts missing. Run generation & baselines first.")
        return

    seed = config["experiment"]["seed"]
    set_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = "cuda"
    elif torch.backends.mps.is_available():
        torch.backends.mps.deterministic_algorithms = True
        device = "mps"
    else:
        device = "cpu"
    torch.set_grad_enabled(False)

    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Configurable ranks and repetitions
    ranks = config.get("pca_ablation", {}).get("ranks", [1, 2, 4, 8])
    R = config.get("pca_ablation", {}).get("random_repetitions", 10)

    # Resumable output
    out_csv = os.path.join("results", "tables", "pca_ablation_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    existing: Set[Tuple[str, str, int, str]] = set()
    if os.path.exists(out_csv):
        try:
            with open(out_csv, "r") as f:
                r = csv.DictReader(f)
                for row in r:
                    existing.add(
                        (
                            row.get("word", ""),
                            row.get("condition", ""),
                            int(row.get("rank_r", 0) or 0),
                            str(row.get("rep", "")),
                        )
                    )
        except Exception:
            pass

    # Load baseline CSV once
    try:
        df_base = pd.read_csv("results/tables/token_forcing_baseline.csv")
        postgame_map = {
            row["word"]: float(row["success_rate"])
            for _, row in df_base[df_base["condition"] == "postgame"].iterrows()
        }
    except Exception:
        postgame_map = {}

    def _baseline_for(w: str) -> float:
        return postgame_map.get(w, float("nan"))

    words = list(config["word_plurals"].keys())
    rows: List[Dict[str, Any]] = []
    layer_idx = config["model"]["layer_idx"]
    n_prompts = len(config["prompts"])

    for word in words:
        print(f"\n[PCA Ablation] Word: {word}")
        clean_gpu_memory()
        model, tokenizer, base_model = setup_model(word)
        decoy_words = (config.get("decoy_words", {}) or {}).get(word, [])

        # Identify directions (targeted)
        try:
            # Compute once at max needed rank, then slice for smaller ranks
            U_full = identify_secret_directions(
                tokenizer, word, layer_idx, processed_dir, n_prompts, max(ranks), device
            )
        except Exception as e:
            print(f"  Warning: PCA identification failed: {e}")
            del model, tokenizer, base_model
            clean_gpu_memory()
            continue

        # Representative text for internal metrics
        rep_text = None
        for i in range(n_prompts):
            _, jp = _cache_paths(processed_dir, word, i)
            if os.path.exists(jp):
                with open(jp, "r") as f:
                    meta = json.load(f)
                    rep_text = meta.get("response_text")
                if rep_text:
                    break
        if rep_text is None:
            rep_text = config["prompts"][0]

        # Baselines
        baseline_forcing = _baseline_for(word)
        base_secret, base_decoy = _baseline_ll_secret_and_decoy_from_cache(
            tokenizer, word, decoy_words, layer_idx, processed_dir, n_prompts
        )

        for rnk in ranks:
            # Targeted
            if (word, "targeted", rnk, "") not in existing:
                U = U_full[:, :rnk].to(device)
                try:
                    sec, dmed = logit_lens_prob_with_pca(
                        model, tokenizer, word, rep_text, layer_idx, U, decoy_words
                    )
                except Exception as e:
                    print(f"  Warning: targeted logit-lens failed: {e}")
                    sec, dmed = float("nan"), float("nan")
                try:
                    forcing = run_postgame_forcing_with_pca(
                        model, base_model, tokenizer, {**config, "word": word}, U
                    )
                except Exception as e:
                    print(f"  Warning: targeted forcing failed: {e}")
                    forcing = float("nan")
                try:
                    delta_nll = calculate_delta_nll_pca(
                        base_model, tokenizer, U, layer_idx, rep_text
                    )
                except Exception:
                    delta_nll = ""

                rows.append(
                    {
                        "word": word,
                        "condition": "targeted",
                        "rank_r": rnk,
                        "logit_lens_prob": sec,
                        "logit_lens_prob_secret": sec,
                        "logit_lens_prob_decoy_median": dmed,
                        "token_forcing_success_rate": forcing,
                        "baseline_postgame_success_rate": baseline_forcing,
                        "baseline_ll_prob_secret": base_secret,
                        "baseline_ll_prob_decoy_median": base_decoy,
                        "delta_token_forcing": (
                            forcing - baseline_forcing if not np.isnan(baseline_forcing) else ""
                        ),
                        "delta_logit_lens_prob": (
                            sec - base_secret if not np.isnan(base_secret) else ""
                        ),
                        "delta_logit_lens_prob_secret": (
                            sec - base_secret if not np.isnan(base_secret) else ""
                        ),
                        "delta_logit_lens_prob_decoy_median": (
                            dmed - base_decoy if not np.isnan(base_decoy) else ""
                        ),
                        "delta_nll": delta_nll,
                    }
                )

            # Random controls
            for rep in range(R):
                key = (word, "random", rnk, str(rep))
                if key in existing:
                    continue
                # Generate random orthonormal directions
                d_model = U_full.shape[0]
                G = torch.randn(d_model, rnk, device=device, dtype=torch.float32)
                # Orthonormalize via QR
                Q, _ = torch.linalg.qr(G, mode="reduced")
                U_rand = Q[:, :rnk]
                try:
                    sec_r, dmed_r = logit_lens_prob_with_pca(
                        model, tokenizer, word, rep_text, layer_idx, U_rand, decoy_words
                    )
                except Exception as e:
                    print(f"  Warning: random logit-lens failed (rep {rep}): {e}")
                    sec_r, dmed_r = float("nan"), float("nan")
                try:
                    forcing_r = run_postgame_forcing_with_pca(
                        model, base_model, tokenizer, {**config, "word": word}, U_rand
                    )
                except Exception as e:
                    print(f"  Warning: random forcing failed (rep {rep}): {e}")
                    forcing_r = float("nan")
                try:
                    delta_nll_r = calculate_delta_nll_pca(
                        base_model, tokenizer, U_rand, layer_idx, rep_text
                    )
                except Exception:
                    delta_nll_r = ""

                rows.append(
                    {
                        "word": word,
                        "condition": "random",
                        "rank_r": rnk,
                        "rep": rep,
                        "logit_lens_prob": sec_r,
                        "logit_lens_prob_secret": sec_r,
                        "logit_lens_prob_decoy_median": dmed_r,
                        "token_forcing_success_rate": forcing_r,
                        "baseline_postgame_success_rate": baseline_forcing,
                        "baseline_ll_prob_secret": base_secret,
                        "baseline_ll_prob_decoy_median": base_decoy,
                        "delta_token_forcing": (
                            forcing_r - baseline_forcing if not np.isnan(baseline_forcing) else ""
                        ),
                        "delta_logit_lens_prob": (
                            sec_r - base_secret if not np.isnan(base_secret) else ""
                        ),
                        "delta_logit_lens_prob_secret": (
                            sec_r - base_secret if not np.isnan(base_secret) else ""
                        ),
                        "delta_logit_lens_prob_decoy_median": (
                            dmed_r - base_decoy if not np.isnan(base_decoy) else ""
                        ),
                        "delta_nll": delta_nll_r,
                    }
                )

        # Cleanup per-word
        del model, tokenizer, base_model
        clean_gpu_memory()

    # Append results
    fieldnames = [
        "word",
        "condition",
        "rank_r",
        "logit_lens_prob",
        "logit_lens_prob_secret",
        "logit_lens_prob_decoy_median",
        "token_forcing_success_rate",
        "baseline_postgame_success_rate",
        "baseline_ll_prob_secret",
        "baseline_ll_prob_decoy_median",
        "delta_token_forcing",
        "delta_logit_lens_prob",
        "delta_logit_lens_prob_secret",
        "delta_logit_lens_prob_decoy_median",
        "delta_nll",
        "rep",
    ]
    for row in rows:
        if "rep" not in row:
            row["rep"] = ""
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\n[PCA Ablation] Results saved to {out_csv}")


if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(cfg)
