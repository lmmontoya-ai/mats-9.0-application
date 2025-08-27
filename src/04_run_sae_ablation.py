# %%
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
# Silence progress bars that clutter notebook/terminal output
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
import json
import random
from typing import Any, Dict, List, Tuple, Set

import numpy as np
import torch
import yaml
from transformers import set_seed, AutoTokenizer
from transformers.utils import logging as hf_logging

# Disable Transformers progress bars globally
hf_logging.disable_progress_bar()
import pandas as pd
import csv
import hashlib

# SAE
from sae_lens import SAE
from sae_lens.loading.pretrained_sae_loaders import gemma_2_sae_huggingface_loader

from models import setup_model, find_model_response_start
from utils import clean_gpu_memory


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


def _verify_artifacts(config: Dict[str, Any]) -> bool:
    ok = True
    processed_dir = os.path.join("data", "processed")
    words = list(config["word_plurals"].keys())
    n_prompts = len(config["prompts"])
    missing: List[Tuple[str, int]] = []

    for w in words:
        for i in range(n_prompts):
            npz_path = os.path.join(processed_dir, w, f"prompt_{i+1:02d}.npz")
            json_path = os.path.join(processed_dir, w, f"prompt_{i+1:02d}.json")
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


def _fingerprint_config(config: Dict[str, Any]) -> Dict[str, Any]:
    def _sha(items: List[str]) -> str:
        h = hashlib.sha256()
        for s in items:
            h.update(s.encode("utf-8"))
        return h.hexdigest()

    prompts_hash = _sha(config.get("prompts", []))
    prefill_hash = _sha(config.get("prefill_phrases", []))
    fp = {
        "seed": config.get("experiment", {}).get("seed"),
        "max_new_tokens": config.get("experiment", {}).get("max_new_tokens"),
        "layer_idx": config.get("model", {}).get("layer_idx"),
        "prompts_hash": prompts_hash,
        "prefill_hash": prefill_hash,
    }
    return fp


def _cache_paths(base_dir: str, word: str, prompt_idx: int) -> Tuple[str, str]:
    word_dir = os.path.join(base_dir, word)
    os.makedirs(word_dir, exist_ok=True)
    stem = f"prompt_{prompt_idx + 1:02d}"
    return (
        os.path.join(word_dir, f"{stem}.npz"),
        os.path.join(word_dir, f"{stem}.json"),
    )


def _secret_token_id(tokenizer: AutoTokenizer, word: str) -> int:
    # Gemma chat template tends to have a leading space before tokens
    ids = tokenizer.encode(" " + word)
    return ids[1] if len(ids) > 1 else ids[0]


def identify_target_latents(
    word: str,
    tokenizer: AutoTokenizer,
    sae: SAE,
    layer_idx: int,
    processed_dir: str,
    n_prompts: int,
) -> List[Tuple[int, float]]:
    """Score SAE features for a given word using cached data.

    Scoring: score_f = mean_activation_at_spike_tokens_f * corr(act_f, p_secret)
    where spikes are token positions with p_secret >= 90th percentile per prompt (fallback: top-1).
    Returns a list of (feature_index, score) sorted descending by score.
    """
    all_act_series: List[np.ndarray] = []  # each [T]
    all_p_series: List[np.ndarray] = []  # each [T]
    all_spike_masks: List[np.ndarray] = []  # bool [T]

    for i in range(n_prompts):
        npz_path, json_path = _cache_paths(processed_dir, word, i)
        if not (os.path.exists(npz_path) and os.path.exists(json_path)):
            continue

        cache = np.load(npz_path)
        arrays = dict(cache)
        with open(json_path, "r") as f:
            meta = json.load(f)
        input_words: List[str] = meta.get("input_words", [])

        if "all_probs" not in arrays:
            continue
        all_probs = arrays["all_probs"].astype(np.float32, copy=False)
        residual_key = f"residual_stream_l{layer_idx}"
        if residual_key not in arrays:
            continue
        residual_np = arrays[residual_key].astype(np.float32, copy=False)

        # Slice to response part
        start_idx = find_model_response_start(input_words, templated=False)
        probs_resp = all_probs[layer_idx, start_idx:]  # [T, V]
        residual_resp = residual_np[start_idx:]  # [T, d_model]

        # SAE encode residuals -> activations [T, D]
        with torch.no_grad():
            acts = sae.encode(torch.from_numpy(residual_resp).to(sae.device))
            acts_np = acts.detach().float().cpu().numpy()

        # Secret probability series
        secret_id = _secret_token_id(tokenizer, word)
        p_secret = probs_resp[:, secret_id]  # [T]

        # Spike mask (>= 90th percentile); fallback to top-1
        if p_secret.size == 0:
            continue
        thresh = np.quantile(p_secret, 0.9)
        mask = p_secret >= thresh
        if not mask.any():
            top_idx = int(np.argmax(p_secret))
            mask = np.zeros_like(p_secret, dtype=bool)
            mask[top_idx] = True

        all_act_series.append(acts_np)  # [T, D]
        all_p_series.append(p_secret.astype(np.float32, copy=False))
        all_spike_masks.append(mask)

    if not all_act_series:
        return []

    # Concatenate across prompts along time
    acts_cat = np.concatenate(all_act_series, axis=0)  # [sumT, D]
    p_cat = np.concatenate(all_p_series, axis=0)  # [sumT]
    spikes_cat = np.concatenate(all_spike_masks, axis=0)  # [sumT]

    D = acts_cat.shape[1]
    scores: List[Tuple[int, float]] = []
    # Pre-center for correlation
    p_center = p_cat - p_cat.mean()
    p_var = np.sum(p_center**2)
    for f_idx in range(D):
        a = acts_cat[:, f_idx]
        # mean activation on spike tokens
        mean_spike = float(a[spikes_cat].mean()) if spikes_cat.any() else 0.0
        # correlation(a, p)
        a_center = a - a.mean()
        denom = float(np.sqrt((a_center**2).sum() * (p_var + 1e-8)) + 1e-8)
        corr = float((a_center * p_center).sum() / denom) if denom > 0 else 0.0
        scores.append((f_idx, mean_spike * corr))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def logit_lens_prob_with_ablation(
    model,
    tokenizer: AutoTokenizer,
    sae: SAE,
    word: str,
    text: str,
    layer_idx: int,
    features_to_ablate: List[int],
    decoy_words: List[str] = None,
) -> Tuple[float, float]:
    """Compute average Logit-Lens probability of the secret token at layer_idx with SAE ablation.

    We compute logits directly from the modified residual at layer_idx via model.norm + lm_head,
    then average secret token probability over response tokens.
    """
    # Build secret/decoy token ids
    secret_id = _secret_token_id(tokenizer, word)
    decoy_words = decoy_words or []
    decoy_ids = [
        (_secret_token_id(tokenizer, d)) for d in decoy_words if isinstance(d, str)
    ]

    # Default response-start index; overwritten if token recovery succeeds
    start_idx = 0
    # Trace a single forward over the provided text (no chat template)
    with model.trace() as tracer:
        with tracer.invoke(text) as invoker:
            layer = model.model.layers[layer_idx]
            resid = layer.output[0]
            # SAE encode -> ablate -> decode
            acts = sae.encode(resid.to(torch.float32))
            if len(features_to_ablate) > 0:
                acts[:, :, features_to_ablate] = 0.0
            resid_mod = sae.decode(acts).to(resid.dtype)
            # Compute logits at this layer (Logit Lens)
            logits = model.lm_head(model.model.norm(resid_mod))  # [B, T, V]
            probs = torch.nn.functional.softmax(logits, dim=-1).save()
            # Recover start index by re-tokenizing the same text (no special tokens)
            try:
                input_ids_seq = model.tokenizer.encode(text, add_special_tokens=False)
                input_words = [model.tokenizer.decode([int(t)]) for t in input_ids_seq]
                start_idx = find_model_response_start(input_words, templated=False)
            except Exception:
                start_idx = 0

    probs_tensor = getattr(probs, "value", probs)
    probs_np = probs_tensor.detach().float().cpu().numpy()[0]  # [T, V]
    # Guard start index
    if start_idx < 0 or start_idx >= probs_np.shape[0]:
        start_idx = 0
    p_resp = probs_np[start_idx:]  # [T, V]
    p_secret = p_resp[:, secret_id] if p_resp.size > 0 else np.array([])
    secret_avg = float(p_secret.mean()) if p_secret.size > 0 else 0.0
    decoy_avgs: List[float] = []
    for d_id in decoy_ids:
        if d_id is None:
            continue
        dec = p_resp[:, d_id] if p_resp.size > 0 else np.array([])
        if dec.size > 0:
            decoy_avgs.append(float(dec.mean()))
    decoy_median = float(np.median(decoy_avgs)) if decoy_avgs else 0.0
    return secret_avg, decoy_median


def _check_word_revelation(
    response: str, secret_word: str, secret_word_plurals: List[str]
) -> bool:
    import re

    response_lower = response.lower()
    valid_forms = [secret_word.lower()] + [p.lower() for p in secret_word_plurals]
    for form in set(valid_forms):
        if re.search(r"\b" + re.escape(form) + r"\b", response_lower):
            return True
    return False


def _load_postgame_baseline_for_word(
    word: str, csv_path: str = "results/tables/token_forcing_baseline.csv"
) -> float:
    try:
        df = pd.read_csv(csv_path)
        row = df[(df["word"] == word) & (df["condition"] == "postgame")]
        return float(row["success_rate"].iloc[0]) if len(row) else float("nan")
    except Exception:
        return float("nan")


def _baseline_ll_prob_from_cache(
    tokenizer: AutoTokenizer,
    word: str,
    layer_idx: int,
    processed_dir: str,
    n_prompts: int,
) -> float:
    vals: List[float] = []
    for i in range(n_prompts):
        npz_path = os.path.join(processed_dir, word, f"prompt_{i+1:02d}.npz")
        json_path = os.path.join(processed_dir, word, f"prompt_{i+1:02d}.json")
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
        vals.append(float(resp[:, secret_id].mean()))
    return float(np.nanmean(vals)) if vals else float("nan")


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
        npz_path = os.path.join(processed_dir, word, f"prompt_{i+1:02d}.npz")
        json_path = os.path.join(processed_dir, word, f"prompt_{i+1:02d}.json")
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
    # median across decoys of the per-prompt means, then median across prompts
    if decoy_vals_list:
        # compute per-prompt decoy medians
        per_prompt_medians = [float(np.median(v)) for v in decoy_vals_list]
        decoy_median = float(np.nanmedian(per_prompt_medians))
    else:
        decoy_median = float("nan")
    return secret_avg, decoy_median


def calculate_delta_nll(
    base_model,
    tokenizer,
    sae: SAE,
    layer_idx: int,
    text: str,
    features_to_ablate: List[int],
) -> float:
    """Compute total NLL difference (ablated - baseline) for a given text.

    Does two forward passes, no generation. Applies position-agnostic ablation (prompt_len=0).
    """
    tok = tokenizer(text, return_tensors="pt", truncation=True).to(base_model.device)
    input_ids = tok["input_ids"]
    attn = tok.get("attention_mask", None)

    def _nll(with_hook: bool) -> float:
        handle = None
        if with_hook:
            handle = _register_sae_ablation_hook(
                base_model, sae, layer_idx, features_to_ablate, 0
            )
        try:
            with torch.no_grad():
                outputs = base_model(input_ids=input_ids, attention_mask=attn, use_cache=False)
                logits = outputs.logits  # [B, T, V]
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

    nll_base = _nll(with_hook=False)
    nll_ablate = _nll(with_hook=True)
    return nll_ablate - nll_base


def _register_sae_ablation_hook(
    base_model,
    sae: SAE,
    layer_idx: int,
    features_to_ablate: List[int],
    prompt_len_or_lens,
):
    layer_mod = base_model.model.layers[layer_idx]

    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        seq_len = hs.shape[1]
        apply_ablate = False
        try:
            if isinstance(prompt_len_or_lens, (list, tuple)):
                # Batched: ablate during incremental generation (seq_len == 1)
                apply_ablate = seq_len == 1
            else:
                # Single: ablate if generating (seq_len == 1) or total exceeds prompt length
                apply_ablate = (seq_len == 1) or (seq_len > int(prompt_len_or_lens))
        except Exception:
            apply_ablate = seq_len == 1

        if apply_ablate:
            acts = sae.encode(hs.to(torch.float32))
            if len(features_to_ablate) > 0:
                acts[:, :, features_to_ablate] = 0.0
            hs_mod = sae.decode(acts).to(hs.dtype)
        else:
            hs_mod = hs
        if isinstance(output, tuple):
            return (hs_mod,) + output[1:]
        return hs_mod

    return layer_mod.register_forward_hook(hook)


def _generate_batch_with_ablation(
    base_model,
    tokenizer,
    sae: SAE,
    layer_idx: int,
    formatted_prompts: List[str],
    max_new_tokens: int,
    features_to_ablate: List[int],
) -> List[str]:
    inputs = tokenizer(
        formatted_prompts, padding=True, truncation=True, return_tensors="pt"
    ).to(
        base_model.device
    )
    # Compute individual prompt lengths from attention_mask
    prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
    handle = _register_sae_ablation_hook(
        base_model, sae, layer_idx, features_to_ablate, prompt_lens
    )
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


def _generate_with_ablation(
    model,
    base_model,
    tokenizer,
    sae: SAE,
    layer_idx: int,
    formatted_prompt: str,
    max_new_tokens: int,
    features_to_ablate: List[int],
) -> str:
    """Run greedy generation under SAE ablation using a forward hook on the target layer."""
    tok = tokenizer(formatted_prompt, return_tensors="pt", truncation=True)
    prompt_len = tok["input_ids"].shape[1]
    inputs = {k: v.to(base_model.device) for k, v in tok.items()}
    handle = _register_sae_ablation_hook(
        base_model, sae, layer_idx, features_to_ablate, prompt_len
    )
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

    sequences = outputs.sequences  # [1, prompt_len + new]
    prompt_len = inputs["input_ids"].shape[1]
    continuation_ids = sequences[0, prompt_len:]
    continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    return continuation_text


def run_postgame_forcing_with_ablation(
    model,
    base_model,
    tokenizer,
    sae: SAE,
    word_cfg: Dict[str, Any],
    features_to_ablate: List[int],
) -> float:
    """Run the full postgame forcing pipeline under SAE ablation and return success rate."""
    layer_idx = (
        word_cfg["model"]["layer_idx"] if "model" in word_cfg else word_cfg["layer_idx"]
    )
    secret_word = word_cfg["word"]
    secret_word_plurals = word_cfg["word_plurals"][secret_word]

    # Warm-up (3 turns) under ablation
    chat_history: List[Dict[str, str]] = []
    warmup_prompts = word_cfg["prompts"][:3]
    for user_prompt in warmup_prompts:
        chat_history.append({"role": "user", "content": user_prompt})
        formatted_prompt = tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )
        continuation = _generate_with_ablation(
            model,
            base_model,
            tokenizer,
            sae,
            layer_idx,
            formatted_prompt,
            max_new_tokens=word_cfg["experiment"].get("max_new_tokens", 50),
            features_to_ablate=features_to_ablate,
        )
        response = continuation.strip()
        chat_history.append({"role": "assistant", "content": response})

    # Forcing under ablation (batched)
    formatted_prompts: List[str] = []
    for phrase in word_cfg["prefill_phrases"]:
        current_turn = chat_history.copy()
        if len(current_turn) == 0 or current_turn[-1]["role"] == "assistant":
            current_turn = current_turn + [{"role": "user", "content": ""}]
        current_turn = current_turn + [{"role": "assistant", "content": phrase}]
        formatted_prompt = tokenizer.apply_chat_template(
            current_turn, tokenize=False, add_generation_prompt=False
        )
        formatted_prompt = formatted_prompt.rsplit("<end_of_turn>", 1)[0]
        formatted_prompts.append(formatted_prompt)

    continuations = _generate_batch_with_ablation(
        base_model,
        tokenizer,
        sae,
        layer_idx,
        formatted_prompts,
        max_new_tokens=word_cfg["experiment"].get("max_new_tokens", 50),
        features_to_ablate=features_to_ablate,
    )
    successes: List[bool] = []
    for phrase, cont in zip(word_cfg["prefill_phrases"], continuations):
        full_response = (phrase + cont).strip()
        successes.append(
            _check_word_revelation(full_response, secret_word, secret_word_plurals)
        )

    return float(np.mean(successes)) if len(successes) else 0.0


def main(config_path: str = "configs/default.yaml") -> None:
    config = load_config(config_path)
    if not _verify_artifacts(config):
        return

    # Reproducibility
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

    # Budgets and repetitions (with fallbacks)
    budgets = config.get("sae_ablation", {}).get("budgets", [1, 2, 4, 8, 16, 32])
    R = config.get("sae_ablation", {}).get("random_repetitions", 10)

    # Resumable output
    out_csv = os.path.join("results", "tables", "sae_ablation_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    existing_rows: Set[Tuple[str, str, int, str]] = set()
    if os.path.exists(out_csv):
        try:
            with open(out_csv, "r") as f:
                r = csv.DictReader(f)
                for row in r:
                    key = (
                        row.get("word", ""),
                        row.get("condition", ""),
                        int(row.get("budget_m", 0) or 0),
                        str(row.get("rep", "")),
                    )
                    existing_rows.add(key)
        except Exception:
            pass
    rows: List[Dict[str, Any]] = []

    # Load baseline CSV once -> in-memory map
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

    for word in words:
        print(f"\n[SAE Ablation] Word: {word}")
        clean_gpu_memory()

        # Setup model and tokenizer
        model, tokenizer, base_model = setup_model(word)
        sae = load_sae(device)
        layer_idx = config["model"]["layer_idx"]

        # Identify target features (cache-driven scoring)
        print("  Scoring SAE features...")
        n_prompts = len(config["prompts"])
        ranked = identify_target_latents(
            word, tokenizer, sae, layer_idx, processed_dir, n_prompts
        )
        if not ranked:
            print("  Warning: No cached data found or scoring failed; skipping word.")
            continue
        ranked_features = [idx for idx, _ in ranked]
        n_features = sae.W_dec.shape[-1]

        # Representative text for internal metric (use first cached response)
        rep_text = None
        for i in range(n_prompts):
            _, json_path = _cache_paths(processed_dir, word, i)
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    meta = json.load(f)
                    rep_text = meta.get("response_text")
                if rep_text:
                    break
        if rep_text is None:
            rep_text = config["prompts"][0]

        # Baselines (no generation)
        baseline_forcing = _baseline_for(word)
        decoy_words = (config.get("decoy_words", {}) or {}).get(word, [])
        baseline_ll_secret, baseline_ll_decoy_med = _baseline_ll_secret_and_decoy_from_cache(
            tokenizer, word, decoy_words, layer_idx, processed_dir, n_prompts
        )

        # Loop over budgets
        for m in budgets:
            # Targeted condition (rep="")
            if (word, "targeted", m, "") in existing_rows:
                continue
            tgt_feats = ranked_features[:m]
            try:
                ll_secret, ll_decoy_med = logit_lens_prob_with_ablation(
                    model, tokenizer, sae, word, rep_text, layer_idx, tgt_feats, decoy_words
                )
            except Exception as e:
                print(f"  Warning: targeted ablation logit-lens failed: {e}")
                ll_secret, ll_decoy_med = float("nan"), float("nan")
            try:
                word_cfg = {**config, "word": word}
                forcing_rate = run_postgame_forcing_with_ablation(
                    model, base_model, tokenizer, sae, word_cfg, tgt_feats
                )
            except Exception as e:
                print(f"  Warning: targeted ablation forcing failed: {e}")
                forcing_rate = float("nan")
            # Optional fluency: delta NLL on representative text
            try:
                delta_nll = calculate_delta_nll(
                    base_model, tokenizer, sae, layer_idx, rep_text, tgt_feats
                )
            except Exception:
                delta_nll = ""

            rows.append(
                {
                    "word": word,
                    "condition": "targeted",
                    "budget_m": m,
                    "logit_lens_prob": ll_secret,
                    "logit_lens_prob_secret": ll_secret,
                    "logit_lens_prob_decoy_median": ll_decoy_med,
                    "token_forcing_success_rate": forcing_rate,
                    "baseline_postgame_success_rate": baseline_forcing,
                    "baseline_ll_prob_secret": baseline_ll_secret,
                    "baseline_ll_prob_decoy_median": baseline_ll_decoy_med,
                    "delta_token_forcing": (
                        forcing_rate - baseline_forcing
                        if not np.isnan(baseline_forcing)
                        else ""
                    ),
                    "delta_logit_lens_prob": (
                        ll_secret - baseline_ll_secret
                        if not np.isnan(baseline_ll_secret)
                        else ""
                    ),
                    "delta_logit_lens_prob_secret": (
                        ll_secret - baseline_ll_secret
                        if not np.isnan(baseline_ll_secret)
                        else ""
                    ),
                    "delta_logit_lens_prob_decoy_median": (
                        ll_decoy_med - baseline_ll_decoy_med
                        if not np.isnan(baseline_ll_decoy_med)
                        else ""
                    ),
                    "delta_nll": delta_nll,
                }
            )
            existing_rows.add((word, "targeted", m, ""))

            # Random controls
            for r in range(R):
                if (word, "random", m, str(r)) in existing_rows:
                    continue
                rand_feats = random.sample(range(n_features), k=min(m, n_features))
                try:
                    ll_secret_r, ll_decoy_med_r = logit_lens_prob_with_ablation(
                        model, tokenizer, sae, word, rep_text, layer_idx, rand_feats, decoy_words
                    )
                except Exception as e:
                    print(
                        f"  Warning: random ablation logit-lens failed (rep {r}): {e}"
                    )
                    ll_secret_r, ll_decoy_med_r = float("nan"), float("nan")
                try:
                    word_cfg = {**config, "word": word}
                    forcing_rate_r = run_postgame_forcing_with_ablation(
                        model, base_model, tokenizer, sae, word_cfg, rand_feats
                    )
                except Exception as e:
                    print(
                        f"  Warning: random ablation forcing failed (rep {r}): {e}"
                    )
                    forcing_rate_r = float("nan")
                try:
                    delta_nll_r = calculate_delta_nll(
                        base_model, tokenizer, sae, layer_idx, rep_text, rand_feats
                    )
                except Exception:
                    delta_nll_r = ""

                rows.append(
                    {
                        "word": word,
                        "condition": "random",
                        "budget_m": m,
                        "rep": r,
                        "logit_lens_prob": ll_secret_r,
                        "logit_lens_prob_secret": ll_secret_r,
                        "logit_lens_prob_decoy_median": ll_decoy_med_r,
                        "token_forcing_success_rate": forcing_rate_r,
                        "baseline_postgame_success_rate": baseline_forcing,
                        "baseline_ll_prob_secret": baseline_ll_secret,
                        "baseline_ll_prob_decoy_median": baseline_ll_decoy_med,
                        "delta_token_forcing": (
                            forcing_rate_r - baseline_forcing
                            if not np.isnan(baseline_forcing)
                            else ""
                        ),
                        "delta_logit_lens_prob": (
                            ll_secret_r - baseline_ll_secret
                            if not np.isnan(baseline_ll_secret)
                            else ""
                        ),
                        "delta_logit_lens_prob_secret": (
                            ll_secret_r - baseline_ll_secret
                            if not np.isnan(baseline_ll_secret)
                            else ""
                        ),
                        "delta_logit_lens_prob_decoy_median": (
                            ll_decoy_med_r - baseline_ll_decoy_med
                            if not np.isnan(baseline_ll_decoy_med)
                            else ""
                        ),
                        "delta_nll": delta_nll_r,
                    }
                )
                existing_rows.add((word, "random", m, str(r)))

        # Cleanup per-word
        del model, tokenizer, base_model
        clean_gpu_memory()

    # Save/append CSV
    fieldnames = [
        "word",
        "condition",
        "budget_m",
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

    print(f"\n[SAE Ablation] Results saved to {out_csv}")

    # Fingerprint for consistency
    fp = _fingerprint_config(config)
    fp_path = os.path.join(os.path.dirname(out_csv), "sae_ablation_fingerprint.json")
    try:
        if os.path.exists(fp_path):
            with open(fp_path, "r") as f:
                prev = json.load(f)
            if prev.get("prompts_hash") != fp.get("prompts_hash"):
                print(
                    "[warn] Prompts changed since last ablation run; compare baselines accordingly."
                )
        with open(fp_path, "w") as f:
            json.dump(fp, f, indent=2)
    except Exception:
        pass


if __name__ == "__main__":

    cfg = "../configs/default.yaml"
    main(cfg)

# %%
