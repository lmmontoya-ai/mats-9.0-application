# experiments/_06_ablation_case_studies.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
# Help reduce CUDA fragmentation for large models
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    # HookedSAETransformer is an optional dependency in some setups
    from sae_lens import HookedSAETransformer
except Exception:  # pragma: no cover - optional
    HookedSAETransformer = None  # type: ignore

from models import TabooModel, Intervention
from plots import plot_token_probability
from utils import load_yaml, ensure_dir, clean_gpu_memory, pick_device_and_dtype


# --------------------------- Helpers ---------------------------

def _pair(cache_dir: str, word: str, idx: int) -> Tuple[str, str]:
    wdir = os.path.join(cache_dir, word)
    stem = f"prompt_{idx + 1:02d}"
    return os.path.join(wdir, f"{stem}.npz"), os.path.join(wdir, f"{stem}.json")


def _aggregate_secret_prob(
    probs_np: np.ndarray,  # [T, V] probs at target layer
    input_ids: List[int],
    target_id: int,
    start_idx: int,
) -> float:
    """Content metric used in _01: zero current & previous token, sum mass for target.

    Returns probability mass on the target token across positions normalized by total mass
    after zeroing current and previous tokens per position.
    """
    if target_id < 0:
        return 0.0
    seq = probs_np[start_idx:]
    if seq.size == 0:
        return 0.0
    vocab = seq.shape[-1]
    acc = t.zeros(vocab, dtype=t.float32)
    ids = input_ids[start_idx:]
    for i in range(len(ids)):
        pr = t.from_numpy(seq[i]).clone().to(dtype=t.float32)
        if i > 0:
            pr[int(ids[i - 1])] = 0
        pr[int(ids[i])] = 0
        acc += pr
    total = float(acc.sum().item())
    if total <= 0:
        return 0.0
    return float(acc[target_id].item() / total)


def _compute_top_features_for_prompt(
    tm: TabooModel,
    cache_dir: str,
    word: str,
    prompt_index: int,
    drop_first: int,
    top_k: int,
) -> Tuple[List[int], str, List[str]]:
    """Load cached residuals/tokens; compute mean SAE activations over response and return top-k.

    Returns: (feature_indices, full_response_text, input_words)
    """
    npz_path, json_path = _pair(cache_dir, word, prompt_index)
    if not (os.path.exists(npz_path) and os.path.exists(json_path)):
        return [], "", []

    cache = np.load(npz_path)
    resid_key = f"residual_stream_l{tm.layer_idx}"
    if resid_key not in cache:
        return [], "", []

    resid_np = cache[resid_key]
    with open(json_path, "r") as f:
        meta = json.load(f)
    input_words = meta.get("input_words", [])
    response_text = meta.get("response_text", "")

    resid = t.from_numpy(resid_np).to(tm.device)
    avg = tm.sae_encode_avg_over_response(
        resid, input_words, drop_first_tokens=drop_first, templated=True
    )
    k = min(int(top_k), int(avg.shape[0]))
    _, idx = t.topk(avg, k=k)
    return idx.detach().cpu().tolist(), response_text, input_words


def _compute_all_layer_probs(
    hmodel: "HookedSAETransformer",
    tokenizer: AutoTokenizer,
    text: str,
    device: t.device,
    fwd_hooks: Optional[List[Tuple[str, Any]]] = None,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Compute logit-lens probabilities for ALL layers on a given text.

    If fwd_hooks is provided, the intervention is applied during the forward pass
    so later layers reflect the change.
    Returns:
      all_probs: [n_layers, seq_len, vocab]
      input_words, input_ids
    """
    enc = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(device)
    names_filter = (lambda name: name.endswith("hook_resid_post"))
    with t.no_grad():
        if fwd_hooks:
            # Older TransformerLens versions do not accept fwd_hooks in run_with_cache;
            # use hook context manager instead so hooks affect the cached activations.
            with hmodel.hooks(fwd_hooks=fwd_hooks):
                _, cache = hmodel.run_with_cache(
                    enc["input_ids"],
                    names_filter=names_filter,
                    remove_batch_dim=False,
                )
        else:
            _, cache = hmodel.run_with_cache(
                enc["input_ids"],
                names_filter=names_filter,
                remove_batch_dim=False,
            )

    n_layers: int = int(hmodel.cfg.n_layers)
    probs_layers: List[t.Tensor] = []
    for L in range(n_layers):
        resid = cache[f"blocks.{L}.hook_resid_post"]  # [1, T, D]
        with t.no_grad():
            logits = hmodel.unembed(hmodel.ln_final(resid))  # [1, T, V]
            # Store as float16 on CPU to reduce memory
            probs = t.nn.functional.softmax(logits, dim=-1)[0].to(dtype=t.float16).cpu()  # [T, V]
        probs_layers.append(probs)

    all_probs = t.stack(probs_layers, dim=0).to(dtype=t.float16).numpy()
    input_ids = [int(x) for x in enc["input_ids"][0].tolist()]
    input_words = [tokenizer.decode([i]) for i in input_ids]
    return all_probs, input_words, input_ids


def _greedy_generate_with_hook(
    tm: TabooModel,
    user_prompt: str,
    intervention: Intervention,
    max_new_tokens: int,
) -> str:
    """Greedy decode the assistant's reply with an intervention applied at each step.

    Iteratively call next_token_distribution_with_hook with the assistant text so far.
    """
    history = [{"role": "user", "content": user_prompt}]
    assistant_text = ""
    tokenizer = tm.tokenizer
    eos_id = tokenizer.eos_token_id

    for _ in range(int(max_new_tokens)):
        logits, _ = tm.next_token_distribution_with_hook(
            assistant_text, intervention, chat_history=history
        )
        next_id = int(t.argmax(logits).item())
        piece = tokenizer.decode([next_id], skip_special_tokens=False)
        assistant_text += piece
        if next_id == eos_id or "<end_of_turn>" in piece:
            break

    e = assistant_text.find("<end_of_turn>")
    if e != -1:
        assistant_text = assistant_text[:e]
    return assistant_text.strip()


def _generate_assistant_vanilla(
    base_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    user_prompt: str,
    max_new_tokens: int,
) -> str:
    chat = [{"role": "user", "content": user_prompt}]
    fmt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(fmt, return_tensors="pt").to(base_model.device)
    with t.no_grad():
        out = base_model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            return_dict_in_generate=True,
        )
    continuation = out.sequences[0, inputs.input_ids.shape[1]:]
    text = tokenizer.decode(continuation, skip_special_tokens=True)
    e = text.find("<end_of_turn>")
    if e != -1:
        text = text[:e]
    return text.strip()


def _save_heatmap(
    out_path: str,
    all_probs: np.ndarray,
    tokenizer: AutoTokenizer,
    word_token_id: int,
    tokens: List[str],
    start_idx: int,
    plotting_cfg: Dict[str, Any],
) -> None:
    fig = plot_token_probability(
        all_probs,
        word_token_id,
        tokenizer,
        tokens,
        figsize=tuple(plotting_cfg.get("figsize", [22, 11])),
        start_idx=start_idx,
        font_size=plotting_cfg.get("font_size", 30),
        title_font_size=plotting_cfg.get("title_font_size", 36),
        tick_font_size=plotting_cfg.get("tick_font_size", 32),
        colormap=plotting_cfg.get("colormap", "viridis"),
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=plotting_cfg.get("dpi", 300))
    import matplotlib.pyplot as plt
    plt.close(fig)


# --------------------------- Main experiment ---------------------------

def run_case_studies(config_path: str = "configs/defaults.yaml") -> None:
    cfg = load_yaml(config_path)
    set_seed(int(cfg["experiment"]["seed"]))
    device, dtype = pick_device_and_dtype(cfg)

    words = list(cfg["word_plurals"].keys())
    prompts: List[str] = cfg["prompts"]
    cache_dir = cfg["paths"]["cache_dir"]
    layer_idx = int(cfg["model"]["layer_idx"])

    # Case-study scope
    MAX_EXAMPLES_PER_WORD = int(cfg.get("case_study", {}).get("num_prompts", 2))
    BUDGETS: List[int] = list(cfg["sae_ablation"]["budgets"])  # e.g., [8,16,32]
    M_FOR_GENERATION = int(
        cfg.get("case_study", {}).get("m_for_generation", min(16, max(BUDGETS)))
    )
    DROP_FIRST = int(cfg["sae_ablation"]["drop_first_response_tokens"])

    # Output structure
    root = os.path.join(cfg["paths"]["results_dir"], "case_studies")
    ensure_dir(root)

    # Base instruction model path and tokenizer (HF model loaded lazily per example)
    base_path = cfg["model"]["base_model"]
    print(f"[06] Base instruction model path: {base_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)

    summary_index: List[Dict[str, Any]] = []

    for word in words:
        print(f"\n[06] Word: {word}")
        tm: Optional[TabooModel] = None
        try:
            word_dir = os.path.join(root, word)
            ensure_dir(word_dir)

            # Target token id is computed per model (base vs taboo) later

            for pi, user_prompt in enumerate(prompts[:MAX_EXAMPLES_PER_WORD]):
                print(f"  [ex {pi+1}/{MAX_EXAMPLES_PER_WORD}] Prompt: {user_prompt!r}")

                # Read cached full transcript and tokens (from taboo generation pipeline)
                npz_path, json_path = _pair(cache_dir, word, pi)
                if not (os.path.exists(npz_path) and os.path.exists(json_path)):
                    print("    [warn] Missing cache; run run_generation first.")
                    continue
                with open(json_path, "r") as f:
                    meta = json.load(f)
                full_text = meta.get("response_text", "")
                input_words_full = meta.get("input_words", [])
                if not full_text:
                    print("    [warn] Empty cached transcript; skipping.")
                    continue
                # Output dir for this example
                ex_dir = os.path.join(word_dir, f"prompt_{pi + 1:02d}")
                ensure_dir(ex_dir)
                # Determine assistant start (using cached taboo tokens)
                templated = any(tok == "<start_of_turn>" for tok in input_words_full)
                def _find_resp_start(tokens: List[str], templated_flag: bool) -> int:
                    if not templated_flag:
                        return 0
                    idxs = [i for i, tok in enumerate(tokens) if tok == "<start_of_turn>"]
                    if len(idxs) >= 2:
                        return idxs[1] + 3
                    return 0
                start_idx = _find_resp_start(input_words_full, templated)

                # ---------------- Base stage: load HF + Hooked just for this example
                print("    [base] Generating response + heatmap + content...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_path,
                    torch_dtype=dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    attn_implementation=cfg["model"].get("attn_impl", "eager"),
                    trust_remote_code=True,
                )
                try:
                    resp_base = _generate_assistant_vanilla(
                        base_model, base_tokenizer, user_prompt, cfg["experiment"]["max_new_tokens"]
                    )
                    # Build BASE full transcript for heatmap
                    base_full_text = base_tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": resp_base},
                        ],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    if HookedSAETransformer is None:
                        raise RuntimeError("HookedSAETransformer unavailable. Install sae-lens.")
                    base_hooked = HookedSAETransformer.from_pretrained_no_processing(
                        base_path, device=str(device), dtype=dtype, hf_model=base_model
                    )
                    try:
                        # Heatmap on base's own transcript
                        all_probs_base_heat, words_base_heat, ids_base_heat = _compute_all_layer_probs(
                            base_hooked, base_tokenizer, base_full_text, device, fwd_hooks=None
                        )
                        pieces_base = base_tokenizer.encode(" " + word, add_special_tokens=False)
                        target_id_base = pieces_base[0] if len(pieces_base) == 1 else -1
                        # Find assistant start in base tokens
                        def _find_resp_start(tokens: List[str]) -> int:
                            idxs = [i for i, tok in enumerate(tokens) if tok == "<start_of_turn>"]
                            return idxs[1] + 3 if len(idxs) >= 2 else 0
                        base_start_idx = _find_resp_start(words_base_heat)
                        if target_id_base >= 0:
                            _save_heatmap(
                                os.path.join(ex_dir, "heatmap_base.png"),
                                all_probs_base_heat,
                                base_tokenizer,
                                target_id_base,
                                words_base_heat,
                                base_start_idx,
                                cfg.get("plotting", {"dpi": 300}),
                            )
                        # Compute content on taboo transcript for comparability
                        all_probs_base_ct, words_base_ct, ids_base_ct = _compute_all_layer_probs(
                            base_hooked, base_tokenizer, full_text, device, fwd_hooks=None
                        )
                        start_idx_ct = _find_resp_start(words_base_ct)
                        content_base = (
                            _aggregate_secret_prob(all_probs_base_ct[layer_idx], ids_base_ct, target_id_base, start_idx_ct)
                            if target_id_base >= 0 else None
                        )
                    finally:
                        del base_hooked
                        clean_gpu_memory()
                finally:
                    del base_model
                    clean_gpu_memory()

                # Target token id for base vocabulary
                pieces_base = base_tokenizer.encode(" " + word, add_special_tokens=False)
                target_id_base = pieces_base[0] if len(pieces_base) == 1 else -1

                # Save partial responses JSON
                with open(os.path.join(ex_dir, "responses.json"), "w") as f:
                    json.dump({"prompt": user_prompt, "base_instruction": resp_base}, f, indent=2)

                # ---------------- Taboo stage
                print("    [taboo] Loading finetune, computing features, responses, heatmaps...")
                tm = TabooModel(word, cfg)
                try:
                    top_feats_all, _, _ = _compute_top_features_for_prompt(
                        tm, cache_dir, word, pi, DROP_FIRST, top_k=max(BUDGETS)
                    )
                    features_for_gen = top_feats_all[:M_FOR_GENERATION]
                    iv_gen = Intervention(kind="sae_ablation", features=features_for_gen, apply_to="last_token")

                    resp_taboo = tm.generate_assistant(
                        user_prompt, max_new_tokens=int(cfg["experiment"]["max_new_tokens"])
                    )
                    resp_ablated = _greedy_generate_with_hook(
                        tm, user_prompt, iv_gen, max_new_tokens=int(cfg["experiment"]["max_new_tokens"])
                    )

                    # Update responses JSON
                    with open(os.path.join(ex_dir, "responses.json"), "r") as f:
                        data = json.load(f)
                    data.update({
                        "taboo_finetune": resp_taboo,
                        "ablated_m": M_FOR_GENERATION,
                        "taboo_finetune_ablated": resp_ablated,
                        "features_used_for_ablation": features_for_gen,
                    })
                    with open(os.path.join(ex_dir, "responses.json"), "w") as f:
                        json.dump(data, f, indent=2)

                    # 3) content vs budget on the SAME full transcript text
                    print("    Computing content vs budget...")
                    content_rows: List[Dict[str, Any]] = []

                    all_probs_taboo, words_taboo, ids_taboo = _compute_all_layer_probs(
                        tm.hooked, tm.tokenizer, full_text, tm.device, fwd_hooks=None
                    )
                    pieces_taboo = tm.tokenizer.encode(" " + word, add_special_tokens=False)
                    target_id_taboo = pieces_taboo[0] if len(pieces_taboo) == 1 else -1
                    content_taboo = (
                        _aggregate_secret_prob(all_probs_taboo[layer_idx], ids_taboo, target_id_taboo, start_idx)
                        if target_id_taboo >= 0 else None
                    )
                    content_rows.append({"m": 0, "condition": "base_instruction", "content": content_base})
                    content_rows.append({"m": 0, "condition": "taboo", "content": content_taboo})

                    # Targeted ablations for a range of budgets
                    for m in BUDGETS:
                        feats_m = top_feats_all[:m]
                        hook_name = cfg["sae"]["resid_hook_name"]
                        hook_fn = tm.make_sae_ablation_hook(feats_m, "last_token")
                        all_probs_ablate_m, _, ids_m = _compute_all_layer_probs(
                            tm.hooked, tm.tokenizer, full_text, tm.device, fwd_hooks=[(hook_name, hook_fn)]
                        )
                        cont_m = (
                            _aggregate_secret_prob(all_probs_ablate_m[layer_idx], ids_m, target_id_taboo, start_idx)
                            if target_id_taboo >= 0 else None
                        )
                        content_rows.append({"m": m, "condition": "taboo_ablated", "content": cont_m})

                    # Save content table
                    with open(os.path.join(ex_dir, "content_curve.json"), "w") as f:
                        json.dump(content_rows, f, indent=2)
                    with open(os.path.join(ex_dir, "content_curve.tsv"), "w") as f:
                        f.write("m\tcondition\tcontent\n")
                        for r in content_rows:
                            f.write(f"{r['m']}\t{r['condition']}\t{r['content'] if r['content'] is not None else 'NA'}\n")

                    # 4) logit-lens heatmaps (base / taboo / ablated) — use each model's own transcript
                    print("    Creating heatmaps...")
                    plotting_cfg = cfg.get("plotting", {"dpi": 300})
                    if target_id_taboo >= 0:
                        # Recompute taboo heatmap on taboo transcript
                        all_probs_taboo_heat, words_taboo_heat, _ = _compute_all_layer_probs(
                            tm.hooked, tm.tokenizer, taboo_full_text, tm.device, fwd_hooks=None
                        )
                        # Find assistant start for taboo transcript
                        def _find_resp_start(tokens: List[str]) -> int:
                            idxs = [i for i, tok in enumerate(tokens) if tok == "<start_of_turn>"]
                            return idxs[1] + 3 if len(idxs) >= 2 else 0
                        taboo_start_idx = _find_resp_start(words_taboo_heat)
                        _save_heatmap(
                            os.path.join(ex_dir, "heatmap_taboo.png"),
                            all_probs_taboo_heat,
                            tm.tokenizer,
                            target_id_taboo,
                            words_taboo_heat,
                            taboo_start_idx,
                            plotting_cfg,
                        )
                    hook_name = cfg["sae"]["resid_hook_name"]
                    hook_fn = tm.make_sae_ablation_hook(features_for_gen, "last_token")
                    all_probs_ablate_gen, words_ablate_gen, _ = _compute_all_layer_probs(
                        tm.hooked, tm.tokenizer, ablated_full_text, tm.device, fwd_hooks=[(hook_name, hook_fn)]
                    )
                    if target_id_taboo >= 0:
                        _save_heatmap(
                            os.path.join(ex_dir, f"heatmap_ablated_m{M_FOR_GENERATION}.png"),
                            all_probs_ablate_gen,
                            tm.tokenizer,
                            target_id_taboo,
                            words_ablate_gen,
                            _find_resp_start(words_ablate_gen),
                            plotting_cfg,
                        )

                    # Summary row
                    summary_index.append(
                        {
                            "word": word,
                            "prompt_index": pi + 1,
                            "prompt": user_prompt,
                            "features_for_generation": features_for_gen,
                            "content_base": content_base,
                            "content_taboo": content_taboo,
                            "content_ablated_m": next(
                                (
                                    r["content"]
                                    for r in content_rows
                                    if r["condition"] == "taboo_ablated" and r["m"] == M_FOR_GENERATION
                                ),
                                None,
                            ),
                            "artifacts_dir": ex_dir,
                        }
                    )
                finally:
                    if tm is not None:
                        tm.close()
                        tm = None
                    clean_gpu_memory()

        finally:
            if tm is not None:
                tm.close()
            clean_gpu_memory()

    # Save a top-level manifest
    with open(os.path.join(root, "index.json"), "w") as f:
        json.dump(
            {
                "config_used": {
                    "layer_idx": layer_idx,
                    "budgets": BUDGETS,
                    "m_for_generation": M_FOR_GENERATION,
                    "max_examples_per_word": MAX_EXAMPLES_PER_WORD,
                },
                "case_studies": summary_index,
            },
            f,
            indent=2,
        )
    print(f"\n[06] Case studies done. Artifacts at: {root}")


def main():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Ablation case studies: responses + lens heatmaps (base vs taboo vs ablated)."
        )
    )
    p.add_argument("--config", type=str, default="configs/defaults.yaml")
    args = p.parse_args()
    run_case_studies(args.config)


if __name__ == "__main__":
    main()
