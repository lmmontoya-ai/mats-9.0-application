# experiments/_03_run_token_forcing.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import re
import json
import hashlib
from typing import Dict, Any, List, Optional

import numpy as np
from transformers import set_seed

from models import TabooModel
from utils import load_yaml, clean_gpu_memory, response_contains_word
from metrics import summarize_token_forcing

import pandas as pd


def _sha(items: List[str]) -> str:
    h = hashlib.sha256()
    for s in items:
        h.update(s.encode("utf-8"))
    return h.hexdigest()


def generate_forced_response(
    base_tm: TabooModel,
    prefill_phrase: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    max_new_tokens: int = 50,
) -> str:
    """Greedy decoding conditioned on an assistant prefill phrase."""
    if chat_history is None:
        chat_history = []
    if len(chat_history) == 0 or chat_history[-1]["role"] == "assistant":
        chat_history = chat_history + [{"role": "user", "content": ""}]
    convo = chat_history + [{"role": "assistant", "content": prefill_phrase}]
    fmt = base_tm.tokenizer.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=False
    )
    fmt = fmt.rsplit("<end_of_turn>", 1)[0]
    inputs = base_tm.tokenizer(fmt, return_tensors="pt").to(base_tm.base_model.device)
    with base_tm.base_model.no_grad():
        pass  # type: ignore[attr-defined]
    from torch import no_grad

    with no_grad():
        out = base_tm.base_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )
    # tail only
    cont = base_tm.tokenizer.decode(
        out.sequences[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    return (prefill_phrase + cont).strip()


def warmup_history(
    tm: TabooModel, prompts: List[str], max_new_tokens: int
) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for p in prompts:
        history.append({"role": "user", "content": p})
        text = tm.generate_assistant(p, max_new_tokens=max_new_tokens)
        history.append({"role": "assistant", "content": text})
    return history


def main(config_path: str = "configs/defaults.yaml"):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["experiment"]["seed"]))

    words = list(cfg["word_plurals"].keys())
    prefill_phrases = cfg["prefill_phrases"]
    max_new = int(cfg["experiment"]["max_new_tokens"])

    # evaluate
    all_results: Dict[str, Dict[str, List[bool]]] = {}

    for w in words:
        print(f"\n[03] Word: {w}")
        tm = TabooModel(w, cfg)
        try:
            pregame = []
            for phrase in prefill_phrases:
                resp = generate_forced_response(
                    tm, phrase, chat_history=None, max_new_tokens=max_new
                )
                ok = response_contains_word(resp, w, cfg["word_plurals"][w])
                pregame.append(ok)
                print(f"  [pre] '{phrase}' -> ok={ok}")
            # postgame: warm up with three baseline hints
            hist = warmup_history(tm, cfg["prompts"][:3], max_new)
            postgame = []
            for phrase in prefill_phrases:
                resp = generate_forced_response(
                    tm, phrase, chat_history=hist, max_new_tokens=max_new
                )
                ok = response_contains_word(resp, w, cfg["word_plurals"][w])
                postgame.append(ok)
                print(f"  [post] '{phrase}' -> ok={ok}")
            all_results[w] = {"pregame": pregame, "postgame": postgame}
        finally:
            tm.close()

    # summarize + save
    import pandas as pd

    df_rows = []
    for w, d in all_results.items():
        pr = d["pregame"]
        po = d["postgame"]
        df_rows.append(
            {
                "word": w,
                "condition": "pregame",
                "success_rate": float(np.mean(pr)),
                "successes": int(sum(pr)),
                "total": len(pr),
            }
        )
        df_rows.append(
            {
                "word": w,
                "condition": "postgame",
                "success_rate": float(np.mean(po)),
                "successes": int(sum(po)),
                "total": len(po),
            }
        )
    df = pd.DataFrame(df_rows)
    out_path = os.path.join(
        cfg["paths"]["results_dir"], "tables", "token_forcing_baseline.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[03] Saved {out_path}\n")
    print(df.to_string(index=False))

    # fingerprint for sanity
    fp = {
        "seed": cfg["experiment"]["seed"],
        "max_new_tokens": cfg["experiment"]["max_new_tokens"],
        "layer_idx": cfg["model"]["layer_idx"],
        "prompts_hash": _sha(cfg["prompts"]),
        "prefill_hash": _sha(cfg["prefill_phrases"]),
    }
    fp_path = os.path.join(os.path.dirname(out_path), "token_forcing_fingerprint.json")
    with open(fp_path, "w") as f:
        json.dump(fp, f, indent=2)
    print(f"Fingerprint -> {fp_path}")


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "configs/defaults.yaml"
    main(path)
