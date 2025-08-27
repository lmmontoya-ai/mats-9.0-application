# utils.py
from __future__ import annotations
import os
import re
import gc
import yaml
from typing import Dict, Any, List

import torch as t

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def set_global_determinism():
    if t.cuda.is_available():
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = False
    if hasattr(t.backends, "mps") and t.backends.mps.is_available():
        if hasattr(t.backends.mps, "deterministic"):
            try:
                t.backends.mps.deterministic = True  # type: ignore[attr-defined]
            except Exception:
                pass

def pick_device_and_dtype(cfg: Dict[str, Any]):
    device_name = cfg["experiment"].get("device", "auto")
    if device_name == "auto":
        if t.cuda.is_available():
            device = t.device("cuda")
        elif hasattr(t.backends, "mps") and t.backends.mps.is_available():
            device = t.device("mps")
        else:
            device = t.device("cpu")
    else:
        device = t.device(device_name)

    if device.type == "cuda":
        dtype = t.bfloat16
    elif device.type == "mps":
        dtype = t.float16
    else:
        dtype = t.float32
    return device, dtype

def clean_gpu_memory():
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()
        t.cuda.reset_peak_memory_stats()
        t.cuda.synchronize()
    elif hasattr(t.backends, "mps") and t.backends.mps.is_available():
        try:
            t.mps.empty_cache()  # type: ignore[attr-defined]
        except Exception:
            pass

# Regex helper used by token forcing
def response_contains_word(response: str, secret_word: str, valid_plurals: List[str]) -> bool:
    txt = response.lower()
    forms = [secret_word.lower()] + [p.lower() for p in valid_plurals]
    for f in set(forms):
        if re.search(r"\b" + re.escape(f) + r"\b", txt):
            return True
    return False
