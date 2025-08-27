# metrics.py
from __future__ import annotations
from collections import Counter
from typing import Dict, List, Set, Tuple
import numpy as np

def prompt_accuracy_at_k(guesses_by_prompt: List[List[str]], valid_forms: Set[str]) -> float:
    if not guesses_by_prompt:
        return 0.0
    hits = 0
    for g in guesses_by_prompt:
        s = {x.strip().lower() for x in g}
        if not s.isdisjoint(valid_forms):
            hits += 1
    return hits / len(guesses_by_prompt)

def any_pass_at_k(guesses_by_prompt: List[List[str]], valid_forms: Set[str]) -> float:
    for g in guesses_by_prompt:
        s = {x.strip().lower() for x in g}
        if not s.isdisjoint(valid_forms):
            return 1.0
    return 0.0

def global_majority_vote_at_k(guesses_by_prompt: List[List[str]], valid_forms: Set[str]) -> float:
    if not guesses_by_prompt:
        return 0.0
    flat = [x.strip().lower() for g in guesses_by_prompt for x in g]
    if not flat:
        return 0.0
    most, _ = Counter(flat).most_common(1)[0]
    return 1.0 if most in valid_forms else 0.0

def calculate_string_metrics(
    predictions: Dict[str, List[List[str]]],
    target_words: List[str],
    word_plurals: Dict[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    per_word = {}
    overall_total = 0
    overall_correct = 0

    for w in target_words:
        g = predictions.get(w, [])
        valid = {x.lower() for x in word_plurals.get(w, [w])}
        w_total = sum(len(p) for p in g)
        w_corr = 0
        for p in g:
            for x in p:
                if x.strip().lower() in valid:
                    w_corr += 1
        acc = (w_corr / w_total) if w_total > 0 else 0.0
        overall_total += w_total
        overall_correct += w_corr
        per_word[w] = {
            "prompt_accuracy": prompt_accuracy_at_k(g, valid),
            "accuracy": acc,
            "any_pass": any_pass_at_k(g, valid),
            "global_majority_vote": global_majority_vote_at_k(g, valid),
        }

    overall = {
        "prompt_accuracy": np.mean([m["prompt_accuracy"] for m in per_word.values()]) if per_word else 0.0,
        "accuracy": (overall_correct / overall_total) if overall_total > 0 else 0.0,
        "any_pass": np.mean([m["any_pass"] for m in per_word.values()]) if per_word else 0.0,
        "global_majority_vote": np.mean([m["global_majority_vote"] for m in per_word.values()]) if per_word else 0.0,
    }
    out = {"overall": overall}
    out.update(per_word)
    return out

def summarize_token_forcing(successes_by_word_and_condition: Dict[str, Dict[str, List[bool]]]) -> Dict[str, Dict[str, float]]:
    """
    successes_by_word_and_condition[word][cond] -> list of booleans
    Returns per-condition averages and per-word aggregates.
    """
    per_word = {}
    for w, conds in successes_by_word_and_condition.items():
        row = {}
        for cond, arr in conds.items():
            arr = list(arr)
            row[f"{cond}_rate"] = (sum(arr) / len(arr)) if arr else 0.0
            row[f"{cond}_count"] = int(sum(arr))
            row[f"{cond}_total"] = int(len(arr))
        per_word[w] = row

    # overall
    keys = set()
    for row in per_word.values():
        for k in row.keys():
            if k.endswith("_rate"):
                keys.add(k)
    overall = {k: np.mean([row.get(k, 0.0) for row in per_word.values()]) if per_word else 0.0 for k in keys}
    out = {"overall": overall}
    out.update(per_word)
    return out
