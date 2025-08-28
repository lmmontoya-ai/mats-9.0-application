#!/usr/bin/env python3
"""
Prepare local cache data for the harness.

Checks whether `data/processed` (or the configured cache_dir) already
contains cached pairs. If not, downloads them from a Hugging Face repo
and places them into the configured cache directory.

Default HF repo: `Luxel/taboo-brittleness` (as provided by the user).

Usage examples:

  python scripts/prepare_data.py \
    --config configs/default.yaml \
    --repo Luxel/taboo-brittleness

  python scripts/prepare_data.py --force   # force re-download/copy

This script does not require GPU.
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict, List, Optional, Tuple

from huggingface_hub import snapshot_download

# Ensure we can import local utils.py when running as a script
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import load_yaml, ensure_dir


def _list_words_from_cfg(cfg: Dict) -> List[str]:
    wp = cfg.get("word_plurals", {})
    return list(wp.keys())


def _has_npz_files(path: str) -> bool:
    for root, _, files in os.walk(path):
        if any(f.endswith(".npz") for f in files):
            return True
    return False


def _has_data_for_words(cache_dir: str, words: List[str]) -> bool:
    # If any .npz exists under the cache_dir, we consider it populated.
    # For stricter checking, ensure at least one .npz per expected word.
    if not os.path.isdir(cache_dir):
        return False
    # Fast path: any .npz anywhere under cache_dir
    if _has_npz_files(cache_dir):
        return True
    # Stricter path: per word
    for w in words:
        wdir = os.path.join(cache_dir, w)
        if os.path.isdir(wdir) and _has_npz_files(wdir):
            return True
    return False


def _candidate_processed_dirs(base: str) -> List[str]:
    candidates = []
    for p in [
        base,
        os.path.join(base, "data", "processed"),
        os.path.join(base, "processed"),
        os.path.join(base, "data"),
    ]:
        if os.path.isdir(p):
            candidates.append(p)
    return candidates


def _score_source_dir(path: str, words: List[str]) -> int:
    # Score by number of word-subdirs containing at least one .npz
    score = 0
    for w in words:
        wdir = os.path.join(path, w)
        if os.path.isdir(wdir) and _has_npz_files(wdir):
            score += 1
    # Fallback: any .npz anywhere
    if score == 0 and _has_npz_files(path):
        score = 1
    return score


def _detect_source_processed_dir(local_repo: str, words: List[str]) -> Optional[str]:
    # Try common locations first, then pick the best scored directory.
    candidates = _candidate_processed_dirs(local_repo)
    best_path: Optional[str] = None
    best_score = -1
    for cand in candidates:
        score = _score_source_dir(cand, words)
        if score > best_score:
            best_score = score
            best_path = cand
    return best_path if best_score > 0 else None


def _copy_tree(src: str, dst: str, force: bool = False) -> None:
    ensure_dir(dst)
    # Copy files recursively, allowing overwrite
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out_dir = dst if rel == "." else os.path.join(dst, rel)
        ensure_dir(out_dir)
        for d in dirs:
            ensure_dir(os.path.join(out_dir, d))
        for f in files:
            src_f = os.path.join(root, f)
            dst_f = os.path.join(out_dir, f)
            if os.path.exists(dst_f) and not force:
                continue
            shutil.copy2(src_f, dst_f)


def prepare_data(
    config_path: str,
    repo_id: str,
    repo_type: str = "dataset",
    cache_dir_override: Optional[str] = None,
    force: bool = False,
) -> Tuple[str, bool]:
    """
    Returns (cache_dir, already_present) where already_present is True if data was
    already present locally before any download/copy actions.
    """
    cfg = load_yaml(config_path)
    words = _list_words_from_cfg(cfg)
    cache_dir = cache_dir_override or cfg["paths"]["cache_dir"]

    if _has_data_for_words(cache_dir, words) and not force:
        print(f"[prepare] Found existing data in '{cache_dir}'. Skipping download.")
        return cache_dir, True

    print(f"[prepare] No local data found in '{cache_dir}' (or force requested).")
    print(f"[prepare] Downloading snapshot from Hugging Face: {repo_id} (type={repo_type})")

    local_repo = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir_use_symlinks=False,
        # Cautious default: keep in hf-cache inside repo to avoid polluting cwd
        local_dir=os.path.join(".hf-cache", repo_id.replace("/", "__")),
        ignore_patterns=["**/.git*", "**/README*", "**/LICENSE*"],
    )

    src_processed = _detect_source_processed_dir(local_repo, words)
    if src_processed is None:
        raise RuntimeError(
            "Could not locate a 'processed' data directory in the downloaded repo. "
            "Expected something like '<repo>/data/processed' or word-subdirectories "
            "with .npz files. Please check the repository layout."
        )

    print(f"[prepare] Copying processed data from: {src_processed}")
    _copy_tree(src_processed, cache_dir, force=force)
    print(f"[prepare] Data is ready in '{cache_dir}'.")
    return cache_dir, False


def main():
    ap = argparse.ArgumentParser("Prepare processed cache data for harness experiments")
    ap.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config (to read cache_dir and words).",
    )
    ap.add_argument(
        "--repo",
        type=str,
        default="Luxel/taboo-brittleness",
        help="Hugging Face repository id containing processed data.",
    )
    ap.add_argument(
        "--repo-type",
        type=str,
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Hugging Face repo type.",
    )
    ap.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override output cache directory (defaults to cfg paths.cache_dir).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force copying over existing files.",
    )
    args = ap.parse_args()

    prepare_data(
        config_path=args.config,
        repo_id=args.repo,
        repo_type=args.repo_type,
        cache_dir_override=args.cache_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
