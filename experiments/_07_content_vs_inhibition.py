# experiments/_07_content_vs_inhibition.py
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import json
from typing import Dict, Any, List

import numpy as np

from utils import load_yaml, ensure_dir
from plots import plot_content_vs_inhibition


def _collect_points_ablation(j: Dict[str, Any]) -> List[Dict[str, Any]]:
    pts = []
    for wres in j["per_word"]:
        word = wres["word"]
        for row in wres["results"]:
            m = row["m"]
            for kind in ["targeted", "random"]:
                pts.append(
                    {
                        "word": word,
                        "exp": "ablation",
                        "kind": kind,
                        "budget_m": m,
                        "content": row[kind]["content"],
                        "inhib_postgame": row[kind]["inhib_postgame"],
                        "inhib_pregame": row[kind]["inhib_pregame"],
                    }
                )
    return pts


def _collect_points_noise(j: Dict[str, Any]) -> List[Dict[str, Any]]:
    pts = []
    for wres in j["per_word"]:
        word = wres["word"]
        for row in wres["results"]:
            mag = row["magnitude"]
            for kind in ["targeted", "random"]:
                pts.append(
                    {
                        "word": word,
                        "exp": "noise",
                        "kind": kind,
                        "magnitude": mag,
                        "content": row[kind]["content"],
                        "inhib_postgame": row[kind]["inhib_postgame"],
                        "inhib_pregame": row[kind]["inhib_pregame"],
                    }
                )
    return pts


def main(config_path: str = "configs/defaults.yaml"):
    cfg = load_yaml(config_path)
    out_dir = os.path.join(cfg["paths"]["results_dir"], "analysis")
    ensure_dir(out_dir)

    abl_path = os.path.join(
        cfg["paths"]["results_dir"], "ablation", "sae_ablation_results.json"
    )
    noi_path = os.path.join(
        cfg["paths"]["results_dir"], "noise", "noise_injection_results.json"
    )

    if not (os.path.exists(abl_path) and os.path.exists(noi_path)):
        print("[07] Missing ablation/noise results. Run 04 and 05 first.")
        return

    with open(abl_path, "r") as f:
        j_ab = json.load(f)
    with open(noi_path, "r") as f:
        j_no = json.load(f)

    pts = _collect_points_ablation(j_ab) + _collect_points_noise(j_no)

    # Build a per-condition scatter: (content, inhibition_postgame)
    x = np.array([p["content"] for p in pts], dtype=float)
    y = np.array([p["inhib_postgame"] for p in pts], dtype=float)
    labels = [
        f"{p['exp']}-{p['kind']}:{p.get('budget_m', p.get('magnitude'))}-{p['word']}"
        for p in pts
    ]

    fig = plot_content_vs_inhibition(
        x, y, labels, title="Content vs Inhibition (postgame)"
    )
    out_png = os.path.join(out_dir, "content_vs_inhibition_postgame.png")
    fig.savefig(out_png, bbox_inches="tight", dpi=cfg["output"].get("dpi", 300))
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"[07] Saved {out_png}")

    # Also pregame
    y2 = np.array([p["inhib_pregame"] for p in pts], dtype=float)
    fig2 = plot_content_vs_inhibition(
        x, y2, labels, title="Content vs Inhibition (pregame)"
    )
    out_png2 = os.path.join(out_dir, "content_vs_inhibition_pregame.png")
    fig2.savefig(out_png2, bbox_inches="tight", dpi=cfg["output"].get("dpi", 300))
    plt.close(fig2)
    print(f"[07] Saved {out_png2}")


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "configs/defaults.yaml"
    main(path)
