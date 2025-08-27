# harness.py
import argparse
import warnings
import torch
from typing import Any, Dict

# Silence Torch TF32 performance suggestion from Inductor without changing behavior.
warnings.filterwarnings(
    "ignore",
    message=(
        ".*TensorFloat32 tensor cores for float32 matrix multiplication available "
        "but not enabled.*"
    ),
    category=UserWarning,
    module="torch._inductor.compile_fx",
)

# Enable TF32 for faster matmul/conv on Ampere+ GPUs.
try:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    # Safe fallback if running on CPU-only or older backends
    pass

from utils import load_yaml
from run_generation import main as cache_main
from experiments._01_reproduce_logit_lens import main as logit_lens_main
from experiments._02_run_sae_baseline import main as sae_baseline_main
from experiments._03_run_token_forcing import main as token_forcing_main
from experiments._04_run_sae_ablation import main as sae_ablation_main
from experiments._05_run_noise_injection import main as noise_injection_main
from experiments._07_content_vs_inhibition import main as cvi_main

def _add_common(parser):
    parser.add_argument("--config", type=str, default="../configs/defaults.yaml")

def cli():
    ap = argparse.ArgumentParser("Taboo Elicitation Harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_cache = sub.add_parser("cache", help="Generate & cache traces and residuals.")
    _add_common(p_cache)

    p_ll = sub.add_parser("logit_lens", help="Reproduce Logit Lens experiment")
    _add_common(p_ll)

    p_sae = sub.add_parser("sae_baseline", help="Run SAE top-k baseline")
    _add_common(p_sae)

    p_tf = sub.add_parser("token_forcing", help="Run token forcing baseline")
    _add_common(p_tf)

    p_ab = sub.add_parser("sae_ablation", help="Run SAE latent ablation experiments")
    _add_common(p_ab)

    p_noise = sub.add_parser("noise_injection", help="Run residual noise injection experiments")
    _add_common(p_noise)

    p_cvi = sub.add_parser("content_vs_inhibition", help="Analyze content vs inhibition trade-off and plot")
    _add_common(p_cvi)

    args = ap.parse_args()
    cfg_path = args.config

    if args.cmd == "cache":
        cache_main(cfg_path)
    elif args.cmd == "logit_lens":
        logit_lens_main(cfg_path)
    elif args.cmd == "sae_baseline":
        sae_baseline_main(cfg_path)
    elif args.cmd == "token_forcing":
        token_forcing_main(cfg_path)
    elif args.cmd == "sae_ablation":
        sae_ablation_main(cfg_path)
    elif args.cmd == "noise_injection":
        noise_injection_main(cfg_path)
    elif args.cmd == "content_vs_inhibition":
        cvi_main(cfg_path)

if __name__ == "__main__":
    cli()
