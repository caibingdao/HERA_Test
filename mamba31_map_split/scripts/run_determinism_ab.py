#!/usr/bin/env python3
"""Run the legacy SAC microbenchmark with determinism explicitly selected."""

import argparse
import os
import runpy
import sys


def parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--deterministic", type=parse_bool, required=True)
args, benchmark_args = parser.parse_known_args()

import torch

if args.deterministic:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
else:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)

benchmark = (
    "/data/home/lzy/cbd/code/ARiADNE_MAMBA_Test/"
    "ARiADNE_MAMBA_3.1/benchmarks/profile_sac_update.py"
)
sys.argv = [benchmark, *benchmark_args]
runpy.run_path(benchmark, run_name="__main__")
