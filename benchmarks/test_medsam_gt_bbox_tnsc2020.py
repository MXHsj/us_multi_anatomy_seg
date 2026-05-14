from __future__ import annotations

import subprocess
import sys

import torch


def mps_is_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if mps_is_available():
        return "mps"
    return "cpu"


DEVICE = default_device()
BENCHMARK_SCRIPT = "benchmarks/medsam_inference.py"
DATASET_ROOT = "datasets/TNSC2020"
OUTPUT_DIR = "results/medsam_gt_bbox_tnsc2020"

DEFAULT_ARGS = [
    "--dataset",
    "tnsc2020",
    "--dataset-root",
    DATASET_ROOT,
    "--max-samples",
    "100",
    "--bbox-jitter-prob",
    "0.0",
    "--device",
    DEVICE,
    "--save-vis",
    "20",
    "--output-dir",
    OUTPUT_DIR,
]


def main() -> None:
    print(f"Starting MedSAM GT benchmark: {' '.join(DEFAULT_ARGS)}", flush=True)

    cmd = [sys.executable, BENCHMARK_SCRIPT, *DEFAULT_ARGS, *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
