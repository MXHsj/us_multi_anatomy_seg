from __future__ import annotations

import subprocess
import sys

import torch


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BENCHMARK_SCRIPT = "benchmarks/ultrasam_inference.py"
DATASET_ROOT = "datasets/GIST514"
OUTPUT_DIR = "results/ultrasam_gt_bbox_gist514"

DEFAULT_ARGS = [
    "--dataset",
    "gist514",
    "--dataset-root",
    DATASET_ROOT,
    "--max-samples",
    "all",
    "--box-padding",
    "0",
    "--device",
    DEVICE,
    "--save-vis",
    "20",
    "--output-dir",
    OUTPUT_DIR,
]


def main() -> None:
    cmd = [sys.executable, BENCHMARK_SCRIPT, *DEFAULT_ARGS, *sys.argv[1:]]
    print(f"Starting UltraSAM GT benchmark: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
