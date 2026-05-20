from __future__ import annotations

import platform
import subprocess
import sys


DEVICE = "mps" if platform.system() == "Darwin" else "cuda:0"
BENCHMARK_SCRIPT = "benchmarks/medsam_inference.py"
DATASET_ROOT = "datasets/RobLUS"
OUTPUT_DIR = "results/medsam_gt_bbox_roblus"

DEFAULT_ARGS = [
    "--dataset",
    "roblus",
    "--dataset-root",
    DATASET_ROOT,
    "--max-samples",
    "764",
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
    cmd = [sys.executable, BENCHMARK_SCRIPT, *DEFAULT_ARGS, *sys.argv[1:]]
    print(f"Starting MedSAM GT benchmark: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
