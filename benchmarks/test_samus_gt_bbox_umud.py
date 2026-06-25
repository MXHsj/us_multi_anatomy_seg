from __future__ import annotations

import platform
import subprocess
import sys


DEVICE = "mps" if platform.system() == "Darwin" else "cuda:0"
BENCHMARK_SCRIPT = "benchmarks/samus_inference.py"
DATASET_ROOT = "datasets/UMUD"
OUTPUT_DIR = "results/samus_gt_bbox_umud"

DEFAULT_ARGS = [
    "--dataset",
    "umud",
    "--dataset-root",
    DATASET_ROOT,
    "--max-samples",
    "all",
    "--device",
    DEVICE,
    "--save-vis",
    "99999999",
    "--output-dir",
    OUTPUT_DIR,
]


def main() -> None:
    cmd = [sys.executable, BENCHMARK_SCRIPT, *DEFAULT_ARGS, *sys.argv[1:]]
    print(f"Starting SAMUS GT benchmark: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
