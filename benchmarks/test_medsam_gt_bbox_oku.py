from __future__ import annotations

import subprocess
import sys


BENCHMARK_SCRIPT = "benchmarks/medsam_inference.py"

DEFAULT_ARGS = [
    "--dataset",
    "oku",
    "--dataset-root",
    "datasets/OKU",
    "--max-samples",
    "all",
    "--bbox-jitter-prob",
    "0.0",
    "--device",
    "mps",
    "--save-vis",
    "20",
    "--output-dir",
    "results/medsam_gt_bbox_oku",
    "--oku-anatomy",
    "Capsule",
]


def main() -> None:
    cmd = [sys.executable, BENCHMARK_SCRIPT, *DEFAULT_ARGS, *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

