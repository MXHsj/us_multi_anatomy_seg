from __future__ import annotations

import subprocess
import sys


BENCHMARK_SCRIPT = "benchmarks/medsam_inference.py"

DEFAULT_ARGS = [
    "--dataset",
    "ultrabones100k",
    "--dataset-root",
    "datasets/UltraBones100k",
    "--max-samples",
    "all",
    "--box-padding",
    "10",
    "--bbox-jitter-prob",
    "1.0",
    "--bbox-jitter-fraction",
    "0.2",
    "--device",
    "cuda:0",
    "--save-vis",
    "20",
    "--output-dir",
    "results/medsam_jitter_bbox_ultrabones100k",
]


def main() -> None:
    cmd = [sys.executable, BENCHMARK_SCRIPT, *DEFAULT_ARGS, *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
