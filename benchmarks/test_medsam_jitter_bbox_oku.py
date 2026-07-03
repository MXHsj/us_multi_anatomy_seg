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
    "1.0",
    "--bbox-jitter-fraction",
    "0.2",
    "--device",
    "mps",
    "--save-vis",
    "20",
    "--output-dir",
    "results/medsam_jitter_bbox_oku",
    "--oku-anatomy",
    "Capsule",
]


def main() -> None:
    cmd = [sys.executable, BENCHMARK_SCRIPT, *DEFAULT_ARGS, *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
