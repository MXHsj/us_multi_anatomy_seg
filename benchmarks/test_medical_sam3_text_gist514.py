from __future__ import annotations

import platform
import subprocess
import sys


DEVICE = "cpu" if platform.system() == "Darwin" else "cuda:0"
BENCHMARK_SCRIPT = "benchmarks/medical_sam3_inference.py"
DATASET_ROOT = "datasets/GIST514"
OUTPUT_DIR = "results/medicalsam3_text_prompt_gist514"

DEFAULT_ARGS = [
    "--dataset",
    "gist514",
    "--dataset-root",
    DATASET_ROOT,
    "--max-samples",
    "all",
    "--device",
    DEVICE,
    "--save-vis",
    "20",
    "--output-dir",
    OUTPUT_DIR,
]


def main() -> None:
    cmd = [sys.executable, BENCHMARK_SCRIPT, *DEFAULT_ARGS, *sys.argv[1:]]
    print(f"Starting Medical SAM3 text benchmark: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
