from __future__ import annotations

import platform
import subprocess
import sys


DEVICE = "cpu" if platform.system() == "Darwin" else "cuda:0"
BENCHMARK_SCRIPT = "benchmarks/medsam3_inference.py"
DATASET_ROOT = "datasets/MMOTU"
OUTPUT_DIR = "results/medsam3_text_prompt_mmotu"

DEFAULT_ARGS = [
    "--dataset",
    "mmotu",
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
    print(f"Starting MedSAM3 text benchmark: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
