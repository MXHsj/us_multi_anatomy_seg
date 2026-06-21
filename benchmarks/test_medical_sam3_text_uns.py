from __future__ import annotations

import platform
import subprocess
import sys


DEVICE = "cpu" if platform.system() == "Darwin" else "cuda:0"
BENCHMARK_SCRIPT = "benchmarks/medical_sam3_inference.py"
DATASET_ROOT = "datasets/UNS"
OUTPUT_DIR = "results/medicalsam3_text_prompt_uns"

# Default concept label per dataset comes from datasets/label_text.py plus the loader
# defaults (camus=1,3 / oku=Capsule / aulid=mass / roblus=pleural_line); no overrides
# here, matching the box wrappers. Pass extra flags through argv to override.
DEFAULT_ARGS = [
    "--dataset",
    "uns",
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
