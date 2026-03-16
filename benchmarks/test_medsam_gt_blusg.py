from __future__ import annotations

import subprocess
import sys


DEFAULT_ARGS = [
    "--dataset",
    "blusg",
    "--dataset-root",
    "datasets/BLUSG",
    "--max-samples",
    "30",
    "--device",
    "mps",
    "--output-dir",
    "results/medsam_blusg",
]


def main() -> None:
    cmd = ["python", "benchmarks/medsam_inference.py", *DEFAULT_ARGS, *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
