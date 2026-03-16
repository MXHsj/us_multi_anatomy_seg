from __future__ import annotations

import subprocess
import sys


DEFAULT_ARGS = [
    "--dataset",
    "tnsc2020",
    "--dataset-root",
    "datasets/TNSC2020",
    "--max-samples",
    "30",
    "--device",
    "mps",
    "--output-dir",
    "results/medsam_tnsc2020",
]


def main() -> None:
    cmd = ["python", "benchmarks/medsam_inference.py", *DEFAULT_ARGS, *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
