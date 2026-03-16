from __future__ import annotations

import subprocess
import sys


DEFAULT_ARGS = [
    "--dataset",
    "oku",
    "--dataset-root",
    "datasets/OKU",
    "--max-samples",
    "30",
    "--device",
    "mps",
    "--output-dir",
    "results/medsam_oku",
    "--oku-anatomy",
    "Capsule",
]


def main() -> None:
    cmd = ["python", "benchmarks/medsam_inference.py", *DEFAULT_ARGS, *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
