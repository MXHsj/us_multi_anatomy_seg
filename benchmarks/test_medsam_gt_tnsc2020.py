from __future__ import annotations

import subprocess
import sys


DEFAULT_ARGS = [
    "--dataset",
    "tnsc2020",
    "--dataset-root",
    "datasets/TNSC2020",
    "--max-samples",
    "100",
    "--bbox-jitter-prob",
    "0.0",
    "--device",
    "mps",
    "--save-vis",
    "20",
    "--output-dir",
    "results/medsam_gt_tnsc2020",
]


def main() -> None:
    cmd = ["python3", "./medsam_inference.py", *DEFAULT_ARGS, *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
