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
    "1.0",
    "--bbox-jitter-fraction",
    "0.2",
    "--device",
    "mps",
    "--save-vis",
    "20",
    "--output-dir",
    "results/medsam_jitter_bbox_tnsc2020",
]


def main() -> None:
    cmd = ["python", "benchmarks/medsam_inference.py", *DEFAULT_ARGS, *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
