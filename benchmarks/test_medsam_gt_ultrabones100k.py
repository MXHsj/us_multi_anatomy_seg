from __future__ import annotations

import subprocess
import sys


DEFAULT_ARGS = [
    "--dataset",
    "ultrabones100k",
    "--dataset-root",
    "datasets/UltraBones100k",
    "--max-samples",
    "100",
    "--box-padding",
    "10",
    "--bbox-jitter-prob",
    "0.0",
    "--device",
    "cuda:0",
    "--save-vis",
    "20",
    "--output-dir",
    "results/medsam_gt_ultrabones100k",
]


def main() -> None:
    cmd = ["python", "benchmarks/medsam_inference.py", *DEFAULT_ARGS, *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()


# python benchmarks\test_medsam_gt_ultrabones100k.py `
#   --dataset-root F:\UltraBones100k `
#   --checkpoint work_dir\MedSAM\medsam_vit_b.pth `
#   --device cuda:0 `
#   --max-samples 5 `
#   --box-padding 10 `
#   --save-vis 5 `
#   --output-dir results\medsam_gt_ultrabones100k_smoke
