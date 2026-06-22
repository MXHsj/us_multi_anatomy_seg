#!/usr/bin/env python3
"""Run MedSAM3 (text prompts) on selected datasets sequentially.

Toggle each dataset below to True/False, then run:

    python work_dir/MedSAM3/run_medsam3.py

Any extra CLI args are forwarded to every benchmark, e.g.:

    python work_dir/MedSAM3/run_medsam3.py --max-samples 50 --device cpu
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
import os


# Repo root is two levels up from this file (work_dir/MedSAM3/..).
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = REPO_ROOT / ".env"

assert os.path.exists(ENV_FILE_PATH), f".env file not found at path {ENV_FILE_PATH}"

load_dotenv(ENV_FILE_PATH)

assert os.environ.get("HF_TOKEN"), f"HF_TOKEN must be set in .env file before using medsam3"

# Toggle which datasets to run.
DATASETS: dict[str, bool] = {
    "aulid": False,
    "blusg": False,
    "busbra": False,
    "busi": False,
    "camus": False,
    "oku": False,
    "roblus": False,
    "tnsc2020": True,
    "ultrabones100k": False,
    "uns": False,
}


def main() -> None:
    extra_args = sys.argv[1:]
    selected = [name for name, enabled in DATASETS.items() if enabled]
    if not selected:
        print("No datasets enabled. Set at least one entry in DATASETS to True.")
        return

    print(f"Running MedSAM3 on {len(selected)} dataset(s): {', '.join(selected)}")

    failures: list[str] = []
    for name in selected:
        script = REPO_ROOT / "benchmarks" / f"test_medsam3_text_prompt_{name}.py"
        if not script.exists():
            print(f"[SKIP] {name}: script not found ({script})")
            failures.append(name)
            continue

        cmd = [sys.executable, str(script), *extra_args]
        print(f"\n{'=' * 70}\n[RUN] {name}\n{' '.join(cmd)}\n{'=' * 70}", flush=True)
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            print(f"[FAIL] {name} exited with code {result.returncode}")
            failures.append(name)
        else:
            print(f"[DONE] {name}")

    print(f"\n{'=' * 70}")
    if failures:
        print(f"Completed with {len(failures)} failure(s): {', '.join(failures)}")
        sys.exit(1)
    print("All selected datasets completed successfully.")


if __name__ == "__main__":
    main()
