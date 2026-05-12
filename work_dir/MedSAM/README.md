MedSAM checkpoints are ignored by git.

By default, `benchmarks/medsam_inference.py` downloads `medsam_vit_b.pth` here from the Hugging Face model repo `GleghornLab/medsam-vit-b` when the file is missing.

For local-only runs, place `medsam_vit_b.pth` in this folder manually and run the benchmark with `--no-auto-download-checkpoint`.
