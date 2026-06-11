# UltraSAM

Commands to run UltraSAM (GT box prompts) on each dataset. Run from the repo root.
The checkpoint `work_dir/UltraSam/UltraSam.pth` is auto-downloaded on first run.

```bash
python benchmarks/test_ultrasam_gt_bbox_aulid.py
python benchmarks/test_ultrasam_gt_bbox_blusg.py
python benchmarks/test_ultrasam_gt_bbox_busbra.py
python benchmarks/test_ultrasam_gt_bbox_busi.py
python benchmarks/test_ultrasam_gt_bbox_camus.py
python benchmarks/test_ultrasam_gt_bbox_oku.py
python benchmarks/test_ultrasam_gt_bbox_roblus.py
python benchmarks/test_ultrasam_gt_bbox_tnsc2020.py
python benchmarks/test_ultrasam_gt_bbox_ultrabones100k.py
python benchmarks/test_ultrasam_gt_bbox_uns.py
```

## Run multiple datasets at once

`run_ultrasam.py` runs the per-dataset benchmarks sequentially. Toggle which
datasets to run by editing the `DATASETS` dict at the top of the file
(`True`/`False` per entry; `ultrabones100k` is off by default), then run:

```bash
python work_dir/UltraSam/run_ultrasam.py
```

Any extra CLI args are forwarded to every benchmark, e.g.:

```bash
# Quick smoke test on 50 samples per dataset
python work_dir/UltraSam/run_ultrasam.py --max-samples 50

# Run on CPU
python work_dir/UltraSam/run_ultrasam.py --device cpu
```

The script prints a per-dataset `[RUN]`/`[DONE]`/`[FAIL]` log and exits with a
non-zero status if any dataset fails.

## Performance note (16 GB GPUs, e.g. RTX 5080)

The benchmark defaults to `--batch-size 4`, which keeps the 1024×1024 SAM-encoder
activations in ~10 GB of VRAM. Larger batches (e.g. 8) saturate a 16 GB card, spill
into system RAM over PCIe, and run several times slower; throughput does not improve
past 4 even on 24 GB cards. Override per run with `--batch-size N` if needed.
