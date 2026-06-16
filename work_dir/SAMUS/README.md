# SAMUS

Commands to run SAMUS (GT box prompts) on each dataset. Run from the repo root.
The trained checkpoint `work_dir/SAMUS/ckp/SAMUS.pth` is auto-downloaded from the
upstream Google Drive release on first run.

```bash
python benchmarks/test_samus_gt_bbox_aulid.py
python benchmarks/test_samus_gt_bbox_blusg.py
python benchmarks/test_samus_gt_bbox_busbra.py
python benchmarks/test_samus_gt_bbox_busi.py
python benchmarks/test_samus_gt_bbox_camus.py
python benchmarks/test_samus_gt_bbox_oku.py
python benchmarks/test_samus_gt_bbox_roblus.py
python benchmarks/test_samus_gt_bbox_tnsc2020.py
python benchmarks/test_samus_gt_bbox_ultrabones100k.py
python benchmarks/test_samus_gt_bbox_uns.py
```

## Layout

- `source_code/` — the vendored upstream SAMUS source (https://github.com/xianlin7/SAMUS),
  imported by `benchmarks/samus_inference.py`. See `source_code/README.md` for the
  original project documentation.
- `run_samus.py` — runs the per-dataset benchmarks sequentially (see below).
- `ckp/SAMUS.pth` — trained SAMUS weights, auto-downloaded on first run.

## Run multiple datasets at once

`run_samus.py` runs the per-dataset benchmarks sequentially. Toggle which
datasets to run by editing the `DATASETS` dict at the top of the file
(`True`/`False` per entry; `ultrabones100k` is off by default), then run:

```bash
python work_dir/SAMUS/run_samus.py
```

Any extra CLI args are forwarded to every benchmark, e.g.:

```bash
# Quick smoke test on 50 samples per dataset
python work_dir/SAMUS/run_samus.py --max-samples 50

# Run on CPU
python work_dir/SAMUS/run_samus.py --device cpu
```

The script prints a per-dataset `[RUN]`/`[DONE]`/`[FAIL]` log and exits with a
non-zero status if any dataset fails.

## Checkpoints

- The trained SAMUS checkpoint downloads automatically to `work_dir/SAMUS/ckp/SAMUS.pth`.
  Pass `--no-auto-download-checkpoint` to require an existing local file instead.
- The base SAM ViT-B checkpoint (`work_dir/SAMUS/checkpoints/sam_vit_b_01ec64.pth`) is
  optional: it is loaded before the trained weights only if present. The trained
  `SAMUS.pth` is a full state dict, so it is not required for inference.

## Performance note

SAMUS uses a 256×256 image encoder (vs. UltraSAM's 1024×1024), so it is light on
VRAM and fast. The benchmark defaults to `--batch-size 1`; raise it with
`--batch-size N` to improve throughput on larger GPUs.
