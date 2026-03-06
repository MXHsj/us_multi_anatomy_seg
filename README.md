## Ultrasound Multi-Anatomy Segmentator

This repo is organized for cross-dataset benchmarking of pretrained and future foundation segmentation models (starting with MedSAM).

## Recommended Project Structure

```text
us_multi_anatomy_seg/
├── datasets/
│   ├── common.py                  # Unified sample schema + shared preprocessing helpers
│   ├── thyroid_TNSC2020.py        # TNSC2020 raw-data decoder
│   ├── heart_CAMUS.py             # CAMUS raw-data decoder
│   ├── TNSC2020/                  # Raw TNSC2020 data
│   └── CAMUS/                     # Raw CAMUS data (optional, add when available)
├── preprocessing/                 # Dataset-level preprocessing scripts (future)
├── postprocessing/                # Postprocessing and analysis utils (future)
├── benchmarks/
│   └── test_medsam_inference.py   # GT-box prompted MedSAM benchmark
├── results/                       # Metrics CSV/JSON + visualizations
├── notebooks/                     # Result exploration notebooks
├── work_dir/MedSAM/               # MedSAM checkpoint
└── MedSAM_Inference.py            # Existing single-image inference script
```

## Decoding Scripts

### TNSC2020

Decode and optionally export a unified manifest:

```bash
python datasets/thyroid_TNSC2020.py --root datasets/TNSC2020 --max-samples 10
python datasets/thyroid_TNSC2020.py --root datasets/TNSC2020 --export-dir results/tnsc_decoded
```

### CAMUS

When CAMUS data is added under `datasets/CAMUS`:

```bash
python datasets/heart_CAMUS.py --root datasets/CAMUS --max-samples 10
python datasets/heart_CAMUS.py --root datasets/CAMUS --export-dir results/camus_decoded
```

Note: CAMUS decoding requires `nibabel`.

## MedSAM Test Benchmark

Use ground-truth masks to build bounding-box prompts, then run MedSAM and report Dice/IoU/latency.

```bash
python benchmarks/test_medsam_inference.py \
  --dataset tnsc2020 \
  --dataset-root datasets/TNSC2020 \
  --checkpoint work_dir/MedSAM/medsam_vit_b.pth \
  --device cuda:0 \
  --max-samples 50 \
  --output-dir results/medsam_test_tnsc
```

Outputs:
- `per_sample_metrics.csv`
- `summary.json`
- `visualizations/*.png`

## Baseline Foundation Models
- [UltraSAM](https://arxiv.org/html/2411.16222v1)

## Dataset References

### Heart
- [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/)

### Liver
- [Liver Ultrasound Semantic Segmentation Dataset](https://universe.roboflow.com/romain-hardy-6ehro/liver-ultrasound-semantic-segmentation)

### Bone
- [UltraBones100k](https://github.com/luohwu/UltraBones100k)

### Thyroid
- [TNSC2020](https://tn-scui2020.grand-challenge.org/Dataset/)
