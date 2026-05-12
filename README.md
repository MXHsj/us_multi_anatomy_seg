## Ultrasound Multi-Anatomy Segmentator

This repo is organized for cross-dataset benchmarking of pretrained and future foundation segmentation models, starting with MedSAM.

## Hugging Face Dataset

Curated ultrasound datasets are hosted on Hugging Face:

- Dataset repo: https://huggingface.co/datasets/us-segmentator/us-segmentation-dataset
- Zip layout: `zips/<Anatomy>/<dataset>.zip`
- TNSC2020 lives under the `Thyroid` anatomy category.
- BCU_PD lives under the `Breast` anatomy category.

Raw datasets should not be committed to this repository. Dataset decoders can materialize supported datasets into ignored local cache folders on demand. For TNSC2020:

```bash
python datasets/hf_materialize.py --output-dir datasets/TNSC2020
python datasets/thyroid_TNSC2020.py --root datasets/TNSC2020 --max-samples 10
```

If more than one Thyroid zip exists, pass the exact Hugging Face path:

```bash
python datasets/hf_materialize.py \
  --repo-path zips/Thyroid/TNSC2020.zip \
  --output-dir datasets/TNSC2020
```

## Project Structure

```text
us_multi_anatomy_seg/
|-- datasets/
|   |-- common.py                  # Unified sample schema + shared preprocessing helpers
|   |-- thyroid_TNSC2020.py        # TNSC2020 decoder with HF-backed materialization
|   |-- heart_CAMUS.py             # CAMUS raw-data decoder
|   |-- kidney_OKU.py              # OKU raw-data decoder
|   |-- hf_materialize.py          # Hugging Face zip download/extraction helpers
|   `-- <dataset cache dirs>/      # ignored local materializations
|-- benchmarks/
|   `-- test_medsam_inference.py   # GT-box prompted MedSAM benchmark
|-- results/                       # Metrics CSV/JSON + visualizations
|-- notebooks/                     # Result exploration notebooks
`-- work_dir/MedSAM/               # MedSAM checkpoint
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

### BCU_PD

Decode breast BCU_PD data after it is materialized from Hugging Face:

```bash
python datasets/breast_BCU_PD.py --root datasets/BCU_PD --max-samples 10
```

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

If `datasets/TNSC2020` is missing, the TNSC2020 decoder downloads the Thyroid zip from Hugging Face automatically before benchmarking. Use `--no-auto-download` to require an existing local cache.

If `work_dir/MedSAM/medsam_vit_b.pth` is missing, the benchmark downloads the checkpoint from `GleghornLab/medsam-vit-b` on Hugging Face. Use `--no-auto-download-checkpoint` to require a local checkpoint or pass a custom path with `--checkpoint`.

For the BCU_PD GT-box benchmark:

```bash
python benchmarks/test_medsam_gt_bcu_pd.py
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

### Kidney

- [Open Kidney Ultrasound Dataset](https://github.com/rsingla92/kidneyUS/tree/main)

## Breast

- [Breast Lesion USG](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/)
