## Ultrasound Multi-Anatomy Segmentator

This repo is organized for cross-dataset benchmarking of pretrained and future foundation segmentation models, starting with MedSAM.

## Installation

Create and activate the project conda environment:

```bash
conda create -n monai-usseg python=3.9
conda activate monai-usseg
```

Install Python dependencies from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Hugging Face Dataset

Curated ultrasound datasets are hosted on Hugging Face:

- Dataset repo: https://huggingface.co/datasets/us-segmentator/us-segmentation-dataset
- Zip layout: `zips/<Anatomy>/<dataset>.zip`

Dataset access is registry-driven. `datasets/registry.py` records the dataset key, anatomy, default local cache path, decoder class, and Hugging Face zip path. Benchmark scripts call `datasets/loader.py`, which checks whether the local cache directory already exists and is non-empty. If it exists, nothing is downloaded. If it is missing and the dataset has a registered Hugging Face path, `datasets/hf_materialize.py` downloads and extracts the zip. Dataset-specific decoders remain responsible for interpreting each dataset's internal image/mask layout.

Raw datasets, checkpoints, and generated benchmark outputs should not be committed or pushed to GitHub. Keep local materializations under ignored folders such as `datasets/TNSC2020/`, `datasets/AULID/`, `datasets/UNS/`, and `results/`.

To list datasets currently registered in this codebase:

```bash
python -m datasets.registry
```

For example, materialize and smoke-test TNSC2020:

```bash
python datasets/hf_materialize.py tnsc2020 --output-dir datasets/TNSC2020
python datasets/thyroid_TNSC2020.py --root datasets/TNSC2020 --max-samples 10
```

Benchmark scripts can also materialize registered datasets automatically:

```bash
python benchmarks/medsam_inference.py \
  --dataset tnsc2020 \
  --max-samples 10 \
  --output-dir results/medsam_test_tnsc
```

Use `--no-auto-download` to require an existing local cache. To override the registered Hugging Face zip path:

```bash
python datasets/hf_materialize.py tnsc2020 \
  --repo-path zips/Thyroid/TNSC2020.zip \
  --output-dir datasets/TNSC2020
```

Available Hugging Face zips scanned from the dataset repo:

| Anatomy | Dataset | HF zip path | Registry key / decoder status |
| --- | --- | --- | --- |
| Bone | UltraBones100k | `zips/Bone/UltraBones100k.zip` | `ultrabones100k` |
| Breast | BrEaST Lesions USG | `zips/Breast/BrEaST-Lesions_USG.zip` | `blusg` |
| Fetus | fetal_planes | `zips/Fetus/fetal_planes.zip` | Not yet implemented |
| Heart | CAMUS | `zips/Heart/CAMUS.zip` | `camus` |
| Kidney | OKU | `zips/kidney/OKU.zip` | `oku` |
| Liver | AULID | `zips/Liver/AULID.zip` | `aulid` |
| Lung | RobLUS | `zips/Lung/RobLUS.zip` | Not yet implemented |
| Nerve | UNS | `zips/Nerve/UNS.zip` | `uns` |
| Thyroid | TNSC2020 | `zips/Thyroid/TNSC2020.zip` | `tnsc2020` |

## Project Structure

```text
us_multi_anatomy_seg/
|-- datasets/
|   |-- common.py                  # Unified sample schema + shared preprocessing helpers
|   |-- registry.py                # Dataset availability, local roots, HF zip paths
|   |-- loader.py                  # Registry-backed decoder construction
|   |-- hf_materialize.py          # Generic Hugging Face zip download/extraction helper
|   |-- thyroid_TNSC2020.py        # TNSC2020 decoder
|   |-- heart_CAMUS.py             # CAMUS raw-data decoder
|   |-- kidney_OKU.py              # OKU raw-data decoder
|   `-- <dataset cache dirs>/      # ignored local materializations
|-- benchmarks/
|   |-- medsam_inference.py        # GT-box prompted MedSAM benchmark engine
|   `-- samus_inference.py         # GT-box prompted SAMUS benchmark engine
|-- results/                       # Ignored generated metrics/visualizations
|-- notebooks/                     # Result exploration notebooks
`-- work_dir/MedSAM/               # MedSAM checkpoint
```

## Benchmark Usage

Use ground-truth masks to build bounding-box prompts, then run MedSAM and report Dice/IoU/latency.

### MedSAM

```bash
python benchmarks/medsam_inference.py \
  --dataset tnsc2020 \
  --checkpoint work_dir/MedSAM/medsam_vit_b.pth \
  --device cuda:0 \
  --max-samples 50 \
  --output-dir results/medsam_test_tnsc
```

If `work_dir/MedSAM/medsam_vit_b.pth` is missing, the benchmark downloads the checkpoint from `GleghornLab/medsam-vit-b` on Hugging Face. Use `--no-auto-download-checkpoint` to require a local checkpoint or pass a custom path with `--checkpoint`.

### SAMUS

```bash
python benchmarks/samus_inference.py \
  --dataset ultrabones100k \
  --device cuda:0 \
  --max-samples 50 \
  --box-padding 10 \
  --output-dir results/samus_test_ultrabones
```

If `work_dir/SAMUS/ckp/SAMUS.pth` is missing, the SAMUS benchmark downloads the checkpoint from the upstream Google Drive release. Use `--no-auto-download-checkpoint` to require a local checkpoint.

### Dataset-Specific Options

- `--dataset`: any key from `python -m datasets.registry`, such as `tnsc2020`, `blusg`, `oku`, `aulid`, `uns`, or `ultrabones100k`.
- `--no-auto-download`: require an existing local dataset cache.
- `--hf-revision`: pin a Hugging Face dataset revision for reproducibility.
- `--oku-anatomy Capsule`: choose OKU annotation anatomy.
- `--aulid-label mass`: choose AULID mask label from `mass`, `liver`, or `outline`.
- `--box-padding 10`: useful for thin/line-like masks such as UltraBones100k.

Outputs:

- `per_sample_metrics.csv`
- `summary.json`
- `visualizations/*.png`

## Adding A New Dataset

1. Upload or confirm the dataset zip in the Hugging Face repo under `zips/<Anatomy>/<dataset>.zip`.
2. Add a `DatasetSpec` entry in `datasets/registry.py` with a stable key, display name, anatomy, default local root, decoder class, and HF zip path.
3. Implement a decoder in `datasets/<anatomy>_<dataset>.py` that yields `DecodedSample` objects. Keep dataset-specific layout assumptions inside the decoder.
4. Add the decoder to `datasets/__init__.py`.
5. Add the local cache folder to `.gitignore`, for example `datasets/NewDataset/`.
6. Smoke-test materialization and decoding:

```bash
python datasets/hf_materialize.py <dataset-key>
python datasets/<decoder_script>.py --root datasets/<DatasetRoot> --max-samples 3
```

7. Smoke-test at least one benchmark:

```bash
python benchmarks/medsam_inference.py \
  --dataset <dataset-key> \
  --max-samples 3 \
  --save-vis 1 \
  --output-dir results/<dataset-key>_smoke
```

8. Update the README table if the HF repo contents or decoder support changed.

## Baseline Foundation Models

- [MedSAM]()
- [UltraSAM](https://arxiv.org/html/2411.16222v1)
