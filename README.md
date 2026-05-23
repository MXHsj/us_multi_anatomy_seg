## Ultrasound Multi-Anatomy Segmentator

This repo is organized for cross-dataset benchmarking of pretrained and future foundation segmentation models, starting with MedSAM.

## Installation

Create and activate the project conda environment:

```bash
conda create -n monai-usseg python=3.10
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

Raw datasets, checkpoints, and generated benchmark outputs should not be committed or pushed to GitHub. Keep local materializations under ignored folders such as `datasets/TNSC2020/`, `datasets/AULID/`, `datasets/UNS/`, `datasets/RobLUS/`, and `results/`.

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
| Breast | BUS-BRA | `zips/Breast/BUSBRA.zip` | `busbra` |
| Breast | BUSI | `zips/Breast/BUSI.zip` | `busi` |
| Heart | CAMUS | `zips/Heart/CAMUS.zip` | `camus` |
| Kidney | OKU | `zips/kidney/OKU.zip` | `oku` |
| Liver | AULID | `zips/Liver/AULID.zip` | `aulid` |
| Lung | RobLUS | `zips/Lung/RobLUS.zip` | `roblus` |
| Nerve | UNS | `zips/Nerve/UNS.zip` | `uns` |
| Thyroid | TNSC2020 | `zips/Thyroid/TNSC2020.zip` | `tnsc2020` |

### Dataset Structures

Each decoder normalizes its source dataset into the shared `DecodedSample` schema from `datasets/common.py`: `dataset`, `sample_id`, `image`, `mask`, and `metadata`. Images are normalized to `uint8`; masks are binary for the current GT-box benchmarks. Decoders preserve the original dataset annotation semantics by default, except UltraBones100k, where thin bone-surface annotations are intentionally hole-filled into a bone-shadow region.

| Key | Local root | Source layout interpreted by decoder | Benchmark sample definition |
| --- | --- | --- | --- |
| `tnsc2020` | `datasets/TNSC2020` | `image/*.PNG` paired with same-named `mask/*.PNG`; optional `train.csv` category metadata keyed by image ID. | One thyroid image/mask pair per PNG. |
| `blusg` | `datasets/BLUSG` | Flat `case*.png` images; masks are sibling `case*_tumor.png` plus optional `case*_other*.png`. | One breast image with tumor mask, optionally merged with other lesion masks unless `--blusg-only-tumor` is used. |
| `busbra` | `datasets/BUSBRA` | `Images/bus_*.png` paired with `Masks/mask_*.png` by shared suffix; `bus_data.csv` provides per-sample BI-RADS, pathology, side, device, histology, and CSV-recorded bounding box. | One breast image/tumor-mask pair per `bus_<id>` stem (1,875 total). Optional `--busbra-pathology` and `--busbra-birads` filters narrow the iteration; CSV metadata is surfaced through `DecodedSample.metadata`. |
| `busi` | `datasets/BUSI` | Category subfolders `benign/`, `malignant/`, `normal/`, each containing `<cat> (N).png` images paired with `<cat> (N)_mask.png`. A handful of cases (mostly benign) also have `<cat> (N)_mask_<k>.png` extra masks that are OR-merged into a single binary mask. | One image per case with merged tumor mask. Default `--busi-categories benign,malignant` excludes `normal` (empty masks unusable for GT-box). `normal` can be added for sensitivity/empty-mask checks. |
| `oku` | `datasets/OKU` | Flat kidney PNG images plus `reviewed_labels_1.csv` / `reviewed_labels_2.csv` polygon annotations. | One image with polygons rasterized for selected anatomy; default benchmark anatomy is `Capsule`. |
| `ultrabones100k` | `datasets/UltraBones100k` | Nested `specimen*/<anatomy>/record*/UltrasoundImages/*.png` paired with `<label-folder>/<timestamp>_label.png`, plus optional `tracking.csv`. | One ultrasound frame and filled bone-shadow mask per paired timestamp. Source labels often trace only the visible bone surface; filling is kept explicitly because line-based segmentation is difficult for region-prompted foundation models, and the filled region has clinical meaning as the acoustic bone shadow. Use decoder option `--no-fill-mask` only for thin-label sensitivity checks. |
| `camus` | `datasets/CAMUS` | `patient*/patient*_2CH|4CH_ED|ES|half_sequence.nii.gz` paired with `_gt.nii.gz`; per-view `Info_*.cfg` files may also be present. | By default, ED/ES only: 500 patients x 2 views x 2 phases = 2000 samples. `--include-half-sequence` also expands cine volumes into frame-level samples. |
| `aulid` | `datasets/AULID` | Category folders `Benign/`, `Malignant/`, `Normal/`, each with `image/*.jpg` and `segmentation/<label>/*.json` polygon masks. | One liver image with selected JSON polygon label; default label is `mass`. |
| `uns` | `datasets/UNS` | `train/*.tif` images paired with same-stem `*_mask.tif`. | One nerve image/mask pair per training TIFF. |
| `roblus` | `datasets/RobLUS` | Cleaned category folders `AP/`, `BM/`, `CP/`, `SG/`, and `XM/`; ultrasound frames are `US_*.jpg`, with paired `mask/pleural_line/mask_*.png` and `mask/rib_shadow/mask_*.png`. Unused `rib` and `cartilage` label folders are removed. | Class-specific lung benchmark. The current default target is pleural-line segmentation only; rib-shadow masks remain available but are not part of the default benchmark. Cleaned local data has 615 annotated frames plus 149 empty-mask negative controls. |

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
|   |-- lung_RobLUS.py             # RobLUS cleaned lung decoder
|   `-- <dataset cache dirs>/      # ignored local materializations
|-- benchmarks/
|   |-- medsam_inference.py        # GT-box prompted MedSAM benchmark engine
|   `-- samus_inference.py         # GT-box prompted SAMUS benchmark engine
|-- analysis/
|   |-- model_across_datasets.py   # Same-model cross-dataset summaries + figures
|   `-- compare_models_same_dataset.py
|-- results/                       # Ignored generated metrics/visualizations
|-- notebooks/                     # Result exploration notebooks
`-- work_dir/                     # Local model code/checkpoints, including MedSAM/SAMUS
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

- `--dataset`: any key from `python -m datasets.registry`, such as `tnsc2020`, `blusg`, `busbra`, `busi`, `oku`, `aulid`, `roblus`, `uns`, or `ultrabones100k`.
- `--no-auto-download`: require an existing local dataset cache.
- `--hf-revision`: pin a Hugging Face dataset revision for reproducibility.
- `--oku-anatomy Capsule`: choose OKU annotation anatomy.
- `--aulid-label mass`: choose AULID mask label from `mass`, `liver`, or `outline`.
- `--busbra-pathology benign,malignant`: optionally restrict BUS-BRA samples by pathology.
- `--busbra-birads 4,5`: optionally restrict BUS-BRA samples by BI-RADS category.
- `--busi-categories benign,malignant`: select BUSI categories to decode; default excludes `normal` (empty masks).
- `--roblus-labels pleural_line`: choose RobLUS labels to merge into the binary lung mask. The default is `pleural_line`; avoid merged class runs for benchmark reporting.
- `--roblus-subjects AP,BM`: optionally restrict RobLUS decoding to selected subjects.
- `--box-padding 10`: used by the UltraBones100k wrappers to give the GT box prompt context around the filled bone-shadow mask.

Outputs:

- `per_sample_metrics.csv`
- `summary.json`
- `visualizations/*.png`; `--save-vis N` saves `N` evenly spaced cases across the evaluated run, not just the first `N`.

### Wrapper Benchmark Parameters

The wrapper scripts under `benchmarks/test_*_gt_bbox_*.py` are the current reproducible GT-box benchmark entry points. Use extra CLI arguments after a wrapper command to override any default below.

| Wrapper | Model | Dataset | Max samples | Box padding | Device default | Saved visualizations | Output dir |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| `test_medsam_gt_bbox_tnsc2020.py` | MedSAM | `tnsc2020` | 100 | 0 | auto CUDA/MPS/CPU | 20 | `results/medsam_gt_bbox_tnsc2020` |
| `test_medsam_gt_bbox_blusg.py` | MedSAM | `blusg` | 100 | 0 | `mps` | 20 | `results/medsam_gt_bbox_blusg` |
| `test_medsam_gt_bbox_busbra.py` | MedSAM | `busbra` | 100 | 0 | `mps` | 20 | `results/medsam_gt_bbox_busbra` |
| `test_medsam_gt_bbox_busi.py` | MedSAM | `busi` (`benign,malignant`) | 100 | 0 | `mps` | 20 | `results/medsam_gt_bbox_busi` |
| `test_medsam_gt_bbox_oku.py` | MedSAM | `oku` | 100 | 0 | `mps` | 20 | `results/medsam_gt_bbox_oku` |
| `test_medsam_gt_bbox_ultrabones100k.py` | MedSAM | `ultrabones100k` | 100 | 10 | `cuda:0` | 20 | `results/medsam_gt_bbox_ultrabones100k` |
| `test_medsam_gt_bbox_camus.py` | MedSAM | `camus` | 2000 | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/medsam_gt_bbox_camus` |
| `test_medsam_gt_bbox_aulid.py` | MedSAM | `aulid` | 100 | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/medsam_gt_bbox_aulid` |
| `test_medsam_gt_bbox_uns.py` | MedSAM | `uns` | 100 | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/medsam_gt_bbox_uns` |
| `test_medsam_gt_bbox_roblus.py` | MedSAM | `roblus` (`pleural_line`) | 764 | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/medsam_gt_bbox_roblus` |
| `test_samus_gt_bbox_tnsc2020.py` | SAMUS | `tnsc2020` | 100 | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/samus_gt_bbox_tnsc2020` |
| `test_samus_gt_bbox_blusg.py` | SAMUS | `blusg` | 100 | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/samus_gt_bbox_blusg` |
| `test_samus_gt_bbox_busbra.py` | SAMUS | `busbra` | 100 | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/samus_gt_bbox_busbra` |
| `test_samus_gt_bbox_busi.py` | SAMUS | `busi` (`benign,malignant`) | 100 | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/samus_gt_bbox_busi` |
| `test_samus_gt_bbox_oku.py` | SAMUS | `oku` | 100 | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/samus_gt_bbox_oku` |
| `test_samus_gt_bbox_ultrabones100k.py` | SAMUS | `ultrabones100k` | 100 | 10 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/samus_gt_bbox_ultrabones100k` |
| `test_samus_gt_bbox_camus.py` | SAMUS | `camus` | 2000 | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/samus_gt_bbox_camus` |
| `test_samus_gt_bbox_aulid.py` | SAMUS | `aulid` | 100 | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/samus_gt_bbox_aulid` |
| `test_samus_gt_bbox_uns.py` | SAMUS | `uns` | 100 | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/samus_gt_bbox_uns` |
| `test_samus_gt_bbox_roblus.py` | SAMUS | `roblus` (`pleural_line`) | 764 | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/samus_gt_bbox_roblus` |

Shared wrapper defaults: GT masks are converted to bounding-box prompts; MedSAM GT wrappers disable bbox jitter with `--bbox-jitter-prob 0.0`; MedSAM and SAMUS GT wrappers now use matched `--max-samples` caps per dataset for comparable default runs (`100` for AULID, BLUSG, BUS-BRA, BUSI, OKU, TNSC2020, UltraBones100k, and UNS; `764` for RobLUS; `2000` for CAMUS ED/ES). UltraBones100k wrappers use `--box-padding 10`; other datasets use `0`. Dataset-specific options include `--oku-anatomy Capsule`, `--aulid-label mass`, `--roblus-labels pleural_line`, `--camus-labels 1,2,3`, `--include-half-sequence`, `--busbra-pathology`, `--busbra-birads`, and `--busi-categories`. RobLUS is benchmarked class-specifically; merged pleural-line/rib-shadow runs are avoided because a single combined bbox is not a valid prompt protocol for separate anatomies/instances. RobLUS negative-control samples have empty masks and are skipped by the current GT-box engines because no bounding box can be generated.

When rerunning into an existing output directory, old visualization PNGs are not automatically deleted. Remove or move the existing `visualizations/` folder first if you need the qualitative sample set to reflect only the latest run.

## Quantitative Analysis

The `analysis/` folder contains lightweight scripts for result aggregation and figure generation from `results/`.

Same-model performance across datasets:

```bash
python analysis/model_across_datasets.py --model medsam
python analysis/model_across_datasets.py --model samus
```

These commands print Dice/IoU mean and standard deviation tables and save paper-style plots with mean bars, standard-deviation error bars, and per-sample scatter points under `analysis/figures/`.

Matched model comparison on the same datasets:

```bash
python analysis/compare_models_same_dataset.py
```

This compares MedSAM and SAMUS using shared `sample_id`s from each pair of `per_sample_metrics.csv` files, which is preferred when one result folder was generated with a larger sample cap than the other. Both analysis scripts support `--output-csv`, `--plot-path`, `--plot-title`, and `--no-plot`.

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
