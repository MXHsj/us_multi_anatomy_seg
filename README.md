## Ultrasound Multi-Anatomy Segmentator

This repo is organized for cross-dataset benchmarking of pretrained and future foundation segmentation models, including MedSAM, SAMUS, and UltraSAM.

## Installation

Create and activate the project conda environment:

```bash
conda create -n monai-usseg python=3.12
conda activate monai-usseg
```

Install Python dependencies from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Development Container

This repo includes a VS Code devcontainer under `.devcontainer/`. Open the
repository in VS Code and choose **Dev Containers: Reopen in Container** to build
the CUDA-enabled Python 3.10 image, install the top-level `requirements.txt`,
install the OpenMMLab stack used by UltraSAM, and clone
`CAMMA-public/UltraSam` into `/opt/UltraSam`.

The container requests all available NVIDIA GPUs. On a CPU-only Docker host,
remove `--gpus=all` from `.devcontainer/devcontainer.json` before rebuilding.

The devcontainer also installs the OpenAI Codex VS Code extension
(`openai.chatgpt`). Your local Codex login/config directory is bind-mounted from
`%USERPROFILE%/.codex` to `/home/vscode/.codex`, so the extension can reuse local
Codex state without storing secrets in the Docker image.

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
| Muscle | UMUD Aponeurosis | `zips/Muscle/UMUD.zip` | `umud` |
| Nerve | UNS | `zips/Nerve/UNS.zip` | `uns` |
| Ovary | MMOTU | `zips/Ovary/MMOTU.zip` | `mmotu` |
| Spinal Cord | Ultrasound Spinal Cord | `zips/SpinalCord/USSC.zip` | `ussc` |
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
| `ussc` | `datasets/USSC` | Upstream `SegmentationDataset.zip`, either with a nested `SegmentationDataset/` folder or direct split folders. `train_images/`, `val_images/`, and `test_images/` PNGs pair with same-name RGB semantic masks in `train_masks/`, `val_masks/`, and `test_masks/`. | CAMUS-style multi-class benchmark: one binary target per selected semantic class per source image. Default `--ussc-labels` uses all non-background labels: dura, CSF, pia, spinal cord, dorsal space, hematoma, dura/pia complex, dura/ventral complex, and ventral space. Source: https://github.com/avishakumar21/ultrasound-spinal-cord-dataset |
| `umud` | `datasets/UMUD` | Raw Kaggle UMUD challenge zip. `apo_imgs_v1/apo_images_new_model_v1/*.tif` pairs with same-name `apo_masks_v1/apo_masks_new_model_v1/*.tif`. | One aponeurosis image/mask pair per shared TIFF stem. Fascicle folders and unlabeled `test_images_v2/` are intentionally ignored. Masks are binarized from `>0` and resized with nearest-neighbor to the paired image shape when needed. |
| `mmotu` | `datasets/MMOTU` | Raw MMOTU zip, either with nested `MMOTU/OTU_2d/` or direct `OTU_2d/`. `images/*.JPG` pairs with `annotations/<stem>_binary.PNG`; `OTU_3d/`, raw annotation PNGs, and duplicated `_binary_binary.PNG` masks are ignored. | One ovarian-tumor image/mask pair per 2D case. The decoder uses `train.txt` and `val.txt` for split metadata and treats `train_cls.txt` / `val_cls.txt` as optional classification metadata, not segmentation classes. Source: https://github.com/cv516Buaa/MMOTU_DS2Net |

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
|   |-- spinal_cord_USSC.py        # USSC semantic spinal cord decoder
|   |-- muscle_UMUD.py             # UMUD aponeurosis decoder
|   |-- ovary_MMOTU.py             # MMOTU 2D ovarian tumor decoder
|   `-- <dataset cache dirs>/      # ignored local materializations
|-- benchmarks/
|   |-- medsam_inference.py        # GT-box prompted MedSAM benchmark engine
|   |-- samus_inference.py         # GT-box prompted SAMUS benchmark engine
|   `-- ultrasam_inference.py      # GT-box prompted UltraSAM/OpenMMLab engine
|-- analysis/
|   |-- model_across_datasets.py   # Same-model cross-dataset summaries + figures
|   `-- compare_models_same_dataset.py
|-- results/                       # Ignored generated metrics/visualizations
|-- notebooks/                     # Result exploration notebooks
`-- work_dir/                     # Local checkpoints, including MedSAM/SAMUS/UltraSAM
```

## Benchmark Usage

Use ground-truth masks to build bounding-box prompts, then run each model and report overlap, pixel-classification, boundary, size-error, and latency metrics.

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

### UltraSAM

UltraSAM uses the upstream OpenMMLab project from `CAMMA-public/UltraSam`. In
the devcontainer, that source tree is available at `/opt/UltraSam` through the
`ULTRASAM_DIR` environment variable.

```bash
python benchmarks/ultrasam_inference.py \
  --dataset tnsc2020 \
  --device cuda:0 \
  --max-samples 50 \
  --output-dir results/ultrasam_test_tnsc
```

If `work_dir/UltraSam/UltraSam.pth` is missing, the benchmark downloads the checkpoint from `https://s3.unistra.fr/camma_public/github/ultrasam/UltraSam.pth`. Use `--no-auto-download-checkpoint` to require a local checkpoint. Outside the devcontainer, pass `--ultrasam-dir path/to/UltraSam`, set `ULTRASAM_DIR`, or add `--auto-clone-source`.

### Medical SAM3 (text-prompted)

Medical SAM3 is a **text-prompted** model: it segments from a clinical concept
string (e.g. `"thyroid nodule"`) instead of a box. Its results are reported in a
**separate table** and never cross-compared with the box-prompted models (a GT-derived
box is a privileged localization cue). The concept for each dataset/class comes from
`datasets/label_text.py`.

It needs the `sam3` package, which is **not** in `requirements.txt` (it is a heavy,
separate environment, so install it on its own — ideally a dedicated conda/uv env).
`sam3` is the Medical-SAM3 repo's own package and is **not on PyPI**, so install it from
source. One-off command (pip clones and builds it for you):

```bash
pip install "git+https://github.com/AIM-Research-Lab/Medical-SAM3.git#egg=sam3[train]"
```

(Equivalently: `git clone` the repo, then `pip install -r requirements.txt && pip install -e ".[train]"` from inside it.) Once `import sam3` works, run:

```bash
python benchmarks/medical_sam3_inference.py \
  --dataset tnsc2020 \
  --device cuda:0 \
  --max-samples 50 \
  --output-dir results/medicalsam3_text_prompt_tnsc2020
```

The fine-tuned `checkpoint_2D.pt` (~10 GB) auto-downloads from the
[`Chongcong/Medical-SAM3`](https://huggingface.co/Chongcong/Medical-SAM3) HF repo to
`work_dir/MedicalSAM3/`; use `--no-auto-download-checkpoint` to require a local file, or
`--checkpoint path/to/checkpoint_2D.pt`. Multi-class datasets (e.g. CAMUS) are queried
per class and scored independently; empty-GT frames are skipped, matching the box engines.

### Dataset-Specific Options

- `--dataset`: any key from `python -m datasets.registry`, such as `tnsc2020`, `blusg`, `busbra`, `busi`, `oku`, `aulid`, `roblus`, `uns`, `ussc`, `umud`, `mmotu`, or `ultrabones100k`.
- `--no-auto-download`: require an existing local dataset cache.
- `--hf-revision`: pin a Hugging Face dataset revision for reproducibility.
- `--oku-anatomy Capsule`: choose OKU annotation anatomy.
- `--aulid-label mass`: choose AULID mask label from `mass`, `liver`, or `outline`.
- `--busbra-pathology benign,malignant`: optionally restrict BUS-BRA samples by pathology.
- `--busbra-birads 4,5`: optionally restrict BUS-BRA samples by BI-RADS category.
- `--busi-categories benign,malignant`: select BUSI categories to decode; default excludes `normal` (empty masks).
- `--roblus-labels pleural_line`: choose RobLUS labels to merge into the binary lung mask. The default is `pleural_line`; avoid merged class runs for benchmark reporting.
- `--roblus-subjects AP,BM`: optionally restrict RobLUS decoding to selected subjects.
- `--ussc-labels spinal_cord,hematoma`: choose USSC semantic classes to decode as separate binary targets. The default is all non-background classes.
- `--box-padding 10`: used by the UltraBones100k wrappers to give the GT box prompt context around the filled bone-shadow mask.

Outputs:

- `per_sample_metrics.csv`
- `summary.json`
- `visualizations/*.png`; `--save-vis N` saves `N` evenly spaced cases across the evaluated run, not just the first `N`.

### Evaluation Metrics

The benchmark writes per-sample values and summary mean/std values for the metrics below. Boundary distances are currently measured in image pixels because most 2D ultrasound decoders do not expose calibrated physical spacing.

| Evaluation aspect | Metric columns | Why included |
| --- | --- | --- |
| Region overlap | `dice`, `iou` | Main segmentation quality scores; capture how much the predicted mask overlaps the ground-truth target. |
| Pixel-level detection balance | `precision`, `recall`, `specificity`, `balanced_accuracy` | Separate over-segmentation, under-segmentation, and background rejection behavior that can be hidden by Dice/IoU alone. |
| Boundary accuracy | `hd95`, `assd` | Quantify contour error: `hd95` captures near-worst boundary misses while reducing single-pixel outlier sensitivity; `assd` captures typical surface-to-surface error. |
| Size and shape bias | `relative_area_error` | Shows whether predictions systematically overestimate or underestimate target area, even when overlap scores look similar. |

### Wrapper Benchmark Parameters

The wrapper scripts under `benchmarks/test_*_gt_bbox_*.py` are the current reproducible GT-box benchmark entry points. Use extra CLI arguments after a wrapper command to override any default below.

| Wrapper | Model | Dataset | Max samples | Box padding | Device default | Saved visualizations | Output dir |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| `test_medsam_gt_bbox_tnsc2020.py` | MedSAM | `tnsc2020` | all | 0 | auto CUDA/MPS/CPU | 20 | `results/medsam_gt_bbox_tnsc2020` |
| `test_medsam_gt_bbox_blusg.py` | MedSAM | `blusg` | all | 0 | `mps` | 20 | `results/medsam_gt_bbox_blusg` |
| `test_medsam_gt_bbox_busbra.py` | MedSAM | `busbra` | all | 0 | `mps` | 20 | `results/medsam_gt_bbox_busbra` |
| `test_medsam_gt_bbox_busi.py` | MedSAM | `busi` (`benign,malignant`) | all | 0 | `mps` | 20 | `results/medsam_gt_bbox_busi` |
| `test_medsam_gt_bbox_oku.py` | MedSAM | `oku` | all | 0 | `mps` | 20 | `results/medsam_gt_bbox_oku` |
| `test_medsam_gt_bbox_ultrabones100k.py` | MedSAM | `ultrabones100k` | all | 10 | `cuda:0` | 20 | `results/medsam_gt_bbox_ultrabones100k` |
| `test_medsam_gt_bbox_camus.py` | MedSAM | `camus` | all | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/medsam_gt_bbox_camus` |
| `test_medsam_gt_bbox_aulid.py` | MedSAM | `aulid` | all | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/medsam_gt_bbox_aulid` |
| `test_medsam_gt_bbox_uns.py` | MedSAM | `uns` | all | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/medsam_gt_bbox_uns` |
| `test_medsam_gt_bbox_roblus.py` | MedSAM | `roblus` (`pleural_line`) | all | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/medsam_gt_bbox_roblus` |
| `test_medsam_gt_bbox_ussc.py` | MedSAM | `ussc` (all non-background classes) | all | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/medsam_gt_bbox_ussc` |
| `test_medsam_gt_bbox_umud.py` | MedSAM | `umud` (`aponeurosis`) | all | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/medsam_gt_bbox_umud` |
| `test_medsam_gt_bbox_mmotu.py` | MedSAM | `mmotu` (`ovarian_tumor`) | all | 0 | `mps` on macOS, otherwise `cuda:0` | 20 | `results/medsam_gt_bbox_mmotu` |
| `test_samus_gt_bbox_tnsc2020.py` | SAMUS | `tnsc2020` | all | 0 | `mps` on macOS, otherwise `cuda:0` | all | `results/samus_gt_bbox_tnsc2020` |
| `test_samus_gt_bbox_blusg.py` | SAMUS | `blusg` | all | 0 | `mps` on macOS, otherwise `cuda:0` | all | `results/samus_gt_bbox_blusg` |
| `test_samus_gt_bbox_busbra.py` | SAMUS | `busbra` | all | 0 | `mps` on macOS, otherwise `cuda:0` | all | `results/samus_gt_bbox_busbra` |
| `test_samus_gt_bbox_busi.py` | SAMUS | `busi` (`benign,malignant`) | all | 0 | `mps` on macOS, otherwise `cuda:0` | all | `results/samus_gt_bbox_busi` |
| `test_samus_gt_bbox_oku.py` | SAMUS | `oku` | all | 0 | `mps` on macOS, otherwise `cuda:0` | all | `results/samus_gt_bbox_oku` |
| `test_samus_gt_bbox_ultrabones100k.py` | SAMUS | `ultrabones100k` | all | 10 | `mps` on macOS, otherwise `cuda:0` | all | `results/samus_gt_bbox_ultrabones100k` |
| `test_samus_gt_bbox_camus.py` | SAMUS | `camus` | all | 0 | `mps` on macOS, otherwise `cuda:0` | all | `results/samus_gt_bbox_camus` |
| `test_samus_gt_bbox_aulid.py` | SAMUS | `aulid` | all | 0 | `mps` on macOS, otherwise `cuda:0` | all | `results/samus_gt_bbox_aulid` |
| `test_samus_gt_bbox_uns.py` | SAMUS | `uns` | all | 0 | `mps` on macOS, otherwise `cuda:0` | all | `results/samus_gt_bbox_uns` |
| `test_samus_gt_bbox_roblus.py` | SAMUS | `roblus` (`pleural_line`) | all | 0 | `mps` on macOS, otherwise `cuda:0` | all | `results/samus_gt_bbox_roblus` |
| `test_samus_gt_bbox_ussc.py` | SAMUS | `ussc` (all non-background classes) | all | 0 | `mps` on macOS, otherwise `cuda:0` | all | `results/samus_gt_bbox_ussc` |
| `test_samus_gt_bbox_umud.py` | SAMUS | `umud` (`aponeurosis`) | all | 0 | `mps` on macOS, otherwise `cuda:0` | all | `results/samus_gt_bbox_umud` |
| `test_samus_gt_bbox_mmotu.py` | SAMUS | `mmotu` (`ovarian_tumor`) | all | 0 | `mps` on macOS, otherwise `cuda:0` | all | `results/samus_gt_bbox_mmotu` |
| `test_ultrasam_gt_bbox_tnsc2020.py` | UltraSAM | `tnsc2020` | all | 0 | CUDA if available, otherwise CPU | all | `results/ultrasam_gt_bbox_tnsc2020` |
| `test_ultrasam_gt_bbox_blusg.py` | UltraSAM | `blusg` | all | 0 | CUDA if available, otherwise CPU | all | `results/ultrasam_gt_bbox_blusg` |
| `test_ultrasam_gt_bbox_busbra.py` | UltraSAM | `busbra` | all | 0 | CUDA if available, otherwise CPU | all | `results/ultrasam_gt_bbox_busbra` |
| `test_ultrasam_gt_bbox_busi.py` | UltraSAM | `busi` (`benign,malignant`) | all | 0 | CUDA if available, otherwise CPU | all | `results/ultrasam_gt_bbox_busi` |
| `test_ultrasam_gt_bbox_oku.py` | UltraSAM | `oku` | all | 0 | CUDA if available, otherwise CPU | all | `results/ultrasam_gt_bbox_oku` |
| `test_ultrasam_gt_bbox_ultrabones100k.py` | UltraSAM | `ultrabones100k` | all | 10 | CUDA if available, otherwise CPU | all | `results/ultrasam_gt_bbox_ultrabones100k` |
| `test_ultrasam_gt_bbox_camus.py` | UltraSAM | `camus` | all | 0 | CUDA if available, otherwise CPU | all | `results/ultrasam_gt_bbox_camus` |
| `test_ultrasam_gt_bbox_aulid.py` | UltraSAM | `aulid` | all | 0 | CUDA if available, otherwise CPU | all | `results/ultrasam_gt_bbox_aulid` |
| `test_ultrasam_gt_bbox_uns.py` | UltraSAM | `uns` | all | 0 | CUDA if available, otherwise CPU | all | `results/ultrasam_gt_bbox_uns` |
| `test_ultrasam_gt_bbox_roblus.py` | UltraSAM | `roblus` (`pleural_line`) | all | 0 | CUDA if available, otherwise CPU | all | `results/ultrasam_gt_bbox_roblus` |
| `test_ultrasam_gt_bbox_ussc.py` | UltraSAM | `ussc` (all non-background classes) | all | 0 | CUDA if available, otherwise CPU | all | `results/ultrasam_gt_bbox_ussc` |
| `test_ultrasam_gt_bbox_umud.py` | UltraSAM | `umud` (`aponeurosis`) | all | 0 | CUDA if available, otherwise CPU | all | `results/ultrasam_gt_bbox_umud` |
| `test_ultrasam_gt_bbox_mmotu.py` | UltraSAM | `mmotu` (`ovarian_tumor`) | all | 0 | CUDA if available, otherwise CPU | all | `results/ultrasam_gt_bbox_mmotu` |

Shared wrapper defaults: GT masks are converted to bounding-box prompts; MedSAM GT wrappers disable bbox jitter with `--bbox-jitter-prob 0.0`; MedSAM, SAMUS, and UltraSAM GT wrappers use `--max-samples all` for full decoded-dataset default runs. Pass a positive integer, such as `--max-samples 100`, to run a capped smoke test with the same single parameter. SAMUS and UltraSAM wrappers request all visualizations; MedSAM wrappers save 20. UltraBones100k wrappers use `--box-padding 10`; other datasets use `0`. Dataset-specific options include `--oku-anatomy Capsule`, `--aulid-label mass`, `--roblus-labels pleural_line`, `--ussc-labels spinal_cord,hematoma`, `--camus-labels 1,3`, `--include-half-sequence`, `--busbra-pathology`, `--busbra-birads`, and `--busi-categories`. RobLUS is benchmarked class-specifically; merged pleural-line/rib-shadow runs are avoided because a single combined bbox is not a valid prompt protocol for separate anatomies/instances. RobLUS negative-control samples have empty masks and are skipped by the current GT-box engines because no bounding box can be generated.

When rerunning into an existing output directory, old visualization PNGs are not automatically deleted. Remove or move the existing `visualizations/` folder first if you need the qualitative sample set to reflect only the latest run.

## Quantitative Analysis

The `analysis/` folder contains lightweight scripts for result aggregation and figure generation from `results/`.

Same-model performance across datasets:

```bash
python analysis/model_across_datasets.py --model medsam
python analysis/model_across_datasets.py --model samus
python analysis/model_across_datasets.py --model ultrasam
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
