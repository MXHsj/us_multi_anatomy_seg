# Project Context: Ultrasound Multi-Anatomy Segmentation Benchmark

## Research Goal

This project benchmarks state-of-the-art foundation models for medical image segmentation on curated ultrasound datasets. The intended study is not only a leaderboard-style comparison, but a research-oriented analysis of when and why segmentation foundation models succeed or fail across anatomy, dataset source, object type, and prompt quality.

The current focus is ultrasound segmentation across multiple anatomies. MedSAM is the first implemented model baseline, and the project is moving toward a larger Hugging Face-hosted dataset workflow, multiple model environments, and more meaningful benchmark criteria.

## Current Repository Status

The repository currently contains:

- Unified dataset sample schema and preprocessing helpers in `datasets/common.py`.
- Dataset decoders for:
  - `TNSC2020`: thyroid ultrasound image/mask pairs.
  - `BLUSG`: breast lesion ultrasound images with tumor and optional other lesion masks.
  - `OKU`: kidney ultrasound images with polygon annotations from reviewed label CSVs.
  - `UltraBones100k`: bone ultrasound records with image/label pairing support.
  - `CAMUS`: heart ultrasound NIfTI decoder support, but local data is not present.
- A MedSAM inference benchmark in `benchmarks/medsam_inference.py`.
- Wrapper scripts for MedSAM ground-truth bounding-box prompts and jittered bounding-box prompts.
- Locally stored raw datasets under `datasets/`, currently including `BLUSG`, `OKU`, and `TNSC2020`.
- Existing MedSAM result folders under `results/`.

The benchmark currently uses ground-truth masks to derive bounding-box prompts, optionally applies bounding-box jitter as a prompt degradation experiment, then reports Dice, IoU, and inference latency.

## Existing MedSAM Benchmark Results

The following completed runs are present in `results/`:

| Dataset | Prompt setting | Samples | Dice mean | Dice std | IoU mean | IoU std |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TNSC2020 | GT bbox | 100 | 0.9444 | 0.0358 | 0.8966 | 0.0598 |
| TNSC2020 | Jittered bbox | 100 | 0.8141 | 0.0872 | 0.6951 | 0.1171 |
| BLUSG | GT bbox | 100 | 0.8660 | 0.0891 | 0.7731 | 0.1203 |
| BLUSG | Jittered bbox | 100 | 0.7771 | 0.1357 | 0.6525 | 0.1550 |
| OKU | GT bbox | 100 | 0.9376 | 0.0340 | 0.8842 | 0.0556 |
| OKU | Jittered bbox | 100 | 0.8433 | 0.0841 | 0.7377 | 0.1195 |

These early results suggest that MedSAM performs strongly with oracle bounding boxes, but accuracy drops substantially under degraded prompts. The drop is especially important for the planned prompt-sensitivity study because it approximates realistic user or detector imperfections.

## Immediate Action Items

### 1. Remove Locally Stored Raw Data

The repository currently tracks or contains raw data under `datasets/BLUSG`, `datasets/OKU`, and `datasets/TNSC2020`. The next cleanup should:

- Remove raw image and mask files from the repository.
- Keep only lightweight code, manifests, dataset cards, or small synthetic fixtures if needed for tests.
- Update `.gitignore` to prevent raw datasets, checkpoints, generated visualizations, and large benchmark artifacts from being committed.
- Preserve decoder code and document the expected local cache layout.

### 2. Move Dataset Management to Hugging Face

The larger curated ultrasound dataset should live on Hugging Face and be loaded on demand. The repository should gain a dataset access layer that:

- Downloads or streams data from Hugging Face using `datasets` or `huggingface_hub`.
- Supports local caching without committing raw data.
- Normalizes all datasets into the existing `DecodedSample`-style schema.
- Records dataset version, split, sample ID, anatomy, source dataset, label definition, and license/provenance metadata.
- Supports deterministic subsets for development, validation, and reproducible benchmark runs.

Useful target design:

- `datasets/registry.py` for dataset names, split definitions, and loader construction.
- `datasets/hf_loader.py` for Hugging Face-backed loading.
- Dataset manifests or metadata files that define anatomy/task-specific label mappings.
- A small fixture dataset for CI or smoke tests.

### 3. Benchmark More Segmentation Models

The benchmark should evolve from MedSAM-specific scripts into a model-agnostic interface. Each model adapter should expose a common contract such as:

- Load model/checkpoint/environment.
- Preprocess input image.
- Accept prompt configuration where applicable.
- Run inference.
- Return binary or multiclass masks plus runtime metadata.

Candidate model families to consider include:

- MedSAM / SAM-derived medical variants.
- SAM 2 or other general segmentation foundation models adapted to medical images.
- Ultrasound-specific models such as UltraSAM.
- Specialist supervised baselines such as nnU-Net trained per dataset.

Because these models may require incompatible dependencies, the project should support multiple model environments. Practical options include per-model Conda environments, Docker/Singularity containers, or thin CLI adapters that write predictions to a shared results format.

### 4. Define Research-Meaningful Benchmark Criteria

The benchmark should support the current research questions:

- Identify common failure and underperforming modes across models and datasets.
- Identify performance-deciding factors such as anatomy, image quality, lesion size, boundary ambiguity, acoustic shadowing, speckle, device/source domain, and mask complexity.
- Quantify how suboptimal prompts affect segmentation accuracy.
- Compare zero-shot foundation model capability against small supervised models trained on specific datasets, such as nnU-Net.

Recommended criteria and analysis dimensions:

- Standard segmentation metrics: Dice, IoU, Hausdorff distance, average surface distance, precision, recall, false-positive area, and false-negative area.
- Robustness to prompts: bbox jitter magnitude, bbox padding, shifted boxes, enlarged boxes, shrunken boxes, missing target coverage, point prompts, negative points, and mixed prompt quality.
- Stratified performance: anatomy, dataset source, object size, object shape complexity, image resolution, mask area fraction, number of connected components, and acquisition/source metadata.
- Failure taxonomy: missed object, leakage into background, boundary undersegmentation, boundary oversegmentation, wrong structure selected, fragmented mask, and prompt instability.
- Statistical reporting: confidence intervals, paired comparisons across models on identical samples, per-dataset and pooled summaries, and worst-case or tail-performance analysis.
- Runtime/resource reporting: latency, GPU memory, preprocessing time, model loading time, and batchability.

## Suggested Near-Term Milestones

1. Repository hygiene:
   - Remove raw data and large artifacts from version control.
   - Strengthen `.gitignore`.
   - Add a documented local cache path.

2. Hugging Face dataset path:
   - Define dataset schema and metadata fields.
   - Upload or prepare curated dataset on Hugging Face.
   - Implement on-demand loading and deterministic subset selection.

3. Benchmark harness refactor:
   - Separate dataset loading, model adapters, prompt generation, metric calculation, and result writing.
   - Preserve MedSAM as the first adapter.
   - Save model, dataset, prompt, and environment metadata with every run.

4. Prompt sensitivity experiments:
   - Generalize the existing bbox jitter code into named prompt perturbation protocols.
   - Run controlled sweeps over perturbation severity.
   - Report accuracy degradation curves per dataset/anatomy/model.

5. Supervised baseline:
   - Add nnU-Net training/evaluation workflow for selected datasets.
   - Compare dataset-trained performance against zero-shot foundation model performance on the same splits.

6. Failure analysis:
   - Add per-sample feature extraction from images and masks.
   - Generate ranked failure cases and stratified summaries.
   - Build notebooks or scripts for qualitative review.

## Open Design Decisions

- Whether the Hugging Face dataset should store raw source-like files, normalized image/mask pairs, or both.
- Whether to stream samples directly from Hugging Face or require explicit local materialization before benchmarking.
- How to represent multi-anatomy and multiclass masks in the shared schema.
- Which prompt protocols should be considered clinically realistic versus stress tests.
- Which model environments should be standardized first: Conda, Docker/Singularity, or external CLI adapters.
- Which supervised baseline datasets and train/test splits should be prioritized for nnU-Net.

## Working Assumptions

- Raw medical imaging data should not live in the Git repository.
- All benchmark runs should be reproducible from dataset version, split, model version/checkpoint, prompt protocol, and environment metadata.
- Foundation models should be evaluated both under oracle prompts and degraded/realistic prompts.
- Mean performance alone is insufficient; the project should emphasize failure modes, stratified results, and prompt robustness.

