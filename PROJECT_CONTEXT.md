# Project Context: Ultrasound Multi-Anatomy Segmentation Benchmark

## Research Goal

This project benchmarks segmentation foundation models on curated ultrasound datasets across multiple anatomies. The study should go beyond a leaderboard: the benchmark should help identify when models succeed or fail by anatomy, dataset source, object type, prompt quality, and image/mask characteristics.

The current practical focus is GT-box prompted benchmarking. Prompt jitter and other degraded-prompt protocols exist for MedSAM, but near-term runs should prioritize clean ground-truth-box comparisons across models and datasets.

## Current Repository Status

The repository now has a registry-driven dataset layer:

- `datasets/registry.py` records dataset keys, local cache roots, decoder classes, and Hugging Face zip paths.
- `datasets/hf_materialize.py` materializes registered dataset zips on demand.
- `datasets/loader.py` builds dataset decoders from benchmark CLI args.
- `datasets/common.py` defines the shared `DecodedSample` schema and common image/mask helpers.

Implemented decoders:

| Dataset key | Anatomy | Decoder status |
| --- | --- | --- |
| `tnsc2020` | Thyroid | PNG image/mask pairs with optional category metadata. |
| `blusg` | Breast | Flat case PNGs with tumor and optional other-lesion masks. |
| `oku` | Kidney | PNG images plus reviewed polygon CSV annotations; default anatomy is `Capsule`. |
| `ultrabones100k` | Bone | Nested specimen/anatomy/record folders with timestamped image/label pairs. |
| `camus` | Heart | NIfTI ED/ES and optional cine half-sequence volumes, paired with `_gt.nii.gz` masks. |
| `aulid` | Liver | JPG images with JSON polygon masks for `mass`, `liver`, or `outline`. |
| `uns` | Nerve | TIFF images paired with `_mask.tif` masks. |
| `roblus` | Lung | Cleaned RobLUS decoder for subjects `AP`, `BM`, `CP`, `SG`, and `XM`; reads `US_*.jpg` frames paired with `pleural_line` and `rib_shadow` masks. |

BCU_PD has been removed and is no longer part of the registry or benchmark wrappers.

## Benchmark Harness

Current model engines:

- `benchmarks/medsam_inference.py`: GT-box prompted MedSAM benchmark with optional bbox jitter controls, checkpoint auto-download, Dice/IoU/latency reporting, and a progress bar.
- `benchmarks/samus_inference.py`: GT-box prompted SAMUS benchmark with checkpoint auto-download, batching, optional threaded loading, Dice/IoU/latency reporting, and progress output.

Both engines write:

- `per_sample_metrics.csv`
- `summary.json`
- `visualizations/*.png`

Visualization dumping now samples evenly across the evaluated run when `--save-vis N` is used, instead of saving only the first `N` consecutive cases.
Existing visualization files are not cleaned automatically when rerunning into an existing result folder, so old PNGs should be removed before regenerating qualitative samples for a run.

Current GT wrapper scripts cover:

- MedSAM: TNSC2020, BLUSG, OKU, UltraBones100k, CAMUS, AULID, UNS, RobLUS.
- SAMUS: TNSC2020, BLUSG, OKU, UltraBones100k, CAMUS, AULID, UNS, RobLUS.

RobLUS GT wrappers iterate all 764 cleaned samples. The current GT-box engines evaluate the 615 annotated pleural-line/rib-shadow frames and skip the 149 empty-mask negative controls because no bounding-box prompt can be generated.

The GT wrapper defaults are aligned across MedSAM and SAMUS for comparable default runs: `--max-samples 100` for AULID, BLUSG, OKU, TNSC2020, UltraBones100k, and UNS; `--max-samples 2000` for CAMUS ED/ES; `--save-vis 20`; `--box-padding 10` only for UltraBones100k and `0` otherwise. MedSAM wrappers additionally pass `--bbox-jitter-prob 0.0` because the MedSAM engine exposes jitter controls.

The CAMUS local materialization contains 500 patient folders. By default, the decoder excludes cine half-sequences and benchmarks 2000 ED/ES samples: 500 patients x 2 views x 2 phases. With `--include-half-sequence`, cine volumes are expanded into frame-level samples.

The project also has an `analysis/` workspace for quantitative result aggregation and paper-style figure generation:

- `analysis/model_across_datasets.py`: same-model cross-dataset Dice/IoU mean/std summaries, with mean bars, standard-deviation error bars, and per-sample scatter plots.
- `analysis/compare_models_same_dataset.py`: matched-sample MedSAM-vs-SAMUS comparisons by dataset, using shared `sample_id`s to avoid misleading comparisons when older result folders used different sample caps.

## Completed Hygiene / Infrastructure

- Raw dataset folders, checkpoints, generated results, cache folders, and zips are ignored by `.gitignore`.
- RobLUS local materializations are ignored under `datasets/RobLUS/`.
- RobLUS has been cleaned locally: unlabeled frames were removed except selected negative controls, empty pleural-line/rib-shadow masks were created for those controls, and unused `rib`/`cartilage` mask folders were removed.
- Dataset decoders preserve original annotation semantics by default. UltraBones100k is the explicit exception: thin bone-surface labels are hole-filled into a bone-shadow region for the main benchmarks because line-based segmentation is not a natural target for region-prompted foundation models, and the filled acoustic shadow is clinically meaningful.
- Dataset access is Hugging Face-backed and local-cache aware.
- `python -m datasets.registry` works and lists the supported registered datasets.
- CAMUS dependencies are represented in `requirements.txt` through `nibabel`.
- SAMUS benchmark CLI now accepts CAMUS.
- MedSAM benchmark now has progress reporting similar to SAMUS.
- SAMUS GT wrapper scripts now include explicit `--max-samples` defaults aligned with MedSAM wrappers.
- Initial quantitative analysis scripts and generated figures exist under `analysis/`.

## Existing Results Snapshot

Existing local result folders include MedSAM GT runs for AULID, BLUSG, CAMUS, OKU, TNSC2020, UltraBones100k, and UNS; SAMUS GT runs for the same seven datasets; and MedSAM jittered bbox runs for BLUSG, OKU, and TNSC2020. Interpret results generated before the wrapper alignment carefully if the model/dataset pair used a different sample cap.

Interpret older results carefully:

- Earlier SAMUS wrappers ran full decoded datasets by default, while MedSAM wrappers often capped runs at 100 samples.
- CAMUS MedSAM was initially run on 100 samples, which covered only the first 25 patients across 2CH/4CH ED/ES. The CAMUS GT wrapper now targets the full default ED/ES set of 2000 samples.
- Existing visualization samples may differ across model folders if one folder was generated before the matched `--max-samples` defaults or if stale PNGs remain from an earlier run.
- Prompt-jitter outputs are useful for future prompt-sensitivity analysis but are not the immediate benchmark priority.

## Near-Term TODOs

1. Finish GT-box model benchmarking:
   - Run MedSAM and SAMUS GT benchmarks on the selected datasets using the matched wrapper defaults documented in `README.md`.
   - Keep UltraBones100k filled bone-shadow masks as the main benchmark target; use original thin surface labels only for sensitivity checks.
   - Run RobLUS GT wrappers using the cleaned lung dataset with pleural-line and rib-shadow masks.
   - Keep result folders model/dataset/prompt-specific, for example `results/medsam_gt_bbox_camus`.

2. Improve result visualization sampling:
   - Add a post-run visualization mode that saves best, median, worst, plus evenly spaced cases.
   - Ranking should use Dice or IoU and ideally avoid duplicates when a best/median/worst case is already part of the evenly spaced set.
   - This will make qualitative folders include representative successes, typical cases, and clear failures.

3. Expand metric reporting:
   - Add Hausdorff distance, average surface distance, precision, recall, false-positive area, and false-negative area.
   - Add mask/image descriptors such as mask area fraction, connected components, bounding-box size, and image resolution.

4. Improve reproducibility metadata:
   - Save model checkpoint path/version, dataset revision, decoder options, prompt protocol, device, package versions, and command line in every `summary.json`.
   - Add deterministic sample selection or manifest-based subsets for comparable cross-model runs.

5. Add unified benchmark hyperparameter control:
   - Introduce one shared configuration layer for all benchmark cases so values such as `max-samples`, `save-vis`, `box-padding`, device, prompt protocol, and output naming do not need to be edited separately in `medsam_inference.py`, `samus_inference.py`, or each wrapper script.
   - Prefer a small YAML/JSON config or central Python defaults module that wrappers can import and model engines can record into `summary.json`.

6. Move toward model-agnostic adapters:
   - Keep MedSAM and SAMUS working as concrete baselines.
   - Define a common model adapter contract for loading, preprocessing, prompt ingestion, inference, and prediction output.
   - Add future model families such as UltraSAM, SAM 2 variants, and supervised baselines such as nnU-Net.

7. Add failure analysis:
   - Generate ranked failure cases and stratified summaries by anatomy, dataset, object size, view/phase, and mask complexity.
   - Extend `analysis/` scripts and/or notebooks for qualitative review and figure preparation.

## Open Design Decisions

- Whether Hugging Face should store source-like raw files, normalized image/mask pairs, or both.
- Whether benchmark runs should always materialize data locally or support streaming.
- How to represent multiclass and multi-object masks beyond the current binary GT-box setup.
- Which prompt perturbation protocols are clinically realistic versus stress tests.
- Which model environment strategy should be standardized first: Conda, Docker/Singularity, or external CLI adapters.
- Which datasets should be prioritized for supervised nnU-Net baselines.

## Working Assumptions

- Raw medical imaging data should not live in the Git repository.
- GT-box benchmarking is the current priority; degraded-prompt experiments should be revisited later.
- Mean performance is not enough; the project should emphasize stratified results, tail failures, and qualitative inspection.
- All reported runs should be reproducible from dataset version, decoder options, sample subset, model checkpoint, prompt protocol, and environment metadata.
