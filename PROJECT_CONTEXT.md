# Project Context: Ultrasound Multi-Anatomy Segmentation Benchmark

## Research Goal

This project benchmarks promptable segmentation foundation models on curated ultrasound datasets across multiple anatomies. The intended study should go beyond a leaderboard: it should identify when SAM-style models succeed or fail as a function of anatomy, dataset source, target morphology, prompt quality, image quality, and mask semantics.

The current practical focus is GT-box prompted benchmarking for MedSAM and SAMUS. GT boxes should be interpreted as an oracle-prompt upper bound, not as a clinical workflow. Prompt jitter experiments exist for MedSAM and are useful as a first robustness probe, but the main benchmark story is still cross-dataset GT-box behavior.

Working academic theme: SAM-based ultrasound segmentation works best for compact region-like targets inside accurate prompts, but degrades on thin structures, sparse targets, ambiguous boundaries, very small lesions, broad boxes with low foreground density, and semantically mixed masks.

## Current Repository Status

The repository has a registry-driven dataset layer:

- `datasets/registry.py` records dataset keys, local cache roots, decoder classes, anatomy labels, and Hugging Face zip paths.
- `datasets/hf_materialize.py` materializes registered dataset zips on demand.
- `datasets/loader.py` builds dataset decoders from benchmark CLI args.
- `datasets/common.py` defines the shared `DecodedSample` schema plus common image, mask, bbox, Dice, and IoU helpers.

Registered datasets:

| Dataset key | Anatomy | Decoder status |
| --- | --- | --- |
| `tnsc2020` | Thyroid | PNG image/mask pairs with optional category metadata. |
| `blusg` | Breast | Flat case PNGs with tumor and optional other-lesion masks. |
| `oku` | Kidney | PNG images plus reviewed polygon CSV annotations; default anatomy is `Capsule`. |
| `ultrabones100k` | Bone | Nested specimen/anatomy/record folders with timestamped image/label pairs; default benchmark fills thin bone-surface labels into bone-shadow regions. |
| `camus` | Heart | NIfTI ED/ES volumes and optional cine half-sequence volumes, paired with `_gt.nii.gz` masks; labels are expanded into per-class binary targets for LV and LA by default. |
| `aulid` | Liver | JPG images with JSON polygon masks for `mass`, `liver`, or `outline`. |
| `uns` | Nerve | TIFF images paired with `_mask.tif` masks. |
| `roblus` | Lung | Cleaned RobLUS decoder for subjects `AP`, `BM`, `CP`, `SG`, and `XM`; default benchmark target is class-specific `pleural_line`. |
| `ussc` | Spinal Cord | RGB semantic masks from the upstream `SegmentationDataset.zip`; CAMUS-style one binary target per selected non-background class. |

BCU_PD has been removed from the registry and wrapper scripts.

## Benchmark Harness

There are two prompting paradigms, reported in **separate tables that are never cross-compared**: box-prompted (GT-derived box = oracle localization) and text-prompted (concept name only = harder, deployment-realistic). A GT box is a privileged cue, so the two are not comparable by design.

Box-prompted model engines:

- `benchmarks/medsam_inference.py`: GT-box prompted MedSAM benchmark with optional bbox jitter controls, checkpoint auto-download, Dice/IoU/latency reporting, visualization output, and progress reporting.
- `benchmarks/samus_inference.py`: GT-box prompted SAMUS benchmark with checkpoint auto-download, batching, optional threaded sample loading, Dice/IoU/latency reporting, visualization output, and progress reporting.
- `benchmarks/ultrasam_inference.py`: GT-box prompted UltraSAM benchmark (COCO export + MMDetection runtime); selects the highest-confidence predicted instance per sample.

Text-prompted model engine:

- `benchmarks/medical_sam3_inference.py`: text-prompted **Medical SAM3** benchmark (model token `medicalsam3`; always spelled "Medical SAM3", never "MedSAM3", to avoid confusion with future Medical SAM3 variants). The prompt is a canonical clinical concept string from `datasets/label_text.py` (`concept_for(dataset, label)`) — derived from the class/label taxonomy, never from the mask. It selects the highest-confidence mask returned for the text query (`argmax(scores)`, matching the paper). Multi-class datasets (e.g. CAMUS LV/LA) are queried per class and scored independently with the existing per-target scoring. Empty-GT frames are skipped, matching the box engines. The fine-tuned `checkpoint_2D.pt` is auto-downloaded from the `Chongcong/Medical-SAM3` HF repo. The `sam3` model package (the Medical-SAM3 repo's own package) must be installed in the environment — see `requirements.txt`; it is imported like `segment_anything`, not referenced as a source clone.

All engines write:

- `per_sample_metrics.csv` (the text engine adds `concept` and `text_score` columns; its `bbox` column is inert)
- `summary.json`
- `visualizations/*.png`

Result-directory naming encodes the protocol so the reporting layer keeps tables separate: box runs are `{model}_gt_bbox_{dataset}` / `{model}_jitter_bbox_{dataset}`; text runs are `medicalsam3_text_prompt_{dataset}`. `analysis/model_across_datasets.py --protocol text` reports the text table on its own.

Visualization dumping samples evenly across the evaluated run when `--save-vis N` is used. Existing visualization files are not automatically cleaned when rerunning into an existing result folder, so stale PNGs can remain unless the folder is cleared first.

Current GT wrapper scripts cover all eight registered datasets for both MedSAM and SAMUS:

- MedSAM: TNSC2020, BLUSG, OKU, UltraBones100k, CAMUS, AULID, UNS, RobLUS.
- SAMUS: TNSC2020, BLUSG, OKU, UltraBones100k, CAMUS, AULID, UNS, RobLUS.

Wrapper defaults are now aligned for comparable default reruns:

- `--max-samples 100` for AULID, BLUSG, OKU, TNSC2020, UltraBones100k, and UNS.
- `--max-samples 4000` for CAMUS ED/ES target rows, corresponding to 2000 source frames x 2 labels.
- `--max-samples 764` for RobLUS, which yields 615 annotated pleural-line samples and skips 149 empty-mask frames because no GT box can be generated.
- `--save-vis 20`.
- `--box-padding 10` only for UltraBones100k and `0` otherwise.
- MedSAM GT wrappers explicitly set `--bbox-jitter-prob 0.0`.

RobLUS benchmark reporting should stay class-specific. Merged pleural-line/rib-shadow runs are not a valid headline protocol because a single combined bbox is not a clean prompt for separate structures.

CAMUS defaults exclude half-sequences and evaluate ED/ES only: 500 patients x 2 views x 2 phases = 2000 source frames. The CAMUS decoder now emits one binary target sample per selected label (`LV` and `LA` by default), so the default CAMUS benchmark has 4000 target evaluations. With `--include-half-sequence`, cine volumes expand into frame-level source samples before label expansion.

## Running Analysis Scripts (guidance for the assistant)

Whenever asked to run something (EDA, results analysis, model comparison, etc.),
first check `scripts/` for the matching runner and run that — do not invoke the
underlying python directly:

- EDA: `scripts/run_eda_analysis.sh`
- Results vs EDA: `scripts/run_quantitative_all.sh`
- Compare models: `scripts/run_compare_models.sh`

If parameters need to change (datasets, metrics, models, toggles), edit the
relevant runner in place and run it. Keep these scripts as simple and plain as
the existing ones: a venv activation, a few config variables up top, and the
python invocation(s) — no added control flow, helpers, or complexity.

## Analysis Workspace

The `analysis/` folder currently provides lightweight result aggregation and figure generation:

- `analysis/model_across_datasets.py`: same-model cross-dataset Dice/IoU summaries and paper-style plots with mean bars, standard-deviation error bars, and per-sample scatter points.
- `analysis/compare_models_same_dataset.py`: matched-sample MedSAM-vs-SAMUS comparison using shared `sample_id`s to avoid misleading comparisons when result folders use different sample caps.

Generated figures currently exist under `analysis/figures/`:

- `model_across_datasets_gt_bbox_medsam.png`
- `model_across_datasets_gt_bbox_samus.png`
- `compare_models_same_dataset_gt_bbox_medsam_vs_samus.png`

## Current Local Results Snapshot

Current local result folders include GT-box runs for MedSAM and SAMUS on all eight registered datasets, plus MedSAM jittered-box runs for BLUSG, OKU, and TNSC2020.

The current CAMUS result folders were generated before the per-class target decoder change and should be treated as legacy merged-mask results until rerun.

Summary-level GT-box results currently present in `results/`:

| Result folder | Evaluated | Skipped | Dice mean | IoU mean |
| --- | ---: | ---: | ---: | ---: |
| `medsam_gt_bbox_aulid` | 100 | 0 | 0.7840 | 0.6595 |
| `medsam_gt_bbox_blusg` | 100 | 0 | 0.8660 | 0.7731 |
| `medsam_gt_bbox_camus` | 2000 | 0 | 0.8361 | 0.7209 |
| `medsam_gt_bbox_oku` | 100 | 0 | 0.9376 | 0.8842 |
| `medsam_gt_bbox_roblus` | 615 | 149 | 0.6018 | 0.4383 |
| `medsam_gt_bbox_tnsc2020` | 100 | 0 | 0.9444 | 0.8966 |
| `medsam_gt_bbox_ultrabones100k` | 100 | 0 | 0.8765 | 0.7814 |
| `medsam_gt_bbox_uns` | 39 | 61 | 0.9203 | 0.8555 |
| `samus_gt_bbox_aulid` | 635 | 0 | 0.8325 | 0.7164 |
| `samus_gt_bbox_blusg` | 252 | 0 | 0.8109 | 0.6879 |
| `samus_gt_bbox_camus` | 2000 | 0 | 0.8500 | 0.7411 |
| `samus_gt_bbox_oku` | 487 | 0 | 0.8540 | 0.7477 |
| `samus_gt_bbox_roblus` | 615 | 149 | 0.4998 | 0.3441 |
| `samus_gt_bbox_tnsc2020` | 3644 | 0 | 0.8396 | 0.7261 |
| `samus_gt_bbox_ultrabones100k` | 100 | 0 | 0.8597 | 0.7544 |
| `samus_gt_bbox_uns` | 2323 | 3312 | 0.8034 | 0.6752 |

Important interpretation notes:

- Some SAMUS result folders were generated before the wrapper defaults were aligned, so raw summary-level MedSAM-vs-SAMUS comparisons are not always fair.
- Current CAMUS summaries may reflect older label selections; rerun the CAMUS wrappers to produce LV/LA target-level results and class-level summaries.
- Use `analysis/compare_models_same_dataset.py` for model comparison because it restricts each pair to shared `sample_id`s.
- The current matched comparison suggests SAMUS is slightly higher on AULID and CAMUS, while MedSAM is higher on BLUSG, OKU, RobLUS, TNSC2020, UltraBones100k, and UNS.
- RobLUS is the clearest shared failure regime: both models perform poorly on pleural-line segmentation, especially compared with compact region-like targets.
- MedSAM jittered-box results show substantial prompt sensitivity: Dice drops by about 0.09 on BLUSG and OKU, and about 0.13 on TNSC2020 compared with matched GT-box runs.

## Repository Hygiene / Infrastructure Status

- Raw dataset materializations, checkpoints, cache folders, and zips are ignored by `.gitignore`.
- `results/` is currently not ignored and many result summaries, CSVs, and visualization PNGs are tracked. This is useful for local analysis but should be revisited if repository size or artifact churn becomes painful.
- Dataset access is Hugging Face-backed and local-cache aware.
- `python -m datasets.registry` works and lists the eight supported datasets.
- CAMUS dependencies are represented in `requirements.txt` through `nibabel`.
- UNS TIFF decoding support is represented through `imagecodecs`.
- MedSAM and SAMUS both have progress reporting.
- SAMUS supports CAMUS and has explicit `--max-samples` wrapper defaults aligned with MedSAM for future reruns.
- UltraBones100k is the one decoder that intentionally changes source label geometry for the main benchmark by filling thin surface labels into bone-shadow regions; original thin labels should be used only for sensitivity checks.

## Known Limitations

- GT-box prompting is an oracle condition and should not be treated as a deployable clinical workflow.
- The benchmark metric set now includes overlap, classification-style pixel metrics, boundary metrics, size error, and latency; clDice remains a structure-specific future addition.
- Empty masks are skipped by GT-box engines, so current summaries do not evaluate specificity or false positives on empty frames.
- Result metadata is incomplete: summaries do not yet consistently store command line, git SHA, checkpoint hash, package versions, dataset revision, decoder options, prompt protocol, and sample manifest.
- Sample selection is not manifest-driven yet. First-`N` ordering can bias small capped runs, especially for datasets grouped by patient, class, or acquisition order.
- MedSAM and SAMUS engines duplicate benchmark logic instead of sharing a model-adapter and runner abstraction.
- CAMUS target-mode visualizations now stream grouped multi-color GT/prediction overlays by class as each selected source frame finishes, but other datasets still use single-target visualizations unless their decoders provide target metadata.
- Current visualizations are evenly spaced samples, not a deliberate best/median/worst/failure atlas.

## Next Phase TODOs

### Academic Direction

1. Rerun a clean, fixed benchmark matrix:
   - Use the same sample manifests, prompt protocols, result-directory hygiene, and model/dataset defaults across MedSAM and SAMUS.
   - Treat GT-box results as an oracle-prompt upper bound, not as the final clinical-use setting.

2. Define stratified evaluation manifests:
   - Stratify by dataset, anatomy, patient/subject, class, view/phase, object size, bounding-box density, and target type.
   - Avoid relying on first-`N` dataset order when reporting headline results.

3. Add prompt-sensitivity protocols:
   - Compare oracle boxes, jittered boxes, loose boxes, point prompts, multi-click prompts, and realistic detector/user boxes.
   - Use prompt perturbation to measure robustness, not just best-case segmentation capacity.

4. Evaluate structure-specific labels:
   - Rerun CAMUS with the LV/LA target decoder and report per-class results.
   - Keep RobLUS pleural line and rib shadow as separate tasks.
   - TODO: Add clDice for RobLUS pleural-line segmentation only, because the pleural line is a thin structure where centerline/topology preservation is meaningful.
   - Keep UltraBones100k filled bone-shadow masks as the main target, with original thin labels as a sensitivity check.

5. Add supervised and newer foundation-model baselines:
   - Include task-trained baselines such as U-Net or nnU-Net.
   - Add future SAM-family models such as UltraSAM and SAM 2 variants when adapters are available.

6. Build a failure-analysis atlas:
   - Save and review best, median, worst, and model-disagreement cases.
   - Summarize failures by anatomy, object size, bbox density, view/phase, and mask complexity.

### Engineering Direction

1. Add a central benchmark configuration layer:
   - Use a YAML/JSON config or central Python defaults module for dataset, model, prompt protocol, sample manifest, device, `max-samples`, `save-vis`, `box-padding`, and output naming.
   - Record the resolved config in every `summary.json`.

2. Refactor into shared benchmark abstractions:
   - Introduce a common `DatasetAdapter` / `ModelAdapter` / `BenchmarkRunner` structure.
   - Keep MedSAM and SAMUS as concrete adapters while sharing metric computation, visualization selection, result writing, and provenance capture.

## Open Design Decisions

- Whether Hugging Face should store source-like raw files, normalized image/mask pairs, or both.
- Whether benchmark runs should always materialize data locally or support streaming.
- How to represent multiclass, multi-structure, and multi-instance masks beyond the current binary GT-box setup.
- Which prompt perturbation protocols are clinically realistic versus stress tests.
- Which model environment strategy should be standardized first: Conda, Docker/Singularity, or external CLI adapters.
- Which supervised baselines should be prioritized first.

## Working Assumptions

- Raw medical imaging data should not live in the Git repository.
- GT-box benchmarking is the current priority, with degraded-prompt experiments used as robustness probes rather than the main benchmark axis.
- Mean performance is not enough; the project should emphasize stratified results, tail failures, and qualitative inspection.
- All reported runs should eventually be reproducible from dataset version, decoder options, sample subset, model checkpoint, prompt protocol, environment metadata, and exact command/config.
