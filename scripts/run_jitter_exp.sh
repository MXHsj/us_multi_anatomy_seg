#!/usr/bin/env bash
set -euo pipefail

# UltraSAM needs the OpenMMLab stack (mmcv._ext), which lives in the `UltraSam` conda env.
eval "$(conda shell.bash hook)"
conda activate UltraSam

gpu_id=0
jitter_root="experiments/prompt_robustness"

# Fraction of equally-spaced frames per record to sample for UltraBones100k (huge dataset);
# applied only to ultrabones100k below.
ultrabones_frame_fraction=0.05

# Datasets to benchmark (each has a benchmarks/test_ultrasam_gt_bbox_<dataset>.py wrapper).
datasets=(
  aulid
  blusg
  busbra
  busi
  camus
  oku
  mmotu
  gist514
  roblus
  tnsc2020
  ultrabones100k
  umud
  uns
)

translation_fractions=(
  0
  0.025
  0.05
  0.075
  0.10
  0.125
  0.15
)

scale_factors=(
  0.85
  0.9
  0.95
  1
  1.05
  1.10
  1.15
)

# The wrappers default the source tree + checkpoint to work_dir/UltraSam (the devcontainer
# path); the args below override them for a local checkout and pass through to
# benchmarks/ultrasam_inference.py.
for dataset in "${datasets[@]}"; do
  for scale_factor in "${scale_factors[@]}"; do
    echo "Running UltraSAM scale ${scale_factor} inference on ${dataset}..."
    extra_args=()
    if [[ "${dataset}" == "ultrabones100k" ]]; then
      extra_args+=(--ultrabones-frame-fraction "${ultrabones_frame_fraction}")
    fi
    python "benchmarks/test_ultrasam_gt_bbox_${dataset}.py" \
      --ultrasam-dir UltraSam \
      --checkpoint UltraSam/weights/UltraSam.pth \
      --no-auto-download-checkpoint \
      --device "cuda:${gpu_id}" \
      --max-samples 100 \
      --bbox-scale-factor "${scale_factor}" \
      --bbox-translation-fraction 0.0 \
      --seed 0 \
      --output-dir "${jitter_root}/scale/ultrasam_scale_${scale_factor}_bbox_${dataset}" \
      "${extra_args[@]}"
  done
done

for dataset in "${datasets[@]}"; do
  for translation_fraction in "${translation_fractions[@]}"; do
    echo "Running UltraSAM translation ${translation_fraction} inference on ${dataset}..."
    extra_args=()
    if [[ "${dataset}" == "ultrabones100k" ]]; then
      extra_args+=(--ultrabones-frame-fraction "${ultrabones_frame_fraction}")
    fi
    python "benchmarks/test_ultrasam_gt_bbox_${dataset}.py" \
      --ultrasam-dir UltraSam \
      --checkpoint UltraSam/weights/UltraSam.pth \
      --no-auto-download-checkpoint \
      --device "cuda:${gpu_id}" \
      --max-samples 100 \
      --bbox-scale-factor 1.0 \
      --bbox-translation-fraction "${translation_fraction}" \
      --seed 0 \
      --output-dir "${jitter_root}/trans/ultrasam_trans_${translation_fraction}_bbox_${dataset}" \
      "${extra_args[@]}"
  done
done

datasets_csv="$(IFS=,; echo "${datasets[*]}")"
translation_fractions_csv="$(IFS=,; echo "${translation_fractions[*]}")"
scale_factors_csv="$(IFS=,; echo "${scale_factors[*]}")"

python analysis/analyse_jitter_exp.py \
  --experiment both \
  --jitter-results-dir "${jitter_root}" \
  --datasets "${datasets_csv}" \
  --scales "${scale_factors_csv}" \
  --translations "${translation_fractions_csv}" \
  --metrics "dice"
