#!/usr/bin/env bash
set -euo pipefail

# UltraSAM needs the OpenMMLab stack (mmcv._ext), which lives in the `UltraSam` conda env.
eval "$(conda shell.bash hook)"
conda activate UltraSam

gpu_id=1
prompt_types=(
  point
  point_bbox
)

# Fraction of equally-spaced frames per record to sample for UltraBones100k (huge dataset);
# applied only to ultrabones100k below.
ultrabones_frame_fraction=0.05

# Datasets to benchmark (each has a benchmarks/test_ultrasam_gt_bbox_<dataset>.py wrapper).
datasets=(
  # aulid
  # blusg
  # busi
  # camus
  # oku
  # uns
  busbra
  gist514
  mmotu
  tnsc2020
  ultrabones100k
  umud
  roblus
)

# The wrappers default the source tree + checkpoint to work_dir/UltraSam (the devcontainer
# path); the args below override them for a local checkout and pass through to
# benchmarks/ultrasam_inference.py.
for prompt_type in "${prompt_types[@]}"; do
  for dataset in "${datasets[@]}"; do
    output_dir="results/ultrasam_gt_${prompt_type}_${dataset}"
    echo "Running UltraSAM GT-${prompt_type} inference on ${dataset}..."
    extra_args=()
    if [[ "${dataset}" == "ultrabones100k" ]]; then
      extra_args+=(--ultrabones-frame-fraction "${ultrabones_frame_fraction}")
    fi
    if [[ "${prompt_type}" == "point_bbox" ]]; then
      extra_args+=(
        --bbox-mode individual
        --bbox-scale-factor 1.0
        --bbox-translation-fraction 0.0
        --point-prompt-jitter-fraction 0.0
      )
    fi
    python "benchmarks/test_ultrasam_gt_bbox_${dataset}.py" \
      --ultrasam-dir UltraSam \
      --checkpoint UltraSam/weights/UltraSam.pth \
      --no-auto-download-checkpoint \
      --device "cuda:${gpu_id}" \
      --prompt-type "${prompt_type}" \
      --output-dir "${output_dir}" \
      "${extra_args[@]}"
  done
done
