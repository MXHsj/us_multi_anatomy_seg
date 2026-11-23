#!/usr/bin/env bash
set -euo pipefail

# UltraSAM needs the OpenMMLab stack (mmcv._ext), which lives in the `UltraSam` conda env.
eval "$(conda shell.bash hook)"
conda activate UltraSam

gpu_id=1
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

point_prompt_jitter_fractions=(
  0
  0.025
  0.05
  0.075
  0.10
  0.125
  0.15
)

heatmap_datasets=(
  mmotu
  busbra
  gist514
  tnsc2020
  ultrabones100k
  umud
  roblus
)
heatmap_result_label="Dice"
heatmap_dice_threshold="1.0"
heatmap_ultrabones_min_solidity="0.2"
heatmap_filter_tnsc2020_target_bbox_one="true"
heatmap_tnsc2020_min_aspect_ratio="0.2"
heatmap_eda_metrics="target_area_fraction,target_bbox_area_ratio,solidity,circularity,aspect_ratio_feret"
heatmap_eda_labels="Target/image,Target/bbox,Solidity,Circularity,Aspect ratio"
heatmap_correlation="spearman"

qualitative_seed="812528458"
qualitative_gist_example_index="15"
qualitative_scale_values="0.85,1,1.15"
qualitative_translation_values="0,0.05,0.1"
qualitative_output_dir="analysis/figures/prompt_robustness"
qualitative_gist_config="analysis/figures/qualitative/dice_0_5/run_config.json"
qualitative_gist_sample_id="lmym/original_lmym064_1"
qualitative_reference_bbox_color="#00d7ff"

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

for dataset in "${datasets[@]}"; do
  for point_prompt_jitter_fraction in "${point_prompt_jitter_fractions[@]}"; do
    echo "Running UltraSAM point prompt jitter ${point_prompt_jitter_fraction} inference on ${dataset}..."
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
      --prompt-type point \
      --bbox-scale-factor 1.0 \
      --bbox-translation-fraction 0.0 \
      --point-prompt-jitter-fraction "${point_prompt_jitter_fraction}" \
      --seed 0 \
      --output-dir "${jitter_root}/point/ultrasam_point_${point_prompt_jitter_fraction}_point_${dataset}" \
      "${extra_args[@]}"
  done
done

datasets_csv="$(IFS=,; echo "${datasets[*]}")"
translation_fractions_csv="$(IFS=,; echo "${translation_fractions[*]}")"
scale_factors_csv="$(IFS=,; echo "${scale_factors[*]}")"
heatmap_datasets_csv="$(IFS=,; echo "${heatmap_datasets[*]}")"

python analysis/analyse_jitter_exp.py \
  --experiment both \
  --jitter-results-dir "${jitter_root}" \
  --datasets "${datasets_csv}" \
  --scales "${scale_factors_csv}" \
  --translations "${translation_fractions_csv}" \
  --metrics "dice" \
  --heatmap-datasets "${heatmap_datasets_csv}" \
  --heatmap-result-label "${heatmap_result_label}" \
  --heatmap-dice-threshold "${heatmap_dice_threshold}" \
  --heatmap-ultrabones-min-solidity "${heatmap_ultrabones_min_solidity}" \
  --heatmap-filter-tnsc2020-target-bbox-one "${heatmap_filter_tnsc2020_target_bbox_one}" \
  --heatmap-tnsc2020-min-aspect-ratio "${heatmap_tnsc2020_min_aspect_ratio}" \
  --heatmap-eda-metrics "${heatmap_eda_metrics}" \
  --heatmap-eda-labels "${heatmap_eda_labels}" \
  --heatmap-correlation "${heatmap_correlation}"

python analysis/qualitative_jitter_robustness.py \
  --jitter-results-dir "${jitter_root}" \
  --output-dir "${qualitative_output_dir}" \
  --datasets "${datasets_csv}" \
  --scale-values "${qualitative_scale_values}" \
  --visual-translation-values "${qualitative_translation_values}" \
  --seed "${qualitative_seed}" \
  --gist-config "${qualitative_gist_config}" \
  --gist-example-index "${qualitative_gist_example_index}" \
  --gist-sample-id "${qualitative_gist_sample_id}" \
  --reference-bbox-color "${qualitative_reference_bbox_color}" \
  --gpu-id "${gpu_id}"
