#!/usr/bin/env bash
set -euo pipefail

{
export CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_IDX=0
SUBSET="dahliafull"
TEST_DATA_ROOT="/scr/yuegao/TRELLIS_datasets/BlenderFlowers_Dahliamorefix20kfullTest_merged"
TEST_SCENE_NAME="Dahliamorefix_020033"
FRAME_IDX=100
FLOWER_NAME="dahlia"
NUM_FRAMES=49
SEED=1
OUTPUT_ROOT="/scr/yuegao/wan2_2_i2v_results_finetune"
LORA_BASE="${LORA_BASE:-/scr/yuegao/wan_models/train_blenderflowers}"
LORA_EPOCH="5"

python3 "${SCRIPT_DIR}/infer_wan2_2_i2v_test_scene.py" \
  --gpu_idx "${GPU_IDX}" \
  --subset "${SUBSET}" \
  --reverse \
  --test_data_root "${TEST_DATA_ROOT}" \
  --test_scene_name "${TEST_SCENE_NAME}" \
  --frame_idx "${FRAME_IDX}" \
  --flower_name "${FLOWER_NAME}" \
  --output_root "${OUTPUT_ROOT}" \
  --num_frames "${NUM_FRAMES}" \
  --seed "${SEED}" \
  --lora_high_noise_path "${LORA_BASE}/${FLOWER_NAME}_high" \
  --lora_low_noise_path "${LORA_BASE}/${FLOWER_NAME}_low" \
  $([ -n "${LORA_EPOCH}" ] && echo "--lora_epoch ${LORA_EPOCH}")

exit 0
}
