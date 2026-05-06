#!/usr/bin/env bash
# Run Wan 2.2 I2V Blender flowers finetuning for all sub-datasets: both high-noise and
# low-noise LoRA per flower (10 jobs total). Optionally create metadata first:
#   bash tools_blenderflowers/create_blender_flowers_metadata_all.sh
# Run from DiffSynth-Studio repo root. For one GPU, run scripts sequentially instead.
#
# Wan 2.2 I2V (README.md, docs/zh/Model_Details/Wan.md, tools/train_I2V_A14B_bubble_*.sh):
#   - high_noise: timesteps [900, 1000] (max_timestep_boundary 0.358, min 0)
#   - low_noise:  timesteps [0, 900)    (max 1, min_timestep_boundary 0.358)
#   Both LoRAs are independent; no required training order. Inference loads both.
set -euo pipefail

{
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# High-noise LoRA (5 jobs; timesteps 900–1000) — GPUs 0–4
CUDA_VISIBLE_DEVICES=0 bash "${SCRIPT_DIR}/train_wan_i2v_blenderflowers_dahlia_high.sh" &
CUDA_VISIBLE_DEVICES=1 bash "${SCRIPT_DIR}/train_wan_i2v_blenderflowers_daisy_high.sh" &
CUDA_VISIBLE_DEVICES=2 bash "${SCRIPT_DIR}/train_wan_i2v_blenderflowers_hibiscus_high.sh" &
CUDA_VISIBLE_DEVICES=3 bash "${SCRIPT_DIR}/train_wan_i2v_blenderflowers_lily_high.sh" &
CUDA_VISIBLE_DEVICES=4 bash "${SCRIPT_DIR}/train_wan_i2v_blenderflowers_rose_high.sh" &

# Low-noise LoRA (5 jobs; timesteps 0–900) — GPUs 5–7, then 0–1 (round-robin over 0–7)
CUDA_VISIBLE_DEVICES=5 bash "${SCRIPT_DIR}/train_wan_i2v_blenderflowers_dahlia_low.sh" &
CUDA_VISIBLE_DEVICES=6 bash "${SCRIPT_DIR}/train_wan_i2v_blenderflowers_daisy_low.sh" &
CUDA_VISIBLE_DEVICES=7 bash "${SCRIPT_DIR}/train_wan_i2v_blenderflowers_hibiscus_low.sh" &
CUDA_VISIBLE_DEVICES=0 bash "${SCRIPT_DIR}/train_wan_i2v_blenderflowers_lily_low.sh" &
CUDA_VISIBLE_DEVICES=1 bash "${SCRIPT_DIR}/train_wan_i2v_blenderflowers_rose_low.sh" &

wait
echo "All Blender flowers finetuning jobs (high + low noise) finished."

exit 0
}
