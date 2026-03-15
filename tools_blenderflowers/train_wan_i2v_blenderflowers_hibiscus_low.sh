#!/usr/bin/env bash
# Finetune Wan 2.2 I2V *low-noise* LoRA on Blender flowers (hibiscus). Same data as
# train_wan_i2v_blenderflowers_hibiscus_high.sh but trains the low-noise model branch.
set -euo pipefail

{
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/scr/yuegao/wan_models}"

DATASET_BASE_PATH="${DATASET_BASE_PATH:-/scr/yuegao/TRELLIS_datasets/BlenderFlowers_Hibiscusmore10kfull_sub_merged}"
METADATA_PATH="${METADATA_PATH:-${DATASET_BASE_PATH}/metadata_wan.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-/scr/yuegao/wan_models/train_blenderflowers/hibiscus_low}"

accelerate launch train.py \
  --dataset_type blender_flowers \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${METADATA_PATH}" \
  --renders_subdir renders \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --frame_step 2 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 50 \
  --auto_resume \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.358 \
  --fp8_models "Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth"

exit 0
}
