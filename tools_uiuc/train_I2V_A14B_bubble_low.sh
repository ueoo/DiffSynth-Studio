{
export CUDA_VISIBLE_DEVICES=1
export DIFFSYNTH_MODEL_BASE_PATH="/scr/yuegao/wan_models"

accelerate launch train.py \
  --dataset_base_path /scr/yuegao/two_phase_flow_data/uiuc_boiling_48_dataset_832x480 \
  --dataset_metadata_path /scr/yuegao/two_phase_flow_data/uiuc_boiling_48_dataset_832x480/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 101 \
  --random_start \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 50 \
  --auto_resume \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/viscam/data/two-phase-flow/wan_models/train/Wan2_2_I2V_A14B_low_lora_UIUC_101frames_randomstart" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.358 \
  --fp8_models "Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth"
# boundary corresponds to timesteps [0, 900)
exit 0
}
