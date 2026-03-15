python inspect_lora_match_runtime.py \
  --lora "/viscam/data/two-phase-flow/wan_models/train/Wan2.2-I2V-A14B_high_noise_lora/epoch-7.safetensors" \
  --origin_file_pattern "high_noise_model/diffusion_pytorch_model*.safetensors" \
  --device cuda \
  --max_show 50


python inspect_lora_match_runtime.py \
  --lora "/viscam/data/two-phase-flow/wan_models/train/Wan2.2-I2V-A14B_low_noise_lora/epoch-1.safetensors" \
  --origin_file_pattern "low_noise_model/diffusion_pytorch_model*.safetensors" \
  --device cuda \
  --max_show 50
