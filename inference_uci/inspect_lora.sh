python inspect_lora_ckpt.py \
  --ckpt "/viscam/data/two-phase-flow/wan_models/train/Wan2.2-I2V-A14B_high_noise_lora/epoch-7.safetensors" \
  --max_show 50

python inspect_lora_ckpt.py \
  --ckpt "/viscam/data/two-phase-flow/wan_models/train/Wan2.2-I2V-A14B_low_noise_lora/epoch-1.safetensors" \
  --max_show 50
