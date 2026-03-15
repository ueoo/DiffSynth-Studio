import os
import sys

import torch

from modelscope import dataset_snapshot_download
from PIL import Image
from prompt_bubble import negative_prompt, prompt
from tqdm import trange


os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# Put pretrained downloads under a shared folder (default would be ./models).
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/scr/yuegao/wan_models")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video


vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            **vram_config,
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
            **vram_config,
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            origin_file_pattern="Wan2.2_VAE.pth",
            **vram_config,
        ),
    ],
)

wan_root_path = "/viscam/data/two-phase-flow/wan_models/train"
num_samples = 5
lora_epoch = 49
pipe.load_lora(
    pipe.dit, f"{wan_root_path}/Wan2_2_TI2V_5B_lora_101frames_randomstart/epoch-{lora_epoch}.safetensors", alpha=1
)
dataset_path = "/viscam/data/two-phase-flow/bubble_boiling_100_uci_dataset_832x480"
input_image = VideoData(f"{dataset_path}/pair_001_cam1.mp4", height=480, width=832)[0]

for i in trange(num_samples, desc="Generating videos"):
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=input_image,
        num_frames=101,
        tiled=True,
    )
    save_video(
        video,
        f"wan_results/video_Wan2_2_TI2V_5B_lora_101frames_randomstart_epoch{lora_epoch}_bubble_{i:02d}.mp4",
        fps=15,
        quality=5,
    )
