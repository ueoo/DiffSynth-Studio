import os
import sys

import torch

from modelscope import dataset_snapshot_download
from PIL import Image
from prompt_bubble import negative_prompt, prompt


# Put pretrained downloads under a shared folder (default would be ./models).
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/viscam/data/two-phase-flow/wan_models")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


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
            model_id="Wan-AI/Wan2.2-T2V-A14B",
            origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
            **vram_config,
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.2-T2V-A14B",
            origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
            **vram_config,
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config
        ),
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024**3) - 2,
)

# Text-to-video
video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=0,
    tiled=True,
)
save_video(video, "wan_results/video_Wan2.2-T2V-A14B_bubble.mp4", fps=15, quality=5)
