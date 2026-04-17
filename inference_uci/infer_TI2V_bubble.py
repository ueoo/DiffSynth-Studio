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
            model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config
        ),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024**3) - 2,
)

# Text-to-video
# video = pipe(
#     prompt="High-speed monochrome macro laboratory video of multiple boiling bubbles above a heated surface. Strong backlight illumination, sharp specular reflections, realistic fluid dynamics, static camera, shallow depth of field, no text, no watermark. Multiple bubbles nucleate from the bottom surface at far left side, left side, and left region, rise upward, wobble, occasionally interact or merge, and shed tiny satellite droplets.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     num_frames=121,
# )
# save_video(video, "video_1_Wan2.2-TI2V-5B.mp4", fps=15, quality=5)

dataset_path = "/viscam/data/two-phase-flow/bubble_boiling_100_uci_dataset_832x480"
input_image = VideoData(f"{dataset_path}/pair_001_cam1.mp4", height=480, width=832)[0]

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=0,
    tiled=True,
    height=704,
    width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "wan_results/video_Wan2.2-TI2V-5B_bubble.mp4", fps=15, quality=5)
