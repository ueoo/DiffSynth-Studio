import os
import sys

import numpy as np
import torch

from PIL import Image
from prompt_bubble import negative_prompt, prompt
from tqdm import trange


os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# Put pretrained downloads under a shared folder (default would be ./models).
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/scr/yuegao/wan_models")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video


def load_clip(video_path: str, *, num_frames: int, height: int, width: int) -> list[Image.Image]:
    """Load a video and return exactly `num_frames` frames.

    If the source is shorter, duplicates the last frame.
    """

    video_data = VideoData(video_path, height=height, width=width)
    video_len = len(video_data)
    if video_len <= 0:
        raise ValueError(f"Input video has no frames: {video_path}")

    if video_len >= num_frames:
        video_data.set_length(num_frames)
        return [video_data[i] for i in range(num_frames)]

    frames = [video_data[i] for i in range(video_len)]
    frames.extend([frames[-1]] * (num_frames - video_len))
    return frames


def load_mask_clip(
    mask_path: str,
    *,
    num_frames: int,
    height: int,
    width: int,
    invert: bool = False,
    threshold: float | None = None,
) -> list[Image.Image]:
    """Load a mask video and return exactly `num_frames` L-mode frames (0..255).

    - `invert=True` flips the mask (foreground<->background)
    - `threshold` binarizes the mask (e.g. 0.5). If None, keeps soft values.
    """

    frames_rgb = load_clip(mask_path, num_frames=num_frames, height=height, width=width)
    out: list[Image.Image] = []
    for f in frames_rgb:
        m = f.convert("L")
        if invert:
            m = Image.fromarray(255 - np.array(m, dtype=np.uint8))
        if threshold is not None:
            arr = np.array(m, dtype=np.float32) / 255.0
            arr = (arr >= threshold).astype(np.uint8) * 255
            m = Image.fromarray(arr, mode="L")
        out.append(m)
    return out


def composite_foreground(
    *,
    generated: list[Image.Image],
    source: list[Image.Image],
    mask_l: list[Image.Image],
) -> list[Image.Image]:
    """(Deprecated) Post-composite helper. Kept for reference.

    This script now uses true mask-guided SDEdit inside the denoising loop.
    """

    if not (len(generated) == len(source) == len(mask_l)):
        raise ValueError(f"Length mismatch: gen={len(generated)}, src={len(source)}, mask={len(mask_l)}")

    out: list[Image.Image] = []
    for g, s, m in zip(generated, source, mask_l):
        g_np = np.array(g.convert("RGB"), dtype=np.float32)
        s_np = np.array(s.convert("RGB"), dtype=np.float32)
        m_np = np.array(m.convert("L"), dtype=np.float32) / 255.0
        m_np = m_np[..., None]  # H W 1
        c_np = g_np * m_np + s_np * (1.0 - m_np)
        c_np = np.clip(c_np, 0, 255).astype(np.uint8)
        out.append(Image.fromarray(c_np, mode="RGB"))
    return out


# NOTE: we reuse the T2V A14B weights
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="Wan-AI/Wan2.2-T2V-A14B",
            origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.2-T2V-A14B",
            origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.2-T2V-A14B",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.2-T2V-A14B",
            origin_file_pattern="Wan2.1_VAE.pth",
        ),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

# Load LoRA
wan_root_path = "/viscam/data/two-phase-flow/wan_models/train"
lora_epoch = 5
pipe.load_lora(
    pipe.dit,
    f"{wan_root_path}/Wan2_2_T2V_A14B_high_lora_101frames_randomstart/epoch-{lora_epoch}.safetensors",
    alpha=1,
)
pipe.load_lora(
    pipe.dit2,
    f"{wan_root_path}/Wan2_2_T2V_A14B_low_lora_101frames_randomstart/epoch-{lora_epoch}.safetensors",
    alpha=1,
)

# V2V SDEdit settings
num_frames = 101
height, width = 480, 832
sdedit_strength = 0.3

# Mask settings
mask_type = "gt_mask"  # e.g. gt_mask / alpha
invert_mask = False
mask_threshold = 0.5  # set None to keep soft mask

exp_logs_root = "/svl/data/two-phase-flow/yuegao/gaussian_tpf_logs/"
exp_part = "tpf_real_charuco_dyn_2dgs_sdf"
exp_folder = "uci_0923_2dgs_sdf_start00002dur100step1_cleanbubble_initym0d017_prunezn0d36n0d35_prunexn0d015p0d015_proj500_zerone2"
exp_path = os.path.join(exp_logs_root, exp_part, exp_folder)

# Input frames (render) and mask frames come from different folders in this experiment.
input_dir = "test_padwan_mp4"
mask_dir = "test_merged"

cam_names = ["camnoveldeg030"]
input_map_type = "render"

for cam_name in cam_names:
    input_video_path = f"{exp_path}/{input_dir}/{input_map_type}_{cam_name}.mp4"
    mask_video_path = f"{exp_path}/{mask_dir}/{mask_type}_{cam_name}.mp4"

    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Missing input video: {input_video_path}")
    if not os.path.exists(mask_video_path):
        raise FileNotFoundError(f"Missing mask video: {mask_video_path} (try changing cam_names or mask_type)")

    input_video = load_clip(input_video_path, num_frames=num_frames, height=height, width=width)
    mask_video_l = load_mask_clip(
        mask_video_path,
        num_frames=num_frames,
        height=height,
        width=width,
        invert=invert_mask,
        threshold=mask_threshold,
    )

    num_samples = 1
    for i in trange(num_samples, desc="SDEdit-ing videos (mask-guided, foreground only)"):
        gen_video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_video=input_video,
            denoising_strength=sdedit_strength,
            sdedit_mask_video=mask_video_l,
            num_frames=num_frames,
            height=height,
            width=width,
            tiled=True,
        )

        save_video(
            gen_video,
            f"wan_results/video_Wan2_2_V2V_A14B_{cam_name}_sdedit_strength{sdedit_strength}_lora_epoch{lora_epoch}_bubble_sdeditmask_{i:02d}.mp4",
            fps=15,
            quality=5,
        )
