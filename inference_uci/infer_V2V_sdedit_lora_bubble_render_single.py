import argparse
import os
import sys

import torch

from prompt_bubble import negative_prompt, prompt
from tqdm import trange


os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/scr/yuegao/wan_models")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video


def load_clip(video_path: str, *, num_frames: int, height: int, width: int):
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


def format_strength(x: float) -> str:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s if s != "" else "0"


def main(args):
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

    wan_root_path = "/scr/yuegao/wan_models/train"
    lora_epoch = args.lora_epoch
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

    num_frames = args.num_frames
    height, width = args.height, args.width
    if args.sdedit_strengths is not None and len(args.sdedit_strengths) > 0:
        sdedit_strengths = args.sdedit_strengths
    elif args.sdedit_strength is not None:
        sdedit_strengths = [args.sdedit_strength]
    else:
        sdedit_strengths = [0.5]

    exp_logs_root = "/svl/data/two-phase-flow/yuegao/gaussian_tpf_logs"
    exp_part = "tpf_real_charuco_dyn_2dgs_icml_single"
    merged_folder = "uci_0923_2dgs_start00002_to_start00102dur1step1_icml_rebut_merged"
    padwan_dir = os.path.join(exp_logs_root, exp_part, merged_folder, "test_renders_padwan")

    cam_names = ["cam2"]
    # for deg in range(30, 360, 30):
    #     cam_names.append(f"camnoveldeg{deg:03d}")

    if args.world_size > 1:
        start = len(cam_names) * args.rank // args.world_size
        end = len(cam_names) * (args.rank + 1) // args.world_size
        cam_names = cam_names[start:end]

    cam_names = cam_names[args.gpu_idx :: args.gpu_num]

    out_root = "/svl/data/two-phase-flow/yuegao/wan_results"

    for cam_name in cam_names:
        video_path = os.path.join(padwan_dir, f"{cam_name}.mp4")
        input_video = load_clip(video_path, num_frames=num_frames, height=height, width=width)

        num_samples = args.num_samples
        for sdedit_strength in sdedit_strengths:
            strength_tag = format_strength(sdedit_strength)
            for i in trange(num_samples, desc=f"GPU {args.gpu_idx}: {cam_name} (sdedit={strength_tag})"):
                out_path = os.path.join(
                    out_root,
                    f"video_Wan2_2_V2V_A14B_single_{cam_name}_sdedit_strength{strength_tag}_"
                    f"lora_epoch{lora_epoch}_bubble_{i:02d}.mp4",
                )
                if args.skip_existing and os.path.exists(out_path):
                    continue
                video = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    input_video=input_video,
                    denoising_strength=sdedit_strength,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    tiled=True,
                )
                save_video(
                    video,
                    out_path,
                    fps=15,
                    quality=5,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)

    parser.add_argument("--num_frames", type=int, default=101)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--sdedit_strength", type=float, default=None)
    parser.add_argument("--sdedit_strengths", type=float, nargs="*", default=None)
    parser.add_argument("--lora_epoch", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output mp4 already exists")

    args = parser.parse_args()
    main(args)
