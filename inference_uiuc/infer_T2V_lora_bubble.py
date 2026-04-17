import argparse
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm

os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/scr/yuegao/wan_models")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def build_pipeline():
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
                model_id="Wan-AI/Wan2.2-T2V-A14B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                **vram_config,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-T2V-A14B",
                origin_file_pattern="Wan2.1_VAE.pth",
                **vram_config,
            ),
        ],
        tokenizer_config=ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        ),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024**3) - 2,
    )
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--lora_epoch", type=int, default=5)
    parser.add_argument(
        "--lora_root",
        type=str,
        default="/viscam/data/two-phase-flow/wan_models/train",
    )
    parser.add_argument(
        "--caption_csv",
        type=str,
        default="/scr/yuegao/two_phase_flow_data/uiuc_boiling_48_dataset_832x480/uiuc_captions.csv",
    )
    parser.add_argument("--output_dir", type=str, default="wan_results_uiuc")
    parser.add_argument("--num_frames", type=int, default=101)
    parser.add_argument("--num_prompts", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.makedirs(args.output_dir, exist_ok=True)

    pipe = build_pipeline()

    pipe.load_lora(
        pipe.dit,
        f"{args.lora_root}/Wan2_2_T2V_A14B_high_lora_UIUC_101frames_randomstart/epoch-{args.lora_epoch}.safetensors",
        alpha=1,
    )
    pipe.load_lora(
        pipe.dit2,
        f"{args.lora_root}/Wan2_2_T2V_A14B_low_lora_UIUC_101frames_randomstart/epoch-{args.lora_epoch}.safetensors",
        alpha=1,
    )

    df = pd.read_csv(args.caption_csv)
    df = df.head(args.num_prompts)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Views"):
        view_idx = int(row["view_idx"])
        prompt = row["prompt"]

        for sample_idx in range(args.num_samples):
            out_path = os.path.join(
                args.output_dir,
                f"video_T2V_A14B_UIUC_view{view_idx:03d}_lora_epoch{args.lora_epoch}_{sample_idx:02d}.mp4",
            )
            if os.path.exists(out_path):
                print(f"Skipping {out_path} (exists)")
                continue

            video = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_frames=args.num_frames,
                seed=args.seed + sample_idx,
                tiled=True,
            )
            save_video(video, out_path, fps=args.fps, quality=5)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
