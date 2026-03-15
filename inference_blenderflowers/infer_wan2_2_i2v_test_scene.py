import argparse
import glob
import os
import re
import sys

import torch

from PIL import Image, ImageOps


# Put pretrained downloads under a shared folder (default would be ./models).
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/scr/yuegao/wan_models")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


SUBSET_CONFIGS = {
    "rosefull": {
        "test_data_root": "/scr/yuegao/TRELLIS_datasets/BlenderFlowers_Rosemore50kfullTest_merged",
    },
    "lilyfull": {
        "test_data_root": "/scr/yuegao/TRELLIS_datasets/BlenderFlowers_Lilymore20kfullTest_merged",
    },
    "dahliafull": {
        "test_data_root": "/scr/yuegao/TRELLIS_datasets/BlenderFlowers_Dahliamorefix20kfullTest_merged",
    },
    "daisyfull": {
        "test_data_root": "/scr/yuegao/TRELLIS_datasets/BlenderFlowers_Daisymorefix20kfullTest_merged",
    },
    "hibiscusfull": {
        "test_data_root": "/scr/yuegao/TRELLIS_datasets/BlenderFlowers_Hibiscusmore10kfullTest_merged",
    },
}

FLOWER_NAME_BY_PREFIX = {
    "rose": "rose",
    "daisy": "daisy",
    "dahlia": "dahlia",
    "hibiscus": "hibiscus",
    "lily": "lily",
}


WAN_INPUT_WIDTH = 832
WAN_INPUT_HEIGHT = 480
WAN_ASPECT_RATIO = WAN_INPUT_WIDTH / WAN_INPUT_HEIGHT


def resolve_input_path(test_data_root: str, scene_name: str, frame_idx: int) -> str:
    candidates = [
        os.path.join(test_data_root, "renders", f"{scene_name}_{frame_idx:03d}", "mv.png"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No input image found for scene={scene_name} frame={frame_idx:03d}. Tried: {candidates}")


def infer_flower_name(scene_name: str, subset: str) -> str:
    scene_lower = scene_name.lower()
    for prefix, flower_name in FLOWER_NAME_BY_PREFIX.items():
        if scene_lower.startswith(prefix):
            return flower_name
    subset_lower = subset.lower()
    for prefix, flower_name in FLOWER_NAME_BY_PREFIX.items():
        if prefix in subset_lower:
            return flower_name
    return "unknown"


def save_video_frames(frames, frames_dir: str):
    os.makedirs(frames_dir, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame.save(os.path.join(frames_dir, f"{idx:03d}.png"))


def _resolve_lora_path(path: str, epoch: int = None):
    """
    Resolve path to a single LoRA .safetensors file.
    If path is a file, return it (if .safetensors). If path is a directory:
      - If epoch is set, use epoch-{epoch}.safetensors if it exists.
      - Else find latest epoch-*.safetensors or step-*.safetensors (same as train).
    Otherwise return None.
    """
    if not path or not os.path.exists(path):
        return None
    if os.path.isfile(path):
        return path if path.endswith(".safetensors") else None

    if epoch is not None:
        candidate = os.path.join(path, f"epoch-{epoch}.safetensors")
        if os.path.isfile(candidate):
            return candidate
        return None

    def collect(kind: str, regex: str):
        out = []
        for p in glob.glob(os.path.join(path, f"{kind}-*.safetensors")):
            base = os.path.basename(p)
            m = re.match(regex, base)
            if m is None:
                continue
            out.append((int(m.group(1)), p))
        if not out:
            return None, None
        out.sort(key=lambda x: x[0])
        return out[-1][0], out[-1][1]

    _, epoch_path = collect("epoch", r"epoch-(\d+)\.safetensors$")
    if epoch_path is not None:
        return epoch_path
    _, step_path = collect("step", r"step-(\d+)\.safetensors$")
    if step_path is not None:
        return step_path
    return None


def pad_width_to_wan_ratio(input_image: Image.Image) -> Image.Image:
    """
    Pad image width to WAN's default I2V aspect ratio (832:480 = 26:15).
    """
    width, height = input_image.size
    target_width = int(round(height * WAN_ASPECT_RATIO))
    if width >= target_width:
        return input_image
    pad_total = target_width - width
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return ImageOps.expand(input_image, border=(pad_left, 0, pad_right, 0), fill=0)


def build_pipeline() -> WanVideoPipeline:
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

    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Wan-AI/Wan2.2-I2V-A14B",
                origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
                **vram_config,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-I2V-A14B",
                origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
                **vram_config,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-I2V-A14B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                **vram_config,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-I2V-A14B",
                origin_file_pattern="Wan2.1_VAE.pth",
                **vram_config,
            ),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024**3) - 2,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument(
        "--subset",
        type=str,
        default="rosefull",
        choices=sorted(SUBSET_CONFIGS.keys()),
        help="Subset config used to resolve default test_data_root.",
    )
    parser.add_argument("--reverse", action="store_true", default=False)
    parser.add_argument("--test_data_root", type=str, default=None)
    parser.add_argument("--test_scene_name", type=str, required=True)
    parser.add_argument("--frame_idx", type=int, required=True)
    parser.add_argument("--flower_name", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="/scr/yuegao/wan2_2_i2v")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--switch_dit_boundary", type=float, default=0.9)
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="blurry, overexposed, low quality, artifacts, text, watermark, distorted shapes",
    )
    parser.add_argument(
        "--lora_high_noise_path",
        type=str,
        default=None,
        help="Path to high-noise LoRA checkpoint (file or dir with epoch-*.safetensors / step-*.safetensors).",
    )
    parser.add_argument(
        "--lora_low_noise_path",
        type=str,
        default=None,
        help="Path to low-noise LoRA checkpoint (file or dir with epoch-*.safetensors / step-*.safetensors).",
    )
    parser.add_argument(
        "--lora_epoch",
        type=int,
        default=None,
        help="Load epoch-N.safetensors from each LoRA dir. If set, overrides 'latest' resolution.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)

    subset_cfg = SUBSET_CONFIGS[args.subset]
    test_data_root = args.test_data_root or subset_cfg["test_data_root"]
    input_image_path = resolve_input_path(test_data_root, args.test_scene_name, args.frame_idx)
    input_image = Image.open(input_image_path).convert("RGB")
    input_image = pad_width_to_wan_ratio(input_image)
    flower_name = args.flower_name or infer_flower_name(args.test_scene_name, args.subset)
    prompt = args.prompt or f"a {flower_name} flower growing"
    if args.reverse:
        prompt = f"a {flower_name} flower reverse growing"

    pipe = build_pipeline()

    high_lora = (
        _resolve_lora_path(args.lora_high_noise_path, epoch=args.lora_epoch)
        if args.lora_high_noise_path
        else None
    )
    low_lora = (
        _resolve_lora_path(args.lora_low_noise_path, epoch=args.lora_epoch)
        if args.lora_low_noise_path
        else None
    )
    if high_lora:
        print(f"Loaded high-noise LoRA checkpoint: {high_lora}")
        pipe.load_lora(pipe.dit, high_lora)
    elif args.lora_high_noise_path:
        print(f"WARNING: No high-noise LoRA checkpoint found at {args.lora_high_noise_path}")
    if low_lora:
        print(f"Loaded low-noise LoRA checkpoint: {low_lora}")
        pipe.load_lora(pipe.dit2, low_lora)
    elif args.lora_low_noise_path:
        print(f"WARNING: No low-noise LoRA checkpoint found at {args.lora_low_noise_path}")

    video = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        input_image=input_image,
        num_frames=args.num_frames,
        seed=args.seed,
        tiled=True,
        switch_DiT_boundary=args.switch_dit_boundary,
    )

    sample_name = f"{args.test_scene_name}_{args.frame_idx:03d}"
    out_dir = os.path.join(args.output_root, sample_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "video.mp4")
    input_frame_out_path = os.path.join(out_dir, "input_frame.png")
    frames_out_dir = os.path.join(out_dir, "images")

    save_video(video, out_path, fps=15, quality=5)
    save_video_frames(video, frames_out_dir)
    input_image.save(input_frame_out_path)

    print(f"Input image: {input_image_path}")
    print(f"Prompt: {prompt}")
    print(f"Saved video: {out_path}")
    print(f"Saved input frame: {input_frame_out_path}")
    print(f"Saved generated frames: {frames_out_dir}")


if __name__ == "__main__":
    main()
