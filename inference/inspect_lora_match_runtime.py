import argparse

from typing import Dict, Iterable, List, Set, Tuple

import torch

from diffsynth.core.loader.file import load_state_dict
from diffsynth.core.vram.layers import AutoWrappedLinear
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.lora.general import GeneralLoRALoader


def _lora_layer_names_from_ckpt(ckpt_path: str, device: str = "cpu") -> Set[str]:
    """
    Return layer names inferred from a LoRA checkpoint, after the same conversion
    pipeline used by BasePipeline.load_lora().
    """
    sd: Dict[str, torch.Tensor] = load_state_dict(ckpt_path, device=device)
    loader = GeneralLoRALoader(device=device, torch_dtype=torch.float32)
    converted = loader.convert_state_dict(sd)
    names = set()
    for k in converted.keys():
        if k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight"):
            names.add(k.rsplit(".lora_", 1)[0])
    return names


def _autowrapped_linear_names(model: torch.nn.Module) -> List[str]:
    names: List[str] = []
    for _, m in model.named_modules():
        if isinstance(m, AutoWrappedLinear):
            names.append(m.name)
    return names


def _print_examples(title: str, items: Iterable[str], max_show: int = 30) -> None:
    items = list(items)
    print(f"{title} (count={len(items)}):")
    for s in items[:max_show]:
        print(f"  - {s}")
    if len(items) > max_show:
        print(f"  ... ({len(items) - max_show} more)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=str, required=True, help="Path to LoRA ckpt (.safetensors)")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.2-I2V-A14B",
        help="Base model id for constructing WanVideoPipeline",
    )
    parser.add_argument(
        "--origin_file_pattern",
        type=str,
        default="high_noise_model/diffusion_pytorch_model*.safetensors",
        help="Which diffusion model to load (high_noise_model or low_noise_model pattern).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for pipeline (default cuda)")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", help="bfloat16/float16/float32")
    parser.add_argument("--max_show", type=int, default=30)
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.torch_dtype.lower(), torch.bfloat16)

    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch_dtype,
        "onload_device": "cpu",
        "preparing_dtype": torch_dtype,
        "preparing_device": args.device,
        "computation_dtype": torch_dtype,
        "computation_device": args.device,
    }

    print("Loading LoRA and extracting layer names...")
    lora_layer_names = _lora_layer_names_from_ckpt(args.lora, device="cpu")
    print(f"LoRA layers inferred from ckpt: {len(lora_layer_names)}")
    _print_examples("LoRA layer name examples", sorted(lora_layer_names), max_show=args.max_show)

    print("\nLoading WanVideoPipeline and collecting AutoWrappedLinear names (this can take a bit)...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=args.device,
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern=args.origin_file_pattern, **vram_config),
            # tokenizer/text encoder/vae are not needed for name matching, but pipeline construction expects a set
            ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
            ModelConfig(model_id=args.model_id, origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
        ],
    )

    if pipe.dit is None:
        raise RuntimeError("pipe.dit is None; cannot inspect runtime layer names.")

    linear_names = _autowrapped_linear_names(pipe.dit)
    linear_names_set = set(linear_names)
    print(f"AutoWrappedLinear modules found in pipe.dit: {len(linear_names)}")
    _print_examples("AutoWrappedLinear.name examples", linear_names, max_show=args.max_show)

    # Compare direct matches
    direct_matches = sorted(lora_layer_names.intersection(linear_names_set))
    print(f"\nDirect LoRA-layer-name matches: {len(direct_matches)}")
    _print_examples("Direct match examples", direct_matches, max_show=args.max_show)

    # Try a simple normalization: drop inner '.module.' segments (common when LoRA was saved under wrapper modules)
    def drop_inner_module(name: str) -> str:
        return name.replace(".module.", ".")

    lora_norm = set(drop_inner_module(n) for n in lora_layer_names)
    linear_norm = set(drop_inner_module(n) for n in linear_names_set)
    norm_matches = sorted(lora_norm.intersection(linear_norm))
    print(f"\nMatches after normalizing '.module.' segments away: {len(norm_matches)}")
    _print_examples("Normalized match examples", norm_matches, max_show=args.max_show)

    # Show top missing patterns if nothing matches
    if len(direct_matches) == 0 and len(norm_matches) == 0:
        sample_lora = next(iter(sorted(lora_layer_names))) if lora_layer_names else None
        sample_linear = next(iter(sorted(linear_names_set))) if linear_names_set else None
        print("\nNo matches found. Sample names:")
        print(f"  - sample LoRA layer:   {sample_lora}")
        print(f"  - sample Linear layer: {sample_linear}")


if __name__ == "__main__":
    main()
