import argparse

from collections import Counter
from typing import Dict, Iterable, List, Tuple

import torch

from diffsynth.core.loader.file import load_state_dict
from diffsynth.utils.lora.general import GeneralLoRALoader


def _top_prefixes(keys: Iterable[str], max_parts: int = 4) -> List[Tuple[str, int]]:
    """
    Return common dotted-prefixes for quick diagnosis, e.g.
    'base_model.model', 'pipe.dit', 'module', 'diffusion_model', etc.
    """
    c: Counter[str] = Counter()
    for k in keys:
        parts = k.split(".")
        for n in range(1, min(max_parts, len(parts)) + 1):
            c[".".join(parts[:n])] += 1
    return c.most_common(30)


def _summarize_keys(keys: List[str], max_show: int) -> None:
    print(f"Total keys: {len(keys)}")
    if not keys:
        return
    print("First few keys:")
    for k in keys[:max_show]:
        print(f"  - {k}")
    print("Common key prefixes (top):")
    for p, n in _top_prefixes(keys):
        print(f"  - {p}: {n}")

    key_str = "\n".join(keys[: min(len(keys), 5000)])
    patterns = [
        ("peft_base_model_prefix", "base_model.model."),
        ("ddp_module_prefix", "module."),
        ("diffusers_diffusion_model_prefix", "diffusion_model."),
        ("diffsynth_training_prefix_pipe_dit", "pipe.dit."),
        ("diffsynth_training_prefix_pipe_dit2", "pipe.dit2."),
        ("lora_A_default", "lora_A.default.weight"),
        ("lora_B_default", "lora_B.default.weight"),
        ("lora_A", "lora_A.weight"),
        ("lora_B", "lora_B.weight"),
        ("lora_up", "lora_up.weight"),
        ("lora_down", "lora_down.weight"),
        ("alpha", ".alpha"),
    ]
    print("Pattern presence (quick check):")
    for name, pat in patterns:
        print(f"  - {name}: {pat in key_str}")


def _target_layer_names(converted_keys: Iterable[str]) -> List[str]:
    # converted keys are like "<layer>.lora_A.weight" / "<layer>.lora_B.weight"
    layer_names = set()
    for k in converted_keys:
        if k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight"):
            layer_names.add(k.rsplit(".lora_", 1)[0])
    return sorted(layer_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to LoRA checkpoint (.safetensors)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load tensors onto (default: cpu)")
    parser.add_argument("--max_show", type=int, default=30, help="Max keys to print for each section")
    parser.add_argument(
        "--dump_targets",
        action="store_true",
        help="Also print inferred target layer names after conversion (can be long).",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    sd: Dict[str, torch.Tensor] = load_state_dict(args.ckpt, device=args.device)
    keys = sorted(sd.keys())

    print("\n== Raw checkpoint keys ==")
    _summarize_keys(keys, args.max_show)

    # Apply the same conversion used by BasePipeline.load_lora()
    loader = GeneralLoRALoader(device=args.device, torch_dtype=torch.float32)
    converted = loader.convert_state_dict(sd)
    ckeys = sorted(converted.keys())

    print("\n== After GeneralLoRALoader.convert_state_dict() ==")
    _summarize_keys(ckeys, args.max_show)

    target_layers = _target_layer_names(ckeys)
    print(f"\nInferred target layers (count): {len(target_layers)}")
    if args.dump_targets:
        for name in target_layers:
            print(f"  - {name}")
    else:
        for name in target_layers[: args.max_show]:
            print(f"  - {name}")
        if len(target_layers) > args.max_show:
            print(f"  ... ({len(target_layers) - args.max_show} more)")

    # Heuristic diagnosis
    print("\n== Heuristic diagnosis ==")
    if any(k.startswith("base_model.model.") for k in keys):
        print(
            "- Detected prefix 'base_model.model.' in saved keys (typical PEFT wrapper). "
            "If runtime patching expects keys without this prefix, it will patch 0 tensors."
        )
    if any(k.startswith("pipe.dit.") for k in keys):
        print(
            "- Detected prefix 'pipe.dit.' in saved keys. If you saved with '--remove_prefix_in_ckpt pipe.dit.' "
            "but still see this, your save config may not be applied as expected."
        )
    if not any(("lora_A" in k or "lora_B" in k or "lora_up" in k or "lora_down" in k) for k in keys):
        print(
            "- No obvious LoRA weight keys found (lora_A/lora_B or lora_up/lora_down). "
            "This file might not be a LoRA checkpoint."
        )
    print("Done.")


if __name__ == "__main__":
    main()
