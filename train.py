import argparse
import glob
import os
import re
import warnings

import accelerate
import torch

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import (
    ImageCropAndResize,
    LoadAudio,
    LoadBlenderFlowersFrames,
    LoadVideo,
    ToAbsolutePath,
)
from diffsynth.diffusion import *
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn(
                "Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing."
            )
            use_gradient_checkpointing = True

        # Load models
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        tokenizer_config = (
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
            if tokenizer_path is None
            else ModelConfig(tokenizer_path)
        )
        audio_processor_config = (
            ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/")
            if audio_processor_path is None
            else ModelConfig(audio_processor_path)
        )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(
                pipe, **inputs_shared, **inputs_posi
            ),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(
                pipe, **inputs_shared, **inputs_posi
            ),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(
                pipe, **inputs_shared, **inputs_posi
            ),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(
                pipe, **inputs_shared, **inputs_posi
            ),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument(
        "--audio_processor_path",
        type=str,
        default=None,
        help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.",
    )
    parser.add_argument(
        "--max_timestep_boundary",
        type=float,
        default=1.0,
        help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).",
    )
    parser.add_argument(
        "--min_timestep_boundary",
        type=float,
        default=0.0,
        help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).",
    )
    parser.add_argument(
        "--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU."
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="default",
        choices=("default", "blender_flowers"),
        help="Dataset type. blender_flowers: load from renders/ with random start, frame_step, random view; pad and resize to Wan I2V size.",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=2,
        help="Frame step when sampling from Blender flowers (e.g. 2 = every 2nd frame). Used only for dataset_type=blender_flowers.",
    )
    parser.add_argument(
        "--renders_subdir",
        type=str,
        default="renders",
        help="Subdir under dataset_base_path containing scene frame dirs. Used only for dataset_type=blender_flowers.",
    )
    return parser


def _find_latest_lora_checkpoint(output_path: str):
    """
    Find the latest LoRA checkpoint under output_path.
    Supported naming:
      - epoch-<N>.safetensors  (preferred if exists)
      - step-<N>.safetensors
    Returns: (kind, index, path) or (None, None, None)
    """
    if output_path is None:
        return None, None, None
    if not os.path.isdir(output_path):
        return None, None, None

    def collect(kind: str, regex: str):
        out = []
        for path in glob.glob(os.path.join(output_path, f"{kind}-*.safetensors")):
            base = os.path.basename(path)
            m = re.match(regex, base)
            if m is None:
                continue
            out.append((int(m.group(1)), path))
        if not out:
            return None, None
        out.sort(key=lambda x: x[0])
        return out[-1][0], out[-1][1]

    epoch_idx, epoch_path = collect("epoch", r"epoch-(\d+)\.safetensors$")
    if epoch_path is not None:
        return "epoch", epoch_idx, epoch_path
    step_idx, step_path = collect("step", r"step-(\d+)\.safetensors$")
    if step_path is not None:
        return "step", step_idx, step_path
    return None, None, None


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

    # Auto-resume from the latest LoRA checkpoint in output_path.
    # This is intentionally lightweight: it restores LoRA weights and
    # advances the epoch counter to avoid overwriting earlier checkpoints.
    if getattr(args, "auto_resume", False):
        kind, idx, path = _find_latest_lora_checkpoint(args.output_path)
        if path is not None:
            if args.lora_checkpoint is None:
                args.lora_checkpoint = path
            if kind == "epoch":
                args.start_epoch = max(getattr(args, "start_epoch", 0), idx + 1)
            elif kind == "step":
                args.start_step = max(getattr(args, "start_step", 0), idx)
            print(f"[auto_resume] found latest LoRA checkpoint: {path}")
            if kind == "epoch":
                print(f"[auto_resume] will start from epoch {args.start_epoch} (num_epochs={args.num_epochs})")
            else:
                print(f"[auto_resume] will set step counter to {args.start_step} (save_steps={args.save_steps})")

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    special_operator_map = {
        "animate_face_video": ToAbsolutePath(args.dataset_base_path)
        >> LoadVideo(
            args.num_frames,
            4,
            1,
            frame_processor=ImageCropAndResize(512, 512, None, 16, 16),
            random_start=getattr(args, "random_start", False),
        ),
        "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
    }

    if getattr(args, "dataset_type", "default") == "blender_flowers":
        # Blender flowers: use metadata_wan.csv. video column = scene_name or scene_name|reverse.
        # Loader: random view from transforms.json, random start when enough frames; frame_step (default 2);
        # if fewer than (num_frames-1)*frame_step+1 frames, start from first frame and pad by repeating last.
        special_operator_map["video"] = LoadBlenderFlowersFrames(
            base_path=args.dataset_base_path,
            renders_subdir=getattr(args, "renders_subdir", "renders"),
            num_frames=args.num_frames,
            frame_step=getattr(args, "frame_step", 2),
            random_start=True,
            random_view=True,
            target_height=args.height or 480,
            target_width=args.width or 832,
        )
        main_data_operator = UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            random_start=False,
        )
    else:
        main_data_operator = UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            random_start=getattr(args, "random_start", False),
        )

    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=main_data_operator,
        special_operator_map=special_operator_map,
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    # Ensure step-based checkpoint naming continues monotonically after resume.
    if getattr(args, "start_step", 0) > 0:
        model_logger.num_steps = args.start_step
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
