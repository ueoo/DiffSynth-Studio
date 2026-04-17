import os
import subprocess

from argparse import ArgumentParser
from multiprocessing import Pool

import torch


def _run(cmd: str) -> int:
    print(cmd, flush=True)
    return subprocess.call(cmd, shell=True)


def parse_gpu_ids(gpu_ids: str | None, *, requested_gpu_num: int | None) -> list[int]:
    available = max(1, torch.cuda.device_count())

    if gpu_ids is not None and gpu_ids.strip() != "":
        ids = [int(x) for x in gpu_ids.split(",") if x.strip() != ""]
        if len(ids) == 0:
            raise ValueError("--gpu_ids was provided but parsed empty")
        return ids

    if requested_gpu_num is None:
        requested_gpu_num = available

    requested_gpu_num = max(1, min(requested_gpu_num, available))
    return list(range(requested_gpu_num))


def launch(args):
    gpu_ids = parse_gpu_ids(args.gpu_ids, requested_gpu_num=args.gpu_num)
    gpu_num = len(gpu_ids)

    cmds: list[str] = []
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        env = f"CUDA_VISIBLE_DEVICES={gpu_id}"
        cmd = (
            f"{env} python inference/infer_V2V_sdedit_lora_bubble_render_single.py"
            f" --rank {args.rank} --world_size {args.world_size}"
            f" --gpu_idx {gpu_idx} --gpu_num {gpu_num}"
            f" --skip_existing --lora_epoch {args.lora_epoch}"
        )
        cmds.append(cmd)

    with Pool(gpu_num) as pool:
        rets = pool.map(_run, cmds)

    failed = [i for i, r in enumerate(rets) if r != 0]
    if len(failed) > 0:
        raise SystemExit(f"Some workers failed: {failed} (exit codes: {rets})")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--lora_epoch", type=int, default=10)

    parser.add_argument("--gpu_ids", type=str, default=None, help='Comma-separated GPU ids, e.g. "4,5"')
    parser.add_argument("--gpu_num", type=int, default=None, help="Number of GPUs to use if --gpu_ids not set")

    args = parser.parse_args()
    launch(args)
