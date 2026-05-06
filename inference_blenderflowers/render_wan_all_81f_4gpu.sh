#!/usr/bin/env bash
# Dispatch the 10 81-frame inference jobs across 4 GPUs in waves of 4.
# Each Wan2.2-I2V-A14B inference uses ~50 GB VRAM, so one job per GPU at a time.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/scr/yuegao/4d_state_machine_all/wan2_2_i2v_results_finetune_81f/_logs}"
mkdir -p "${LOG_DIR}"

# Use the DiffSynth conda env (has torch, diffsynth, etc.); the per-script
# `python3` calls resolve via this PATH.
export PATH=/svl/u/yuegao/miniconda/envs/ds/bin:${PATH}
export PYTHONPATH=/svl/u/yuegao/4DStateMachine/DiffSynth-Studio:${PYTHONPATH:-}

# (GPU, script) tuples — 10 jobs across 4 GPUs in 3 waves
JOBS=(
  # wave 1
  "0 render_wan_i2v_rosefull_forward_81f.sh"
  "1 render_wan_i2v_rosefull_reverse_81f.sh"
  "2 render_wan_i2v_lilyfull_forward_81f.sh"
  "3 render_wan_i2v_lilyfull_reverse_81f.sh"
  # wave 2
  "0 render_wan_i2v_daisyfull_forward_81f.sh"
  "1 render_wan_i2v_daisyfull_reverse_81f.sh"
  "2 render_wan_i2v_dahliafull_forward_81f.sh"
  "3 render_wan_i2v_dahliafull_reverse_81f.sh"
  # wave 3
  "0 render_wan_i2v_hibiscusfull_forward_81f.sh"
  "1 render_wan_i2v_hibiscusfull_reverse_81f.sh"
)

run_wave () {
  local pids=()
  for entry in "$@"; do
    local gpu="${entry%% *}"
    local script="${entry#* }"
    local logf="${LOG_DIR}/${script%.sh}.log"
    echo "[wave] GPU $gpu  $script  -> $logf"
    CUDA_VISIBLE_DEVICES=$gpu bash "${SCRIPT_DIR}/${script}" > "$logf" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  echo "[wave done]"
}

echo "=== wave 1 (4 jobs, GPUs 0-3) ==="
run_wave "${JOBS[0]}" "${JOBS[1]}" "${JOBS[2]}" "${JOBS[3]}"
echo "=== wave 2 (4 jobs, GPUs 0-3) ==="
run_wave "${JOBS[4]}" "${JOBS[5]}" "${JOBS[6]}" "${JOBS[7]}"
echo "=== wave 3 (2 jobs, GPUs 0-1) ==="
run_wave "${JOBS[8]}" "${JOBS[9]}"

echo "=== ALL DONE ==="
echo "Outputs: /scr/yuegao/4d_state_machine_all/wan2_2_i2v_results_finetune_81f/"
echo "Per-job logs: ${LOG_DIR}/"
