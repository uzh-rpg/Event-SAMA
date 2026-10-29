#!/bin/bash
#SBATCH --job-name=sama-ablation-patchrandom
#SBATCH --partition=normal
#SBATCH --account=uzh55
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=640G
#SBATCH --time=24:00:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/sama_ablation_patchrandom_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/sama_ablation_patchrandom_%j.err

# ABLATION: ViT patch_embedding initialised randomly, vs the baseline (main
# run) which averages RGB conv weights across the input dim. Event adapter
# and SAM2 decoder left at baseline defaults -- isolates only the
# patch-embedding init variable.
# Multi-node (4x4=16 GPU), FRESH start from Sa2VA-4B_SAMA pretrained weights.
# batch_size=3 from the start (learned from the 2x-OOM noadapter ablation --
# avoids repeating that crash cycle here).
# 1 epoch budget (not the main run's 4) -- bounded for the Aug 31 deadline,
# still a complete warmup+cosine-decay LR cycle. Same self-chaining pattern.

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
CONFIG=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs/sama_4b_event_v3_ablation_patchrandom.py
WEIGHTS_DIR=/iopsstor/scratch/cscs/sfedorchenko/weights
DATA_DIR=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset
EVENT_DIR_CAPSTOR=/capstor/scratch/cscs/mzehnde/sama_converted
EVENT_DIR_CAPSTOR_V2=/capstor/scratch/cscs/mzehnde/sama_converted_v2
EVENT_DIR_CAPSTOR_TRANSFER=/capstor/scratch/cscs/mzehnde/sama_converted_transfer
EVENT_DIR_CAPSTOR_REVOS=/capstor/scratch/cscs/mzehnde/sama_converted_revos
EVENT_DIR_IOPSSTOR=/iopsstor/scratch/cscs/mzehnde/sama_converted
RUNS_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs
WORK_DIR=${RUNS_DIR}/sama_4b_event_v4_ablation_patchrandom
SCRIPT_PATH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs/train_ablation_patchrandom.sbatch
MAX_ITERS=32711   # 1 epoch, batch_size=3, 16 GPUs (matches noadapter ablation's validated run)
# Self-contained arm64 image (NGC torch base + mmcv2.2.0); deps baked, no PYDEPS.
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event-sama-arm64.sqsh

mkdir -p ${WORK_DIR} ${RUNS_DIR}

echo "=== SAMA-4B Event V4 Training (patch-embed random-init ablation) ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "Config:                   ${CONFIG}"
echo "Work dir:                 ${WORK_DIR}"

# ── Chain control: stop when complete, otherwise queue a successor ────
parse_ckpt_iter() {
  # Echo the iteration number of the current last_checkpoint, or 0 if none.
  local f="${WORK_DIR}/last_checkpoint"
  [ -f "$f" ] || { echo 0; return; }
  local p; p=$(cat "$f")
  local n; n=$(basename "$p" | grep -oE '[0-9]+' | tail -1)
  echo "${n:-0}"
}

if [ -f "${WORK_DIR}/TRAINING_COMPLETE" ]; then
  echo "[chain] TRAINING_COMPLETE present; nothing to do. Exiting."
  exit 0
fi

CUR_ITER=$(parse_ckpt_iter)
if [ "${CUR_ITER}" -ge "${MAX_ITERS}" ]; then
  echo "[chain] reached MAX_ITERS (${CUR_ITER}/${MAX_ITERS}); marking complete."
  touch "${WORK_DIR}/TRAINING_COMPLETE"
  exit 0
fi

# Stall guard: if the previous job left the ckpt where the one before it did,
# count it as no-progress. Stop after 3 consecutive stalls (crash loop).
PREV_ITER=$(cat "${WORK_DIR}/.chain_prev_iter" 2>/dev/null || echo -1)
STALL=$(cat "${WORK_DIR}/.chain_stall" 2>/dev/null || echo 0)
if [ "${CUR_ITER}" -le "${PREV_ITER}" ]; then
  STALL=$((STALL + 1))
else
  STALL=0
fi
echo "${CUR_ITER}" > "${WORK_DIR}/.chain_prev_iter"
echo "${STALL}"    > "${WORK_DIR}/.chain_stall"
if [ "${STALL}" -ge 3 ]; then
  echo "[chain] 3 consecutive jobs made no progress (stall at iter ${CUR_ITER}); stopping chain."
  exit 1
fi

# Queue the successor now (runs after this job ends, for any reason) so the
# chain survives both 6h timeouts and node preemption.
NEXT_JID=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} ${SCRIPT_PATH} 2>/dev/null || true)
echo "[chain] cur_iter=${CUR_ITER} stall=${STALL} queued successor=${NEXT_JID:-<none>}"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"

SHM_DIR=/tmp/sama_shm_${SLURM_JOB_ID}
# /tmp is node-local: `mkdir` here only creates SHM_DIR on the batch script's
# own (head) node. With --nodes>1, srun launches the container on every
# allocated node, and enroot tries to bind-mount SHM_DIR on each of them --
# it must exist on all of them first, not just the head node.
srun --nodes=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 mkdir -p ${SHM_DIR}

srun \
  --container-image=${SQSH} \
  --container-mounts=${PROJECT_DIR}:${PROJECT_DIR},${WEIGHTS_DIR}:${WEIGHTS_DIR},${DATA_DIR}:${DATA_DIR},${EVENT_DIR_CAPSTOR}:${EVENT_DIR_CAPSTOR},${EVENT_DIR_CAPSTOR_V2}:${EVENT_DIR_CAPSTOR_V2},${EVENT_DIR_CAPSTOR_TRANSFER}:${EVENT_DIR_CAPSTOR_TRANSFER},${EVENT_DIR_CAPSTOR_REVOS}:${EVENT_DIR_CAPSTOR_REVOS},${EVENT_DIR_IOPSSTOR}:${EVENT_DIR_IOPSSTOR},${RUNS_DIR}:${RUNS_DIR},/tmp:/tmp,${SHM_DIR}:/dev/shm \
  --no-container-mount-home \
  bash -c '
    set -e
    export PYTHONPATH='"${PROJECT_DIR}"':$PYTHONPATH
    export TMPDIR=/tmp
    mkdir -p $TMPDIR

    export DISPLAY=""
    export MPLBACKEND=Agg
    export TRANSFORMERS_NO_FLASH_ATTN_2=1
    export TRANSFORMERS_CACHE=/tmp/transformers_cache
    # Redirect HOME + all HF caches off the host HOME. The HF dynamic-module
    # cache (transformers_modules, holds Sa2VAChatConfig code) defaults to
    # $HOME/.cache/huggingface/modules; a full HOME corrupts it and breaks
    # AutoTokenizer with "Unrecognized configuration class" at iter 0.
    export HOME=/tmp/hfhome
    export HF_HOME=/tmp/hfhome/huggingface
    export HF_MODULES_CACHE=/tmp/hfhome/huggingface/modules
    export HUGGINGFACE_HUB_CACHE=/tmp/hfhome/huggingface/hub
    mkdir -p $TRANSFORMERS_CACHE $HF_HOME $HF_MODULES_CACHE $HUGGINGFACE_HUB_CACHE

    export RANK=${SLURM_PROCID}
    export LOCAL_RANK=0
    export WORLD_SIZE=${SLURM_NTASKS}
    echo "[task ${SLURM_PROCID}/${SLURM_NTASKS}] host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  RANK=${RANK} LOCAL_RANK=${LOCAL_RANK} WORLD_SIZE=${WORLD_SIZE}"

    cd '"${PROJECT_DIR}"'

    DS_CONFIG=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs/deepspeed_zero2_event.json
    PINNED_TRAIN=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs/train_no_fa2_pinned.py
    LAST_CKPT_FILE='"${WORK_DIR}"'/last_checkpoint
    RESUME_ARGS=""
    if [ -f "$LAST_CKPT_FILE" ]; then
      LAST_CKPT=$(cat "$LAST_CKPT_FILE")
      if [ -d "$LAST_CKPT" ] || [ -f "$LAST_CKPT" ]; then
        RESUME_ARGS="--resume $LAST_CKPT"
        echo "[sbatch] resuming from $LAST_CKPT"
      else
        echo "[sbatch] last_checkpoint pointed at missing $LAST_CKPT; starting fresh"
      fi
    else
      echo "[sbatch] no prior checkpoint in '"${WORK_DIR}"'; starting fresh from pretrained_pth"
    fi
    python $PINNED_TRAIN '"${CONFIG}"' \
      --launcher pytorch \
      --deepspeed $DS_CONFIG \
      --work-dir='"${WORK_DIR}"' \
      $RESUME_ARGS \
      2>&1
  '
SRUN_RC=$?

# If training finished gracefully and reached the schedule, stop the chain.
END_ITER=$(parse_ckpt_iter)
echo "[chain] srun_rc=${SRUN_RC} end_iter=${END_ITER}/${MAX_ITERS}"
if [ "${SRUN_RC}" -eq 0 ] && [ "${END_ITER}" -ge "${MAX_ITERS}" ]; then
  echo "[chain] training complete at iter ${END_ITER}; writing TRAINING_COMPLETE."
  touch "${WORK_DIR}/TRAINING_COMPLETE"
fi
