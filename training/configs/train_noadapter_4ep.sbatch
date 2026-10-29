#!/bin/bash
#SBATCH --job-name=sama-noadapter-4ep
#SBATCH --partition=normal
#SBATCH --account=uzh55
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=640G
#SBATCH --time=24:00:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/sama_noadapter_4ep_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/sama_noadapter_4ep_%j.err

# FULL production run, same setup as the 4 ablations: multi-node (4x4=16
# GPU), batch_size=3 (the OOM-safe value the ablations settled on),
# baseline defaults (event adapter ON, SAM2 trainable, RGB-mean
# patch-embed init, ViT frozen except patch_embedding) -- i.e. no ablation
# flag flipped, full unified data mix, full 4-epoch schedule.
#
# Launched fresh (new WORK_DIR, starts from pretrained_pth) rather than
# resuming the single-node main run's iter_285000 checkpoint: the
# single-node run's different effective batch size (16 vs this run's 48)
# and different LR-schedule shape (tied to a 392528-iter plan vs this
# run's 130844-iter plan) made its checkpoint an apples-to-oranges
# comparison against the multi-node ablations. This run uses the exact
# same recipe as the ablations, so it becomes a real "no ablation" baseline
# once it's done -- and needs to be re-evaluated as such (the single-node
# run's numbers are not a substitute).
#
# MAX_ITERS corrected 2026-08-22: the actual mmengine-computed total for
# this config (batch_size=3, 16 GPUs, this exact data mix) is 127468, not
# 130844 -- confirmed live from the training log ("Iter(train)
# [.../127468]"). The old 130844 value meant the chain's completion check
# (CUR_ITER >= MAX_ITERS) would never fire once training genuinely
# finished at 127468: the next chained job would resume, correctly do
# zero more iterations, and after 3 such no-op resumes the stall guard
# would kill the chain WITHOUT ever writing TRAINING_COMPLETE -- exactly
# what silently happened to the single-node run (stalled at 382416/399492
# real-vs-configured, chain gave up at job 4522485 despite training
# actually being 100% done).
# Same self-chaining pattern as everything else this session.

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
CONFIG=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs/sama_4b_event_v3_noadapter_4ep.py
WEIGHTS_DIR=/iopsstor/scratch/cscs/sfedorchenko/weights
DATA_DIR=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset
EVENT_DIR_CAPSTOR=/capstor/scratch/cscs/mzehnde/sama_converted
EVENT_DIR_CAPSTOR_V2=/capstor/scratch/cscs/mzehnde/sama_converted_v2
EVENT_DIR_CAPSTOR_TRANSFER=/capstor/scratch/cscs/mzehnde/sama_converted_transfer
EVENT_DIR_CAPSTOR_REVOS=/capstor/scratch/cscs/mzehnde/sama_converted_revos
EVENT_DIR_IOPSSTOR=/iopsstor/scratch/cscs/mzehnde/sama_converted
RUNS_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs
WORK_DIR=${RUNS_DIR}/sama_4b_event_v4_noadapter_4ep
SCRIPT_PATH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs/train_noadapter_4ep.sbatch
MAX_ITERS=127468   # 4 epochs, batch_size=3, 16 GPUs; real mmengine-computed total, confirmed from live log
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event-sama-arm64.sqsh

mkdir -p ${WORK_DIR} ${RUNS_DIR}

echo "=== SAMA-4B Event V4 Training (full production run, multi-node, same setup as ablations) ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "Config:                   ${CONFIG}"
echo "Work dir:                 ${WORK_DIR}"

# ── Chain control: stop when complete, otherwise queue a successor ────
parse_ckpt_iter() {
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

END_ITER=$(parse_ckpt_iter)
echo "[chain] srun_rc=${SRUN_RC} end_iter=${END_ITER}/${MAX_ITERS}"
if [ "${SRUN_RC}" -eq 0 ] && [ "${END_ITER}" -ge "${MAX_ITERS}" ]; then
  echo "[chain] training complete at iter ${END_ITER}; writing TRAINING_COMPLETE."
  touch "${WORK_DIR}/TRAINING_COMPLETE"
fi
