#!/bin/bash
#SBATCH --job-name=sama-event-v3-dbg
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=460G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/sama_event_v3_dbg_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/sama_event_v3_dbg_%j.err

# Event SAMA V3: same bulletproof 4-way DDP / GPU-isolation layout as v2
# (see train_sama_event_v2.sbatch for the full rationale). Only the CONFIG
# and WORK_DIR change: v3 trains on the full converted event mix
# (mevis + lvvis + chatunivi + refcoco/+/g + revos-uvo + refytvos + sav +
#  flickr-gcg + grandf-gcg + llava-qa).

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
CONFIG=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs/sama_4b_event_v3.py
WEIGHTS_DIR=/iopsstor/scratch/cscs/sfedorchenko/weights
DATA_DIR=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset
EVENT_DIR_CAPSTOR=/capstor/scratch/cscs/mzehnde/sama_converted
EVENT_DIR_IOPSSTOR=/iopsstor/scratch/cscs/mzehnde/sama_converted
RUNS_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs
# Throwaway work dir so this 30-min smoke can't pollute the real run's ckpts.
WORK_DIR=${RUNS_DIR}/sama_4b_event_v3_debugtest
# Self-contained arm64 image (NGC torch base + mmcv2.2.0); deps baked, no PYDEPS.
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event-sama-arm64.sqsh

mkdir -p ${WORK_DIR} ${RUNS_DIR}

echo "=== SAMA-4B Event V3 Training (ALL converted event sources) ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "Config:                   ${CONFIG}"
echo "Work dir:                 ${WORK_DIR}"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"

SHM_DIR=/tmp/sama_shm_${SLURM_JOB_ID}
mkdir -p ${SHM_DIR}

srun \
  --container-image=${SQSH} \
  --container-mounts=${PROJECT_DIR}:${PROJECT_DIR},${WEIGHTS_DIR}:${WEIGHTS_DIR},${DATA_DIR}:${DATA_DIR},${EVENT_DIR_CAPSTOR}:${EVENT_DIR_CAPSTOR},${EVENT_DIR_IOPSSTOR}:${EVENT_DIR_IOPSSTOR},${RUNS_DIR}:${RUNS_DIR},/tmp:/tmp,${SHM_DIR}:/dev/shm \
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
    mkdir -p $TRANSFORMERS_CACHE

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
      echo "[sbatch] no prior checkpoint in '"${WORK_DIR}"'; starting fresh"
    fi
    python $PINNED_TRAIN '"${CONFIG}"' \
      --launcher pytorch \
      --deepspeed $DS_CONFIG \
      --work-dir='"${WORK_DIR}"' \
      $RESUME_ARGS \
      2>&1
  '
