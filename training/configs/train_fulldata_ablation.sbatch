#!/bin/bash
#SBATCH --job-name=sama-abl-fd-TAGPLACEHOLDER
#SBATCH --partition=normal
#SBATCH --account=uzh55
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=640G
#SBATCH --time=24:00:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/sama_abl_fd_%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/sama_abl_fd_%x_%j.err

# Generic full-corpus (1-epoch) architectural ablation, self-chaining.
# Usage: sbatch --job-name=sama-abl-fd-<tag> train_fulldata_ablation.sbatch <tag> <config_basename> [extra_env]
#   <tag>            short name, e.g. noadapter / sam2frozen / vitunfreeze / patchrandom / nopolarity
#   <config_basename> e.g. sama_4b_event_v3_fulldata_noadapter.py
#   [extra_env]      optional VAR=value to export before training (nopolarity needs EVENT_MERGE_POLARITY=1;
#                     must be set here, NOT inside the .py config -- mmengine deep-copies the config
#                     namespace and a bound module object crashes with "cannot pickle 'module' object").
#
# MAX_ITERS is NOT hardcoded: 1-epoch total iterations for the full corpus
# wasn't known in advance, so this chain scrapes the true total from the
# training log's own "Iter(train) [cur/total]" line the first time it
# appears, caches it in WORK_DIR/.max_iters, and uses that for all
# subsequent completion checks. Job 1 (no prior log) simply can't know the
# total yet, which is fine: it can't be complete on iteration 1 anyway.

TAG=$1
CONFIG_NAME=$2
EXTRA_ENV=$3
if [ -z "$TAG" ] || [ -z "$CONFIG_NAME" ]; then
  echo "Usage: sbatch --job-name=sama-abl-fd-<tag> train_fulldata_ablation.sbatch <tag> <config_basename> [extra_env]"
  exit 1
fi

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
CFG_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs
CONFIG=${CFG_DIR}/${CONFIG_NAME}
WEIGHTS_DIR=/iopsstor/scratch/cscs/sfedorchenko/weights
DATA_DIR=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset
EVENT_DIR_CAPSTOR=/capstor/scratch/cscs/mzehnde/sama_converted
EVENT_DIR_CAPSTOR_V2=/capstor/scratch/cscs/mzehnde/sama_converted_v2
EVENT_DIR_CAPSTOR_TRANSFER=/capstor/scratch/cscs/mzehnde/sama_converted_transfer
EVENT_DIR_CAPSTOR_REVOS=/capstor/scratch/cscs/mzehnde/sama_converted_revos
EVENT_DIR_IOPSSTOR=/iopsstor/scratch/cscs/mzehnde/sama_converted
RUNS_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs
WORK_DIR=${RUNS_DIR}/sama_4b_event_v4_fulldata_${TAG}
SCRIPT_PATH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs/train_fulldata_ablation.sbatch
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event-sama-arm64.sqsh
JOB_NAME=sama-abl-fd-${TAG}

mkdir -p ${WORK_DIR} ${RUNS_DIR}

echo "=== Full-corpus 1-epoch ablation: ${TAG} ==="
echo "Job: ${SLURM_JOB_ID}  Name: ${JOB_NAME}  Node: $(hostname)  Date: $(date)"
echo "Config:   ${CONFIG}"
echo "Work dir: ${WORK_DIR}"
echo "Extra env: ${EXTRA_ENV:-<none>}"

parse_ckpt_iter() {
  local f="${WORK_DIR}/last_checkpoint"
  [ -f "$f" ] || { echo 0; return; }
  local p; p=$(cat "$f")
  local n; n=$(basename "$p" | grep -oE '[0-9]+' | tail -1)
  echo "${n:-0}"
}

get_max_iters() {
  if [ -f "${WORK_DIR}/.max_iters" ]; then
    cat "${WORK_DIR}/.max_iters"
    return
  fi
  local latest_log
  latest_log=$(ls -t ${RUNS_DIR}/sama_abl_fd_${JOB_NAME}_*.out 2>/dev/null | head -1)
  if [ -n "$latest_log" ]; then
    local total
    total=$(grep -oE 'Iter\(train\) \[[[:space:]]*[0-9]+/[0-9]+\]' "$latest_log" | tail -1 | grep -oE '/[0-9]+\]' | tr -d '/]')
    if [ -n "$total" ]; then
      echo "$total" > "${WORK_DIR}/.max_iters"
      echo "$total"
      return
    fi
  fi
  echo ""
}

if [ -f "${WORK_DIR}/TRAINING_COMPLETE" ]; then
  echo "[chain] TRAINING_COMPLETE present; nothing to do. Exiting."
  exit 0
fi

# Duplicate-chain guard, keyed to this ablation's own job name only.
OTHER_RUNNING=$(squeue -u "$USER" -h -t RUNNING -o '%i %j' | grep "${JOB_NAME}\$" | grep -v "^${SLURM_JOB_ID} " || true)
if [ -n "${OTHER_RUNNING}" ]; then
  echo "[chain] another ${JOB_NAME} job is already RUNNING (${OTHER_RUNNING}); this is a duplicate fork. Exiting without training or re-chaining."
  exit 0
fi

MAX_ITERS=$(get_max_iters)
CUR_ITER=$(parse_ckpt_iter)
if [ -n "${MAX_ITERS}" ] && [ "${CUR_ITER}" -ge "${MAX_ITERS}" ]; then
  echo "[chain] reached MAX_ITERS (${CUR_ITER}/${MAX_ITERS}); marking complete."
  touch "${WORK_DIR}/TRAINING_COMPLETE"
  exit 0
fi

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

NEXT_JID=$(sbatch --parsable --job-name="${JOB_NAME}" --dependency=afterany:${SLURM_JOB_ID} ${SCRIPT_PATH} "${TAG}" "${CONFIG_NAME}" "${EXTRA_ENV}" 2>/dev/null || true)
echo "[chain] cur_iter=${CUR_ITER} max_iters=${MAX_ITERS:-<unknown>} stall=${STALL} queued successor=${NEXT_JID:-<none>}"

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

    if [ -n "'"${EXTRA_ENV}"'" ]; then
      export '"${EXTRA_ENV}"'
      echo "[sbatch] exported extra env: '"${EXTRA_ENV}"'"
    fi

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

MAX_ITERS=$(get_max_iters)
END_ITER=$(parse_ckpt_iter)
echo "[chain] srun_rc=${SRUN_RC} end_iter=${END_ITER}/${MAX_ITERS:-<unknown>}"
if [ "${SRUN_RC}" -eq 0 ] && [ -n "${MAX_ITERS}" ] && [ "${END_ITER}" -ge "${MAX_ITERS}" ]; then
  echo "[chain] training complete at iter ${END_ITER}; writing TRAINING_COMPLETE."
  touch "${WORK_DIR}/TRAINING_COMPLETE"
fi
