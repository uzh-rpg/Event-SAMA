#!/bin/bash
#SBATCH --job-name=sama-eval-smoke
#SBATCH --partition=normal
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval_smoke_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval_smoke_%j.err

# Smoke test: run the event-native VOS eval pipeline on just 2
# (video, expression) items from MeViS valid_u, single GPU, single ablation
# checkpoint (noadapter). Verifies the whole chain -- EventRefVOSDataset,
# predict_forward_event, mask_save -- before committing to the full
# 50-video x 4-checkpoint run.

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
RUNS_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs
WEIGHTS_DIR=/iopsstor/scratch/cscs/sfedorchenko/weights
DATA_DIR=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset
EVENT_DIR_CAPSTOR=/capstor/scratch/cscs/mzehnde/sama_converted
EVENT_DIR_CAPSTOR_V2=/capstor/scratch/cscs/mzehnde/sama_converted_v2
EVENT_DIR_CAPSTOR_TRANSFER=/capstor/scratch/cscs/mzehnde/sama_converted_transfer
EVENT_DIR_CAPSTOR_REVOS=/capstor/scratch/cscs/mzehnde/sama_converted_revos
EVENT_DIR_IOPSSTOR=/iopsstor/scratch/cscs/mzehnde/sama_converted
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event-sama-arm64.sqsh
MODEL_PATH=${RUNS_DIR}/hf_checkpoints/sam2frozen
WORK_DIR=${RUNS_DIR}/eval_smoke_test

mkdir -p ${WORK_DIR}

srun \
  --container-image=${SQSH} \
  --container-mounts=${PROJECT_DIR}:${PROJECT_DIR},${WEIGHTS_DIR}:${WEIGHTS_DIR},${DATA_DIR}:${DATA_DIR},${EVENT_DIR_CAPSTOR}:${EVENT_DIR_CAPSTOR},${EVENT_DIR_CAPSTOR_V2}:${EVENT_DIR_CAPSTOR_V2},${EVENT_DIR_CAPSTOR_TRANSFER}:${EVENT_DIR_CAPSTOR_TRANSFER},${EVENT_DIR_CAPSTOR_REVOS}:${EVENT_DIR_CAPSTOR_REVOS},${EVENT_DIR_IOPSSTOR}:${EVENT_DIR_IOPSSTOR},${RUNS_DIR}:${RUNS_DIR},/tmp:/tmp \
  --no-container-mount-home \
  bash -c '
    export PYTHONPATH='"${PROJECT_DIR}"':$PYTHONPATH
    export TMPDIR=/tmp
    export DISPLAY=""
    export MPLBACKEND=Agg
    export TRANSFORMERS_NO_FLASH_ATTN_2=1
    export TRANSFORMERS_CACHE=/tmp/transformers_cache
    export HOME=/tmp/hfhome
    export HF_HOME=/tmp/hfhome/huggingface
    export HF_MODULES_CACHE=/tmp/hfhome/huggingface/modules
    export HUGGINGFACE_HUB_CACHE=/tmp/hfhome/huggingface/hub
    mkdir -p $TRANSFORMERS_CACHE $HF_HOME $HF_MODULES_CACHE $HUGGINGFACE_HUB_CACHE

    cd '"${PROJECT_DIR}"'

    python projects/llava_sam2/evaluation/ref_vos_eval_event.py \
      '"${MODEL_PATH}"' \
      --dataset MEVIS_U_EVENT \
      --launcher none \
      --work_dir '"${WORK_DIR}"' \
      --limit 2 \
      2>&1
  '
echo "[smoke] rc=$?"
