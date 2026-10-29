#!/bin/bash
#SBATCH --job-name=sama-smoke-v3img
#SBATCH --partition=normal
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/smoke_v3img_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/smoke_v3img_%j.err

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
WEIGHTS_DIR=/iopsstor/scratch/cscs/sfedorchenko/weights
DATA_DIR=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset
EVENT_DIR_CAPSTOR=/capstor/scratch/cscs/mzehnde/sama_converted
EVENT_DIR_IOPSSTOR=/iopsstor/scratch/cscs/mzehnde/sama_converted
RUNS_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event-sama-arm64.sqsh

SHM_DIR=/tmp/sama_smoke_shm_${SLURM_JOB_ID}
mkdir -p ${SHM_DIR}

srun \
  --container-image=${SQSH} \
  --container-mounts=${PROJECT_DIR}:${PROJECT_DIR},${WEIGHTS_DIR}:${WEIGHTS_DIR},${DATA_DIR}:${DATA_DIR},${EVENT_DIR_CAPSTOR}:${EVENT_DIR_CAPSTOR},${EVENT_DIR_IOPSSTOR}:${EVENT_DIR_IOPSSTOR},${RUNS_DIR}:${RUNS_DIR},/tmp:/tmp,${SHM_DIR}:/dev/shm \
  --no-container-mount-home \
  bash -c '
    set -e
    export PYTHONPATH='"${PROJECT_DIR}"':$PYTHONPATH
    export TMPDIR=/tmp
    export MPLBACKEND=Agg
    export TRANSFORMERS_NO_FLASH_ATTN_2=1
    export TRANSFORMERS_CACHE=/tmp/transformers_cache
    mkdir -p $TRANSFORMERS_CACHE
    cd '"${PROJECT_DIR}"'
    echo "=== [smoke] python + import chain ==="
    python -c "import torch,mmengine,mmcv,mmdet,transformers,xtuner; from mmengine.config import Config; print(\"imports OK\", torch.__version__)"
    echo "=== [smoke] V3 datasets ==="
    python /iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs/smoke_event_v3.py
  '
