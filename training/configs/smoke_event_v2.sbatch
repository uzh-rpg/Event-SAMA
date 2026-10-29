#!/bin/bash
#SBATCH --job-name=smoke-event-v2
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/smoke_event_v2_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/smoke_event_v2_%j.err

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
WEIGHTS_DIR=/iopsstor/scratch/cscs/sfedorchenko/weights
DATA_DIR=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset
EVENT_DIR_CAPSTOR=/capstor/scratch/cscs/mzehnde/sama_converted
EVENT_DIR_IOPSSTOR=/iopsstor/scratch/cscs/mzehnde/sama_converted
RUNS_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs
SQSH=${HOME}/nvidia+pytorch+24.01-py3.sqsh
PYDEPS=/iopsstor/scratch/cscs/sfedorchenko/event_sama_pydeps

echo "=== V2 dataset smoke test ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"

srun \
  --container-image=${SQSH} \
  --container-mounts=${PROJECT_DIR}:${PROJECT_DIR},${WEIGHTS_DIR}:${WEIGHTS_DIR},${DATA_DIR}:${DATA_DIR},${EVENT_DIR_CAPSTOR}:${EVENT_DIR_CAPSTOR},${EVENT_DIR_IOPSSTOR}:${EVENT_DIR_IOPSSTOR},${RUNS_DIR}:${RUNS_DIR},${PYDEPS}:${PYDEPS},/tmp:/tmp \
  --no-container-mount-home \
  bash -c "
    set -e
    export PYTHONPATH=${PYDEPS}:${PROJECT_DIR}:\$PYTHONPATH
    if ! python -c 'import h5py' >/dev/null 2>&1; then
      echo '[setup] installing h5py into PYDEPS...'
      pip install --no-deps --target=${PYDEPS} 'h5py>=3.10,<4'
    fi
    python ${RUNS_DIR}/configs/smoke_event_v2.py
  "
