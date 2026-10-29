#!/bin/bash
#SBATCH --job-name=build-meta-cache
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/build_meta_cache_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/build_meta_cache_%j.err

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
RUNS_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs
EVENT_DIR_IOPSSTOR=/iopsstor/scratch/cscs/mzehnde/sama_converted
EVENT_DIR_CAPSTOR=/capstor/scratch/cscs/mzehnde/sama_converted
# Self-contained arm64 image (deps baked into /opt/sama_deps); no PYDEPS.
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event-sama-arm64.sqsh

echo "=== Build ChatUniVi event-meta cache (V3, full capstor/chatunivi) ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"

srun \
  --container-image=${SQSH} \
  --container-mounts=${PROJECT_DIR}:${PROJECT_DIR},${EVENT_DIR_IOPSSTOR}:${EVENT_DIR_IOPSSTOR},${EVENT_DIR_CAPSTOR}:${EVENT_DIR_CAPSTOR},${RUNS_DIR}:${RUNS_DIR},/tmp:/tmp \
  --no-container-mount-home \
  bash -c "
    set -e
    export PYTHONPATH=${PROJECT_DIR}:\$PYTHONPATH
    export TMPDIR=/tmp
    python ${RUNS_DIR}/configs/build_meta_cache_v3.py
  "
