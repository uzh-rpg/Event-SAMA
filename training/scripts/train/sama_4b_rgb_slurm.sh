#!/bin/bash
#SBATCH --job-name=sama4b-rgb
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/slurm_sama4b_rgb_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/slurm_sama4b_rgb_%j.err

PROJECT_DIR=/users/sfedorchenko/event_sama/SAMA
CONFIG=${PROJECT_DIR}/projects/llava_sam2/configs/sama_4b_rgb.py
WEIGHTS_DIR=/iopsstor/scratch/cscs/sfedorchenko/weights
DATA_DIR=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset
RUNS_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs
WORK_DIR=${RUNS_DIR}/sama_4b_rgb
SQSH=${HOME}/nvidia+pytorch+24.01-py3.sqsh
PYDEPS=/iopsstor/scratch/cscs/sfedorchenko/event_sama_pydeps
LOGFILE=${RUNS_DIR}/slurm_sama4b_rgb_${SLURM_JOB_ID}.log

mkdir -p ${WORK_DIR} ${RUNS_DIR}

echo "=== SAMA-4B RGB Video Pipeline Training ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "Config: ${CONFIG}"
echo "Work dir: ${WORK_DIR}"

srun \
  --container-image=${SQSH} \
  --container-mounts=${PROJECT_DIR}:${PROJECT_DIR},${WEIGHTS_DIR}:${WEIGHTS_DIR},${DATA_DIR}:${DATA_DIR},${RUNS_DIR}:${RUNS_DIR},${PYDEPS}:${PYDEPS},/tmp:/tmp \
  --no-container-mount-home \
  bash -c "
    set -e
    export PYTHONPATH=${PYDEPS}:\$PYTHONPATH
    export TMPDIR=/tmp
    mkdir -p \$TMPDIR

    export DISPLAY=\"\"
    export MPLBACKEND=Agg

    # DISABLE Flash Attention 2 – container ships 2.0.4, transformers requires >= 2.1.0
    export TRANSFORMERS_NO_FLASH_ATTN_2=1
    export TRANSFORMERS_CACHE=/tmp/transformers_cache
    mkdir -p \$TRANSFORMERS_CACHE
    export PYDEPS=${PYDEPS}

    cd ${PROJECT_DIR}

    torchrun --nnodes=1 --nproc_per_node=4 \
      tools/train_no_fa2.py ${CONFIG} \
      --work-dir=${WORK_DIR} \
      2>&1 | tee ${LOGFILE}
  "
