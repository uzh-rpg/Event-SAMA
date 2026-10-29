#!/bin/bash
#SBATCH --job-name=real-dsec-infer
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/real_events/infer_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/real_events/infer_%j.err

# Inference of the headline full-corpus Event-SAMA checkpoint on REAL DSEC
# event clips (no ground truth; qualitative + mask statistics only).
# Pass 1: 10 ms window (training-matched, primary). Pass 2: 20 ms (sensitivity).
# Usage: sbatch real_infer.sbatch [config_basename] [pth] [tag]

CONFIG_NAME=${1:-sama_4b_event_v3_fulldata.py}
PTH=${2:-/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/sama_4b_event_v4_fulldata/iter_133208.pth}
TAG=${3:-fulldata_133208}

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
EVAL_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck
CFG_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs
CONFIG=${CFG_DIR}/${CONFIG_NAME}
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/container/event-sama-arm64.sqsh
RE=/iopsstor/scratch/cscs/sfedorchenko/real_events

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((22000 + SLURM_JOB_ID % 8000))

echo "Job ${SLURM_JOB_ID} on $(hostname)  config=${CONFIG_NAME}  pth=${PTH}"

for FPS in 100 50; do
  WIN_MS=$((1000 / FPS))
  OUT=${RE}/runs/${TAG}_${WIN_MS}ms_${SLURM_JOB_ID}
  mkdir -p "$OUT"
  echo "=== inference: window ${WIN_MS} ms -> ${OUT} ==="
  srun --container-image=$SQSH \
       --container-mounts=/iopsstor:/iopsstor,/capstor:/capstor,/users:/users \
       --container-workdir=$PROJECT_DIR \
       --no-container-mount-home \
       bash -c '
         export PYTHONPATH='"$PROJECT_DIR"':'"$EVAL_DIR"':'"$CFG_DIR"':$PYTHONPATH
         export TMPDIR=/tmp
         export MPLBACKEND=Agg
         export TRANSFORMERS_NO_FLASH_ATTN_2=1
         export RANK=${SLURM_PROCID:-0}
         export LOCAL_RANK=0
         export WORLD_SIZE=${SLURM_NTASKS:-1}
         cd '"$PROJECT_DIR"'
         python3 '"$EVAL_DIR"'/ref_vos_event_eval_real.py \
           --config '"$CONFIG"' \
           --pth_model '"$PTH"' \
           --dataset REAL_DSEC \
           --work_dir '"$OUT"' \
           --launcher pytorch \
           --event_fps '"$FPS"' \
           2>&1
       '

  echo "=== rendering panels + mask stats (${WIN_MS} ms) ==="
  srun --ntasks=1 --gpus-per-task=0 \
       --container-image=$SQSH \
       --container-mounts=/iopsstor:/iopsstor \
       --no-container-mount-home \
       bash -c "python3 ${RE}/render_overlays.py ${OUT} ${TAG}_${WIN_MS}ms ${WIN_MS}" 2>&1
done
echo "=== all done ==="
