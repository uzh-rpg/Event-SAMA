#!/bin/bash
#SBATCH --job-name=mouse-train-inf
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/real_events/mousesis_train_inf_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/real_events/mousesis_train_inf_%j.err
ulimit -c 0
# Headline full-corpus Event-SAMA on REAL MouseSIS events (val split, 4 clips x 4 expressions).
# MouseSIS TRAIN split (21 clips, public GT, never used for training): density-matched 75 ms
# window (rule fixed on val), the 10 ms training window, and 150/300 ms for the per-clip density rule
# (train clips are sparser: 0.2-1.3 ev/px/s). Scoring on the login node.
PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
EVAL_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck
CFG_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs
CONFIG=${CFG_DIR}/sama_4b_event_v3_fulldata.py
PTH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/sama_4b_event_v4_fulldata/iter_133208.pth
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/container/event-sama-arm64.sqsh
RE=/iopsstor/scratch/cscs/sfedorchenko/real_events
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((22000 + SLURM_JOB_ID % 8000))
for PASS in "13.333 75ms" "100 10ms" "6.667 150ms" "3.333 300ms"; do
  set -- $PASS; FPS=$1; TAG=$2
  OUT=${RE}/runs/mousesis_train_${TAG}_${SLURM_JOB_ID}
  mkdir -p "$OUT"
  echo "=== pass ${TAG}: event_fps=${FPS} -> ${OUT} ==="
  srun --container-image=$SQSH \
       --container-mounts=/iopsstor:/iopsstor,/capstor:/capstor,/users:/users \
       --container-workdir=$PROJECT_DIR --no-container-mount-home \
       bash -c '
         export PYTHONPATH='"$PROJECT_DIR"':'"$EVAL_DIR"':'"$CFG_DIR"':$PYTHONPATH
         export TMPDIR=/tmp MPLBACKEND=Agg TRANSFORMERS_NO_FLASH_ATTN_2=1
         export RANK=${SLURM_PROCID:-0} LOCAL_RANK=0 WORLD_SIZE=${SLURM_NTASKS:-1}
         cd '"$PROJECT_DIR"'
         python3 '"$EVAL_DIR"'/ref_vos_event_eval_real.py \
           --config '"$CONFIG"' --pth_model '"$PTH"' --dataset REAL_MOUSESIS_TRAIN \
           --work_dir '"$OUT"' --launcher pytorch --event_fps '"$FPS"' 2>&1
       ' | grep -v -i -E 'warning|libaio|CUTLASS|triton|pkg_resources|ds_accelerator'
done
echo "=== all done ==="
