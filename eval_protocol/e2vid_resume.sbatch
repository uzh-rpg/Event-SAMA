#!/bin/bash
#SBATCH --job-name=e2vid-resume
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/e2vid/base_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/e2vid/base_%j.err
ulimit -c 0
# Baseline asked by reviewers: events -> E2VID reconstruction (pretrained, default settings)
# -> UNMODIFIED RGB SAMA-4B (weights/SAMA_4B, its own eval script).
PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
EVAL_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck
CFG_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/container/event-sama-arm64.sqsh
B=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/pseudorgb
RE=/iopsstor/scratch/cscs/sfedorchenko/real_events
MV=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/mevis/valid_u
MODEL=/iopsstor/scratch/cscs/sfedorchenko/weights/SAMA_4B
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((22000 + SLURM_JOB_ID % 8000))
CM="--container-image=$SQSH --container-mounts=/iopsstor:/iopsstor,/capstor:/capstor,/users:/users --container-workdir=$PROJECT_DIR --no-container-mount-home"
ENVS='export PYTHONPATH='"$PROJECT_DIR"':'"$EVAL_DIR"':'"$CFG_DIR"':$PYTHONPATH TMPDIR=/tmp MPLBACKEND=Agg; cd '"$PROJECT_DIR"

E=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/e2vid
echo "=== RGB SAMA-4B on MeViS valid_u E2VID ==="
run_rgb MEVIS_U_E2VID $E/run_mevis
echo "masks: $(find $E/run_mevis/Annotations -name '*.png' | wc -l)"
echo "=== all done ==="
