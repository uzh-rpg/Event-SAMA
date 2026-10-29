#!/bin/bash
#SBATCH --job-name=pseudo-ncalt
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/pseudorgb/pncalt_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/pseudorgb/pncalt_%j.err
ulimit -c 0
# Pseudo-RGB baseline on N-Caltech101 (hot-pixel-filtered, 10 ms window = Event-SAMA primary pass),
# frozen RGB SAMA-4B. Scoring on the login node with eval_ncaltech.py.
PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
EVAL_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck
CFG_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/container/event-sama-arm64.sqsh
B=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/pseudorgb
RE=/iopsstor/scratch/cscs/sfedorchenko/real_events
MODEL=/iopsstor/scratch/cscs/sfedorchenko/weights/SAMA_4B
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((22000 + SLURM_JOB_ID % 8000))
CM="--container-image=$SQSH --container-mounts=/iopsstor:/iopsstor,/capstor:/capstor,/users:/users --container-workdir=$PROJECT_DIR --no-container-mount-home"
ENVS='export PYTHONPATH='"$PROJECT_DIR"':'"$EVAL_DIR"':'"$CFG_DIR"':$PYTHONPATH TMPDIR=/tmp MPLBACKEND=Agg; cd '"$PROJECT_DIR"
srun $CM bash -c "$ENVS"'
  python3 '"$EVAL_DIR"'/render_pseudorgb_frames.py '"$RE"'/ncaltech_converted_hf '"$RE"'/ncaltech_converted_hf/meta_expressions.json '"$B"'/ncalt_hf_10ms 100 $SLURM_PROCID 4
' 2>&1 | grep -v -i warning
srun $CM bash -c "$ENVS"'
  export RANK=${SLURM_PROCID:-0} LOCAL_RANK=0 WORLD_SIZE=${SLURM_NTASKS:-1}
  python3 '"$EVAL_DIR"'/ref_vos_eval_st_pseudo.py '"$MODEL"' --dataset NCALT_PSEUDO --launcher pytorch --work_dir '"$B"'/run_ncalt
' 2>&1 | grep -v -i -E 'warning|libaio|CUTLASS|triton|pkg_resources|ds_accelerator' | tail -30
echo "masks: $(find $B/run_ncalt/Annotations -name '*.png' | wc -l)"
