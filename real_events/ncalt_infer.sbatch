#!/bin/bash
#SBATCH --job-name=ncalt-infer
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/real_events/ncalt_infer_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/real_events/ncalt_infer_%j.err

# Headline full-corpus Event-SAMA on REAL N-Caltech101 events (animals), scored against the GT contour.
# Passes: (1) hot-pixel-filtered, 10 ms  [primary]   (2) filtered, 20 ms   (3) raw, 10 ms

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
EVAL_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck
CFG_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs
CONFIG=${CFG_DIR}/sama_4b_event_v3_fulldata.py
PTH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/sama_4b_event_v4_fulldata/iter_133208.pth
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/container/event-sama-arm64.sqsh
RE=/iopsstor/scratch/cscs/sfedorchenko/real_events

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((22000 + SLURM_JOB_ID % 8000))

run_pass () {   # $1=dataset  $2=event_fps  $3=tag  $4=converted dir
  DS=$1; FPS=$2; TAG=$3; CONV=$4
  OUT=${RE}/runs/ncalt_${TAG}_${SLURM_JOB_ID}
  mkdir -p "$OUT"
  echo "=== pass ${TAG}: dataset=${DS} event_fps=${FPS} -> ${OUT} ==="
  srun --container-image=$SQSH \
       --container-mounts=/iopsstor:/iopsstor,/capstor:/capstor,/users:/users \
       --container-workdir=$PROJECT_DIR \
       --no-container-mount-home \
       bash -c '
         export PYTHONPATH='"$PROJECT_DIR"':'"$EVAL_DIR"':'"$CFG_DIR"':$PYTHONPATH
         export TMPDIR=/tmp MPLBACKEND=Agg TRANSFORMERS_NO_FLASH_ATTN_2=1
         export RANK=${SLURM_PROCID:-0} LOCAL_RANK=0 WORLD_SIZE=${SLURM_NTASKS:-1}
         cd '"$PROJECT_DIR"'
         python3 '"$EVAL_DIR"'/ref_vos_event_eval_real.py \
           --config '"$CONFIG"' --pth_model '"$PTH"' --dataset '"$DS"' \
           --work_dir '"$OUT"' --launcher pytorch --event_fps '"$FPS"' 2>&1
       '
  echo "=== eval vs GT: ${TAG} ==="
  srun --ntasks=1 --gpus-per-task=0 --container-image=$SQSH \
       --container-mounts=/iopsstor:/iopsstor --no-container-mount-home \
       python3 ${RE}/eval_ncaltech.py "$OUT" "$CONV" 2>&1 | grep -v -i -E 'warning|libaio|CUTLASS|triton|pkg_resources|ds_accelerator'
}

run_pass REAL_NCALT_HF  100 hf10ms  ${RE}/ncaltech_converted_hf
run_pass REAL_NCALT_HF   50 hf20ms  ${RE}/ncaltech_converted_hf
run_pass REAL_NCALT_RAW 100 raw10ms ${RE}/ncaltech_converted
echo "=== all done ==="
