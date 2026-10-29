#!/bin/bash
#SBATCH --job-name=eval-sweep
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck/sweep_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck/sweep_%j.err

# Protocol-robustness sweeps on the baseline epoch-1 checkpoint (old
# protocol). Usage:
#   sbatch sweep_eval.sbatch <tag> [extra ref_vos_event_eval args...]
# e.g.  sbatch sweep_eval.sbatch K8  --max_dense_frames 8
#       sbatch sweep_eval.sbatch W20ms --event_fps 50
# Fixed checkpoint/config: baseline multinode_full epoch-1 (iter34000),
# the K=32/10ms cell of the sweep is the existing oldproto_baseline run
# (J&F 23.04) -- do not resubmit it.

TAG=$1; shift
EXTRA_ARGS="$@"
if [ -z "$TAG" ]; then
  echo "Usage: sbatch sweep_eval.sbatch <tag> [extra eval args]"
  exit 1
fi

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
EVAL_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck
CFG_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs
CONFIG=${CFG_DIR}/sama_4b_event_v3_multinode_full.py
PTH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/epoch_checkpoints/multinode_full_epoch1_iter34000.pth
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event-sama-arm64.sqsh

EVAL_OUT=${EVAL_DIR}/runs/sweep_${TAG}_${SLURM_JOB_ID}
mkdir -p "$EVAL_OUT"

echo "=== PROTOCOL SWEEP: ${TAG}  extra_args='${EXTRA_ARGS}' ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "Pth: ${PTH}"
echo "Out: ${EVAL_OUT}"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((22000 + SLURM_JOB_ID % 8000))

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
       python3 '"$EVAL_DIR"'/ref_vos_event_eval_newdata.py \
         --config '"$CONFIG"' \
         --pth_model '"$PTH"' \
         --dataset MEVIS_VALID_U \
         --work_dir '"$EVAL_OUT"' \
         --launcher pytorch \
         '"$EXTRA_ARGS"' \
         2>&1
     '

echo ""
echo "=== Computing J&F metrics ==="
EXP_FILE=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/mevis/valid_u/meta_expressions_v2.json
MASK_FILE=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/mevis/valid_u/mask_dict.json
PRED_PATH=${EVAL_OUT}/Annotations
SAVE_FILE=${EVAL_OUT}/mevis_valu_result.json

srun --ntasks=1 --gpus-per-task=0 \
     --container-image=$SQSH \
     --container-mounts=/iopsstor:/iopsstor,/users:/users \
     --container-workdir=$PROJECT_DIR \
     --no-container-mount-home \
     bash -c '
       export PYTHONPATH='"$PROJECT_DIR"':'"$PROJECT_DIR"'/tools/eval_video_seg:$PYTHONPATH
       cd '"$PROJECT_DIR"'/tools/eval_video_seg
       python3 eval_mevis.py \
         --mevis_pred_path '"$PRED_PATH"' \
         --mevis_exp_path '"$EXP_FILE"' \
         --mevis_mask_path '"$MASK_FILE"' \
         --save_name '"$SAVE_FILE"'
     '

echo ""
echo "=== Final result (SWEEP ${TAG}) ==="
cat "${EVAL_OUT}/mevis_result.json" 2>/dev/null || cat "${SAVE_FILE}" 2>/dev/null || echo "(no results.json found)"
