#!/bin/bash
#SBATCH --job-name=eval-oldproto
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck/oldproto_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck/oldproto_%j.err

# Old-protocol (July-style) MeViS valid_u eval on current data, parametrized.
# Usage: sbatch datacheck_eval_param.sbatch <name> <config> <pth>
# Same harness as the sam2frozen control (job 4523624, J&F=22.24): old eval
# script + old dataset class, event_folder pointed at the current
# sama_converted_transfer/max_sergei/mevis directory. Loads raw .pth
# directly (no HF conversion).

NAME=$1
CONFIG=$2
PTH=$3
if [ -z "$NAME" ] || [ -z "$CONFIG" ] || [ -z "$PTH" ]; then
  echo "Usage: sbatch datacheck_eval_param.sbatch <name> <config> <pth>"
  exit 1
fi

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
EVAL_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck
CFG_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event-sama-arm64.sqsh

EVAL_OUT=${EVAL_DIR}/runs/oldproto_${NAME}_${SLURM_JOB_ID}
mkdir -p "$EVAL_OUT"

echo "=== OLD-PROTOCOL EVAL: ${NAME} ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "Config: ${CONFIG}"
echo "Pth:    ${PTH}"
echo "Out:    ${EVAL_OUT}"

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
       echo "[srun] rank=${RANK}/${WORLD_SIZE} cuda_visible=${CUDA_VISIBLE_DEVICES:-?}"
       cd '"$PROJECT_DIR"'
       python3 '"$EVAL_DIR"'/ref_vos_event_eval_newdata.py \
         --config '"$CONFIG"' \
         --pth_model '"$PTH"' \
         --dataset MEVIS_VALID_U \
         --work_dir '"$EVAL_OUT"' \
         --launcher pytorch \
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
echo "=== Final result (OLD PROTOCOL, ${NAME}) ==="
cat "${EVAL_OUT}/mevis_result.json" 2>/dev/null || cat "${SAVE_FILE}" 2>/dev/null || echo "(no results.json found)"
