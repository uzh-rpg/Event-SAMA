#!/bin/bash
#SBATCH --job-name=eval-revos
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck/revos_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck/revos_%j.err

# ReVOS valid eval (partial coverage: 329/416 videos have real event data)
# for one of the 3 final trained models, using the validated methodology
# (direct xtuner-time model class, 5 frames to LLM, 32 dense frames to SAM2).
# Usage: sbatch revos_eval.sbatch <tag> <config_basename> <pth_path>

TAG=$1
CONFIG_NAME=$2
PTH=$3
if [ -z "$TAG" ] || [ -z "$CONFIG_NAME" ] || [ -z "$PTH" ]; then
  echo "Usage: sbatch revos_eval.sbatch <tag> <config_basename> <pth_path>"
  exit 1
fi

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
EVAL_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck
CFG_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs
CONFIG=${CFG_DIR}/${CONFIG_NAME}
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event-sama-arm64.sqsh

EVAL_OUT=${EVAL_DIR}/runs/revos_${TAG}_${SLURM_JOB_ID}
mkdir -p "$EVAL_OUT"

echo "=== ReVOS valid eval: ${TAG} ==="
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
       cd '"$PROJECT_DIR"'
       python3 '"$EVAL_DIR"'/ref_vos_event_eval_newdata.py \
         --config '"$CONFIG"' \
         --pth_model '"$PTH"' \
         --dataset REVOS_VALID \
         --work_dir '"$EVAL_OUT"' \
         --launcher pytorch \
         2>&1
     '

echo ""
echo "=== Computing J&F metrics (ReVOS, via eval_revos.py) ==="
EXP_FILE=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/sa2va_training/video_datas/revos/meta_expressions_valid_.json
MASK_FILE=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/sa2va_training/video_datas/revos/mask_dict.json
FG_MASK_FILE=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/sa2va_training/video_datas/revos/mask_dict_foreground.json
PRED_PATH=${EVAL_OUT}/Annotations
SAVE_JSON=${EVAL_OUT}/revos_result.json
SAVE_CSV=${EVAL_OUT}/revos_result.csv

srun --ntasks=1 --gpus-per-task=0 \
     --container-image=$SQSH \
     --container-mounts=/iopsstor:/iopsstor,/users:/users \
     --container-workdir=$PROJECT_DIR \
     --no-container-mount-home \
     bash -c '
       export PYTHONPATH='"$PROJECT_DIR"':'"$PROJECT_DIR"'/tools/eval_video_seg:$PYTHONPATH
       cd '"$PROJECT_DIR"'/tools/eval_video_seg
       python3 eval_revos.py \
         --visa_pred_path '"$PRED_PATH"' \
         --visa_exp_path '"$EXP_FILE"' \
         --visa_mask_path '"$MASK_FILE"' \
         --visa_foreground_mask_path '"$FG_MASK_FILE"' \
         --save_json_name '"$SAVE_JSON"' \
         --save_csv_name '"$SAVE_CSV"'
     '

echo ""
echo "=== Final ReVOS result (${TAG}) ==="
cat "${SAVE_JSON}" 2>/dev/null || echo "(no result json found)"
