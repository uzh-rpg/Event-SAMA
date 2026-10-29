#!/bin/bash
#SBATCH --job-name=eval-datacheck
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck/datacheck_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck/datacheck_%j.err

# CONTROLLED EXPERIMENT: run the OLD (July) eval script/dataset class,
# UNCHANGED except event_folder repointed at the CURRENT data directory,
# against a checkpoint whose score under the NEW eval pipeline is already
# known (sam2frozen ablation, iter_32711, new-pipeline J&F=16.24).
# Isolates code vs. data: if this run also lands near ~16, the eval code
# is what moved the number, not the data. If it lands near ~21 (July's
# number), the data itself differs from what's on disk now.

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
EVAL_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck
CFG_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs
CONFIG=${CFG_DIR}/sama_4b_event_v3_ablation_sam2frozen.py
PTH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/sama_4b_event_v4_ablation_sam2frozen/iter_32711.pth
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event-sama-arm64.sqsh

PTH_NAME=$(basename "$PTH" .pth)
EVAL_OUT=${EVAL_DIR}/runs/datacheck_sam2frozen_${PTH_NAME}_${SLURM_JOB_ID}
mkdir -p "$EVAL_OUT"

echo "=== DATA-VS-CODE CONTROL: old eval script + new data dir ==="
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
echo "=== Final result (DATACHECK, old-code + new-data, sam2frozen iter_32711) ==="
cat "${EVAL_OUT}/mevis_result.json" 2>/dev/null || cat "${SAVE_FILE}" 2>/dev/null || echo "(no results.json found)"
