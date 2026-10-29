#!/bin/bash
#SBATCH --job-name=eval-rebuilt-validate
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck/fulldata_final_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck/fulldata_final_%j.err

# Evaluates the single-node main run's FINAL checkpoint (iter_382416, ~4
# epochs) with the validated eval methodology (direct xtuner-time model
# class, no HF-conversion round-trip; 5 frames to the LLM matching
# DENSE_LEN, 32 dense frames to SAM2) -- the only one of the 3 final
# trained models nobody had evaluated correctly yet.

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
EVAL_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck
CFG_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs
CONFIG=${CFG_DIR}/sama_4b_event_v3_fulldata.py
PTH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/sama_4b_event_v4_fulldata/iter_133208.pth
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/container/event-sama-arm64.sqsh

PTH_NAME=$(basename "$PTH" .pth)
EVAL_OUT=${EVAL_DIR}/runs/fulldata_final_${PTH_NAME}_${SLURM_JOB_ID}
mkdir -p "$EVAL_OUT"

echo "=== Full-data-mix run FINAL checkpoint (iter_133208, genuinely complete 11-source corpus) (validated methodology) ==="
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
echo "=== Final result (fulldata run, iter_133208, ~4 epochs, full corpus) ==="
cat "${EVAL_OUT}/mevis_result.json" 2>/dev/null || cat "${SAVE_FILE}" 2>/dev/null || echo "(no results.json found)"
