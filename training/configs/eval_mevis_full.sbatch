#!/bin/bash
#SBATCH --job-name=sama-eval-mevis
#SBATCH --partition=normal
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval_mevis_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval_mevis_%j.err

# Full MeViS valid_u event-VOS eval (50 videos, 793 expression items) for
# one ablation checkpoint. 4-GPU distributed inference (DistributedSampler),
# same env-var-export pattern proven in the training sbatch scripts.
# NAME is passed as $1 (noadapter / sam2frozen / patchrandom / vitunfreeze).

NAME=$1
MERGE_POLARITY=${2:-0}   # pass 1 for the nopolarity checkpoint, to match its training-time preprocessing
if [ -z "$NAME" ]; then
  echo "Usage: sbatch eval_mevis_full.sbatch <ablation_name> [merge_polarity=0|1]"
  exit 1
fi

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
RUNS_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs
WEIGHTS_DIR=/iopsstor/scratch/cscs/sfedorchenko/weights
DATA_DIR=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset
EVENT_DIR_CAPSTOR=/capstor/scratch/cscs/mzehnde/sama_converted
EVENT_DIR_CAPSTOR_V2=/capstor/scratch/cscs/mzehnde/sama_converted_v2
EVENT_DIR_CAPSTOR_TRANSFER=/capstor/scratch/cscs/mzehnde/sama_converted_transfer
EVENT_DIR_CAPSTOR_REVOS=/capstor/scratch/cscs/mzehnde/sama_converted_revos
EVENT_DIR_IOPSSTOR=/iopsstor/scratch/cscs/mzehnde/sama_converted
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event-sama-arm64.sqsh
MODEL_PATH=${RUNS_DIR}/hf_checkpoints/${NAME}
WORK_DIR=${RUNS_DIR}/eval_mevis_u/${NAME}

mkdir -p ${WORK_DIR}

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}  NAME=${NAME}"

srun \
  --container-image=${SQSH} \
  --container-mounts=${PROJECT_DIR}:${PROJECT_DIR},${WEIGHTS_DIR}:${WEIGHTS_DIR},${DATA_DIR}:${DATA_DIR},${EVENT_DIR_CAPSTOR}:${EVENT_DIR_CAPSTOR},${EVENT_DIR_CAPSTOR_V2}:${EVENT_DIR_CAPSTOR_V2},${EVENT_DIR_CAPSTOR_TRANSFER}:${EVENT_DIR_CAPSTOR_TRANSFER},${EVENT_DIR_CAPSTOR_REVOS}:${EVENT_DIR_CAPSTOR_REVOS},${EVENT_DIR_IOPSSTOR}:${EVENT_DIR_IOPSSTOR},${RUNS_DIR}:${RUNS_DIR},/tmp:/tmp \
  --no-container-mount-home \
  bash -c '
    export PYTHONPATH='"${PROJECT_DIR}"':$PYTHONPATH
    export TMPDIR=/tmp
    export DISPLAY=""
    export MPLBACKEND=Agg
    export TRANSFORMERS_NO_FLASH_ATTN_2=1
    export TRANSFORMERS_CACHE=/tmp/transformers_cache
    export HOME=/tmp/hfhome_'"${NAME}"'_${SLURM_PROCID}
    export HF_HOME=$HOME/huggingface
    export HF_MODULES_CACHE=$HOME/huggingface/modules
    export HUGGINGFACE_HUB_CACHE=$HOME/huggingface/hub
    mkdir -p $TRANSFORMERS_CACHE $HF_HOME $HF_MODULES_CACHE $HUGGINGFACE_HUB_CACHE

    export RANK=${SLURM_PROCID}
    export LOCAL_RANK=0
    export WORLD_SIZE=${SLURM_NTASKS}
    export EVENT_MERGE_POLARITY='"${MERGE_POLARITY}"'
    echo "[task ${SLURM_PROCID}/${SLURM_NTASKS}] host=$(hostname) RANK=${RANK} WORLD_SIZE=${WORLD_SIZE} EVENT_MERGE_POLARITY=${EVENT_MERGE_POLARITY}"

    cd '"${PROJECT_DIR}"'

    python projects/llava_sam2/evaluation/ref_vos_eval_event.py \
      '"${MODEL_PATH}"' \
      --dataset MEVIS_U_EVENT \
      --launcher pytorch \
      --work_dir '"${WORK_DIR}"' \
      2>&1
  '
echo "[eval-${NAME}] rc=$?"
