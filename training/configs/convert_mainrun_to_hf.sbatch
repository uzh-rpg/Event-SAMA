#!/bin/bash
#SBATCH --job-name=sama-convert-main
#SBATCH --partition=normal
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/convert_main_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/convert_main_%j.err

# Converts the single-node run's checkpoint to HF format. NOT iso-compute
# with the ablations (this run used a different effective batch size/LR
# schedule) -- just a snapshot of that production model, not an ablation-
# table entry.
#
# Corrected 2026-08-22: was pointed at iter_285000 (74% through training).
# That run has since actually finished -- real final checkpoint is
# iter_382416 (confirmed: "Iter(train) [382416/382416]" in the log, and
# the chain's own MAX_ITERS bookkeeping was stale so it never flagged
# completion -- see train_multinode_full.sbatch/train_multinode_best.sbatch
# for the same class of bug). Converting iter_285000 now would silently
# ship a less-trained checkpoint instead of the finished model.

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
RUNS_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs
CONFIGS_DIR=${RUNS_DIR}/configs
WEIGHTS_DIR=/iopsstor/scratch/cscs/sfedorchenko/weights
DATA_DIR=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset
EVENT_DIR_CAPSTOR=/capstor/scratch/cscs/mzehnde/sama_converted
EVENT_DIR_CAPSTOR_V2=/capstor/scratch/cscs/mzehnde/sama_converted_v2
EVENT_DIR_CAPSTOR_TRANSFER=/capstor/scratch/cscs/mzehnde/sama_converted_transfer
EVENT_DIR_CAPSTOR_REVOS=/capstor/scratch/cscs/mzehnde/sama_converted_revos
EVENT_DIR_IOPSSTOR=/iopsstor/scratch/cscs/mzehnde/sama_converted
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event-sama-arm64.sqsh
CONFIG=${CONFIGS_DIR}/sama_4b_event_v3.py
PTH=${RUNS_DIR}/sama_4b_event_v4/iter_382416.pth
SAVE=${RUNS_DIR}/hf_checkpoints/mainrun_iter382416

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
    export HOME=/tmp/hfhome_main
    export HF_HOME=$HOME/huggingface
    export HF_MODULES_CACHE=$HOME/huggingface/modules
    export HUGGINGFACE_HUB_CACHE=$HOME/huggingface/hub
    mkdir -p $TRANSFORMERS_CACHE $HF_HOME $HF_MODULES_CACHE $HUGGINGFACE_HUB_CACHE

    cd '"${PROJECT_DIR}"'

    python '"${CONFIGS_DIR}"'/convert_to_hf_event.py \
      --config '"${CONFIG}"' \
      --pth_model '"${PTH}"' \
      --save_path '"${SAVE}"' \
      2>&1
  '
echo "[convert-main] rc=$?"
