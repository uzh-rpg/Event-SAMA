#!/bin/bash
#SBATCH --job-name=sama-convert-hf
#SBATCH --partition=normal
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/convert_hf_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/convert_hf_%j.err

# Converts each ablation's DeepSpeed ZeRO-2 checkpoint (iter_32711.pth dir)
# into a loadable HF model dir, for eval/inference. Sequential, single GPU
# (loading + merging one checkpoint at a time is enough; no need for 16-way
# distributed loading here).

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
RUNS_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs
CONFIGS_DIR=${RUNS_DIR}/configs
HF_OUT_DIR=${RUNS_DIR}/hf_checkpoints
WEIGHTS_DIR=/iopsstor/scratch/cscs/sfedorchenko/weights
DATA_DIR=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset
EVENT_DIR_CAPSTOR=/capstor/scratch/cscs/mzehnde/sama_converted
EVENT_DIR_CAPSTOR_V2=/capstor/scratch/cscs/mzehnde/sama_converted_v2
EVENT_DIR_CAPSTOR_TRANSFER=/capstor/scratch/cscs/mzehnde/sama_converted_transfer
EVENT_DIR_CAPSTOR_REVOS=/capstor/scratch/cscs/mzehnde/sama_converted_revos
EVENT_DIR_IOPSSTOR=/iopsstor/scratch/cscs/mzehnde/sama_converted
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event-sama-arm64.sqsh

mkdir -p ${HF_OUT_DIR}

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
    export HOME=/tmp/hfhome
    export HF_HOME=/tmp/hfhome/huggingface
    export HF_MODULES_CACHE=/tmp/hfhome/huggingface/modules
    export HUGGINGFACE_HUB_CACHE=/tmp/hfhome/huggingface/hub
    mkdir -p $TRANSFORMERS_CACHE $HF_HOME $HF_MODULES_CACHE $HUGGINGFACE_HUB_CACHE

    cd '"${PROJECT_DIR}"'

    for NAME in noadapter sam2frozen patchrandom vitunfreeze; do
      CONFIG='"${CONFIGS_DIR}"'/sama_4b_event_v3_ablation_${NAME}.py
      PTH='"${RUNS_DIR}"'/sama_4b_event_v4_ablation_${NAME}/iter_32711.pth
      SAVE='"${HF_OUT_DIR}"'/${NAME}
      if [ -f "$SAVE/config.json" ]; then
        echo "=== Skipping ${NAME} (already converted) ==="
        continue
      fi
      echo "=== Converting ${NAME} ==="
      echo "config: $CONFIG"
      echo "pth:    $PTH"
      echo "save:   $SAVE"
      mkdir -p "$SAVE"
      python '"${CONFIGS_DIR}"'/convert_to_hf_event.py \
        --config "$CONFIG" \
        --pth_model "$PTH" \
        --save_path "$SAVE" \
        2>&1
      if [ -f "$SAVE/config.json" ]; then
        echo "=== Done ${NAME} ==="
      else
        echo "=== FAILED ${NAME} (no config.json written) -- continuing to next ==="
      fi
    done
  '
echo "[chain] all conversions attempted, rc=$?"
