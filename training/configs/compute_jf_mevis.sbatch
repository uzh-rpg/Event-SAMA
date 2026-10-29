#!/bin/bash
#SBATCH --job-name=sama-jf-mevis
#SBATCH --partition=normal
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/jf_mevis_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/jf_mevis_%j.err

# Compute J&F for one ablation's MeViS valid_u predictions via the
# unmodified upstream eval_mevis.py (handles pred/GT resolution mismatch
# internally via nearest-neighbour resize). CPU-bound (128 workers), but
# requesting a GPU node anyway since --gpus-per-task=1 is what our
# container/allocation pattern is set up for on this cluster; the job
# itself doesn't touch the GPU.
# NAME is passed as $1.

NAME=$1
if [ -z "$NAME" ]; then
  echo "Usage: sbatch compute_jf_mevis.sbatch <ablation_name>"
  exit 1
fi

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
RUNS_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs
DATA_DIR=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event-sama-arm64.sqsh
PRED_PATH=${RUNS_DIR}/eval_mevis_u/${NAME}/Annotations
EXP_PATH=${DATA_DIR}/mevis/valid_u/meta_expressions_v2.json
MASK_PATH=${DATA_DIR}/mevis/valid_u/mask_dict.json

srun \
  --container-image=${SQSH} \
  --container-mounts=${PROJECT_DIR}:${PROJECT_DIR},${DATA_DIR}:${DATA_DIR},${RUNS_DIR}:${RUNS_DIR},/tmp:/tmp \
  --no-container-mount-home \
  bash -c '
    export PYTHONPATH='"${PROJECT_DIR}"':$PYTHONPATH
    export HOME=/tmp/hfhome_jf_'"${NAME}"'
    mkdir -p $HOME
    cd '"${PROJECT_DIR}"'/tools/eval_video_seg

    python eval_mevis.py \
      --mevis_pred_path '"${PRED_PATH}"' \
      --mevis_exp_path '"${EXP_PATH}"' \
      --mevis_mask_path '"${MASK_PATH}"' \
      --save_name '"${NAME}"'_mevis_result.json \
      2>&1
  '
echo "[jf-${NAME}] rc=$?"
# eval_mevis.py always writes to <dirname(pred_path)>/mevis_result.json
# regardless of --save_name (a quirk of the upstream script).
echo "=== ${NAME} J&F result ==="
cat ${RUNS_DIR}/eval_mevis_u/${NAME}/mevis_result.json 2>/dev/null || echo "(result file not found -- check output above)"
