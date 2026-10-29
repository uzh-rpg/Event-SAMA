#!/bin/bash
#SBATCH --job-name=sama-fixsmoke-v3
#SBATCH --partition=normal
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/fixsmoke_v3_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/fixsmoke_v3_%j.err

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
WEIGHTS_DIR=/iopsstor/scratch/cscs/sfedorchenko/weights
DATA_DIR=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset
EVENT_DIR_CAPSTOR=/capstor/scratch/cscs/mzehnde/sama_converted
EVENT_DIR_IOPSSTOR=/iopsstor/scratch/cscs/mzehnde/sama_converted
RUNS_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs
SQSH=${HOME}/nvidia+pytorch+24.01-py3.sqsh
PYDEPS=/iopsstor/scratch/cscs/sfedorchenko/event_sama_pydeps

SHM_DIR=/tmp/sama_smoke_shm_${SLURM_JOB_ID}
mkdir -p ${SHM_DIR}

srun \
  --container-image=${SQSH} \
  --container-mounts=${PROJECT_DIR}:${PROJECT_DIR},${WEIGHTS_DIR}:${WEIGHTS_DIR},${DATA_DIR}:${DATA_DIR},${EVENT_DIR_CAPSTOR}:${EVENT_DIR_CAPSTOR},${EVENT_DIR_IOPSSTOR}:${EVENT_DIR_IOPSSTOR},${RUNS_DIR}:${RUNS_DIR},${PYDEPS}:${PYDEPS},/tmp:/tmp,${SHM_DIR}:/dev/shm \
  --no-container-mount-home \
  bash -c '
    set -e
    export PYTHONPATH='"${PYDEPS}"':'"${PROJECT_DIR}"':$PYTHONPATH
    export TMPDIR=/tmp
    export MPLBACKEND=Agg
    export TRANSFORMERS_NO_FLASH_ATTN_2=1
    export TRANSFORMERS_CACHE=/tmp/transformers_cache
    mkdir -p $TRANSFORMERS_CACHE

    echo "=== [fix] ensuring complete rich + termcolor in PYDEPS ==="
    # Reinstall a complete rich (the PYDEPS copy was missing progress.py etc).
    pip install --no-cache-dir --target='"${PYDEPS}"' --upgrade \
        "rich>=13,<14" "termcolor>=2,<3" || echo "[fix] pip install failed (offline?); will rely on existing/stub"

    echo "=== [check] import chain ==="
    python -c "import termcolor; from termcolor import colored; print(\"termcolor\", getattr(termcolor,\"__version__\",\"?\"), termcolor.__file__)"
    python -c "import rich, rich.progress; print(\"rich\", rich.__version__, rich.__file__)"
    python -c "from mmengine import Config; print(\"mmengine Config import OK\")"

    cd '"${PROJECT_DIR}"'
    echo "=== [smoke] running V3 dataset smoke test ==="
    python /iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs/smoke_event_v3.py
  '
